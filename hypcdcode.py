import math
import random
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


# =========================
# 1. CONFIG
# =========================
@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset split: known vs unknown classes
    known_classes: tuple = (0, 1, 2, 3, 4)
    unknown_classes: tuple = (5, 6, 7, 8, 9)
    num_total_classes: int = 10

    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 2

    emb_dim: int = 128
    hidden_dim: int = 256

    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 15
    pseudo_start_epoch: int = 6
    pseudo_conf_threshold: float = 0.55

    temperature: float = 0.2
    lambda_cls: float = 1.0
    lambda_euc: float = 1.0
    lambda_hyp: float = 1.0
    lambda_pseudo: float = 0.5

    # Quick demo mode
    train_subset: int = 10000
    test_subset: int = 3000


cfg = Config()


# =========================
# 2. REPRODUCIBILITY
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.seed)


# =========================
# 3. HYPERBOLIC OPS
# =========================
EPS = 1e-6


def artanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -1 + EPS, 1 - EPS)
    return 0.5 * torch.log((1 + x) / (1 - x))


class AdaptiveCurvature(nn.Module):
    def __init__(self, init_c: float = 1.0):
        super().__init__()
        self.raw_c = nn.Parameter(torch.tensor([init_c], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return F.softplus(self.raw_c) + 1e-4


def project_to_ball(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    maxnorm = (1.0 - 1e-3) / torch.sqrt(c)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def expmap0(u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    sqrt_c = torch.sqrt(c)
    u_norm = torch.norm(u, dim=-1, keepdim=True).clamp_min(EPS)
    gamma = torch.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project_to_ball(gamma, c)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    diff2 = torch.sum((x - y) ** 2, dim=-1, keepdim=True)

    num = 2.0 * c * diff2
    den = (1.0 - c * x2).clamp_min(EPS) * (1.0 - c * y2).clamp_min(EPS)
    z = 1.0 + num / den
    z = torch.clamp(z, min=1.0 + EPS)
    return torch.acosh(z) / torch.sqrt(c)


def pairwise_poincare_distance(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    xi = x.unsqueeze(1).expand(n, n, -1)
    xj = x.unsqueeze(0).expand(n, n, -1)
    d = poincare_distance(xi, xj, c).squeeze(-1)
    return d


# =========================
# 4. DATASET
# =========================
class CIFAR10GCD(Dataset):
    """
    Returns:
        img1, img2, label, is_labeled, index
    img1/img2 are two augmented views of the same sample.
    """
    def __init__(self, train=True, subset_size=None):
        self.train = train

        self.transform1 = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.transform2 = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.base_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
        ])

        self.dataset = datasets.CIFAR10(root="./data", train=train, download=True)

        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        if subset_size is not None:
            indices = indices[:subset_size]

        self.samples = []
        for idx in indices:
            img, label = self.dataset[idx]
            is_labeled = int(label in cfg.known_classes)
            self.samples.append((img, label, is_labeled, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label, is_labeled, original_idx = self.samples[idx]
        img1 = self.transform1(img) if self.train else self.base_transform(img)
        img2 = self.transform2(img) if self.train else self.base_transform(img)
        return img1, img2, torch.tensor(label), torch.tensor(is_labeled), torch.tensor(idx)


# =========================
# 5. MODEL
# =========================
class ImprovedHypCD(nn.Module):
    def __init__(self, emb_dim=128, num_classes=10):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, emb_dim),
        )

        self.classifier = nn.Linear(emb_dim, num_classes)
        self.curvature = AdaptiveCurvature(init_c=1.0)

    def encode(self, x: torch.Tensor):
        feats = self.backbone(x)
        z_euc = self.projector(feats)
        z_euc = F.normalize(z_euc, dim=-1)
        c = self.curvature()
        z_hyp = expmap0(z_euc, c)
        logits = self.classifier(z_euc)
        return feats, z_euc, z_hyp, logits, c

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        feats1, z_euc1, z_hyp1, logits1, c = self.encode(x1)
        feats2, z_euc2, z_hyp2, logits2, _ = self.encode(x2)

        return {
            "feats1": feats1,
            "z_euc1": z_euc1,
            "z_hyp1": z_hyp1,
            "logits1": logits1,
            "feats2": feats2,
            "z_euc2": z_euc2,
            "z_hyp2": z_hyp2,
            "logits2": logits2,
            "c": c,
        }


# =========================
# 6. LOSSES
# =========================
def supervised_loss(logits, labels, is_labeled):
    if is_labeled.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(logits[is_labeled], labels[is_labeled])


def supervised_contrastive_loss(features, labels, temperature=0.2):
    """
    features: [B, D], labels: [B]
    """
    device = features.device
    features = F.normalize(features, dim=1)
    sim = torch.matmul(features, features.T) / temperature

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    pos_per_row = mask.sum(dim=1)
    valid_rows = pos_per_row > 0
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_per_row + 1e-8)

    if valid_rows.sum() == 0:
        return torch.tensor(0.0, device=device)

    return -mean_log_prob_pos[valid_rows].mean()


def hyperbolic_contrastive_loss(z_hyp, labels, c, temperature=0.2):
    device = z_hyp.device
    d = pairwise_poincare_distance(z_hyp, c)
    sim = -d / temperature

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    pos_per_row = mask.sum(dim=1)
    valid_rows = pos_per_row > 0
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_per_row + 1e-8)

    if valid_rows.sum() == 0:
        return torch.tensor(0.0, device=device)

    return -mean_log_prob_pos[valid_rows].mean()


# =========================
# 7. PSEUDO LABELING
# =========================
@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_z = []
    all_y = []
    all_lab = []

    for x1, x2, y, is_labeled, idx in loader:
        x1 = x1.to(device)
        _, z_euc, z_hyp, logits, c = model.encode(x1)
        all_z.append(z_euc.cpu())
        all_y.append(y.cpu())
        all_lab.append(is_labeled.cpu())

    return (
        torch.cat(all_z).numpy(),
        torch.cat(all_y).numpy(),
        torch.cat(all_lab).numpy(),
    )


def run_kmeans(features, n_clusters):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=cfg.seed)
    pred = km.fit_predict(features)
    centers = km.cluster_centers_
    return pred, centers


def compute_cluster_confidence(features, centers, pred):
    dists = np.linalg.norm(features - centers[pred], axis=1)
    dmin = dists.min()
    dmax = dists.max()
    conf = 1.0 - (dists - dmin) / (dmax - dmin + 1e-8)
    return conf


def build_pseudo_label_dict(model, loader, device):
    features, labels, is_labeled = extract_embeddings(model, loader, device)
    pred, centers = run_kmeans(features, cfg.num_total_classes)
    conf = compute_cluster_confidence(features, centers, pred)

    pseudo_dict = {}
    for i in range(len(pred)):
        if is_labeled[i] == 1:
            pseudo_dict[i] = (int(labels[i]), 1.0, True)
        else:
            pseudo_dict[i] = (int(pred[i]), float(conf[i]), False)
    return pseudo_dict, pred, labels


# =========================
# 8. METRICS
# =========================
def clustering_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for yp, yt in zip(y_pred, y_true):
        w[yp, yt] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / len(y_true)


def evaluate_clustering(model, loader, device):
    features, labels, _ = extract_embeddings(model, loader, device)
    pred, _ = run_kmeans(features, cfg.num_total_classes)

    acc = clustering_accuracy(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    ari = adjusted_rand_score(labels, pred)
    return acc, nmi, ari


# =========================
# 9. TRAINING
# =========================
def train():
    device = cfg.device
    print("Using device:", device)

    train_dataset = CIFAR10GCD(train=True, subset_size=cfg.train_subset)
    test_dataset = CIFAR10GCD(train=False, subset_size=cfg.test_subset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    eval_train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = ImprovedHypCD(
        emb_dim=cfg.emb_dim,
        num_classes=cfg.num_total_classes
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    pseudo_label_dict = None

    for epoch in range(cfg.epochs):
        model.train()
        total_loss_meter = 0.0

        # build pseudo-labels after warmup
        if epoch >= cfg.pseudo_start_epoch:
            pseudo_label_dict, _, _ = build_pseudo_label_dict(model, eval_train_loader, device)
            print(f"\n[Epoch {epoch+1}] Pseudo labels updated.")

        for x1, x2, y, is_labeled, idx in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            is_labeled = is_labeled.to(device).bool()
            idx = idx.numpy()

            out = model(x1, x2)

            # classification on labeled samples
            cls_loss = supervised_loss(out["logits1"], y, is_labeled)

            # create training labels for contrastive part
            contrast_labels = y.clone()

            # apply pseudo-labels to confident unlabeled samples
            pseudo_mask = torch.zeros_like(is_labeled, dtype=torch.bool, device=device)
            if pseudo_label_dict is not None:
                for b in range(len(idx)):
                    sample_idx = int(idx[b])
                    pseudo_lbl, conf, is_true_labeled = pseudo_label_dict[sample_idx]
                    if (not is_true_labeled) and conf >= cfg.pseudo_conf_threshold:
                        contrast_labels[b] = pseudo_lbl
                        pseudo_mask[b] = True

            # known labels + confident pseudo labels
            usable_mask = is_labeled | pseudo_mask

            if usable_mask.sum() > 1:
                z_euc = torch.cat([out["z_euc1"][usable_mask], out["z_euc2"][usable_mask]], dim=0)
                z_hyp = torch.cat([out["z_hyp1"][usable_mask], out["z_hyp2"][usable_mask]], dim=0)
                labels_for_contrast = torch.cat([contrast_labels[usable_mask], contrast_labels[usable_mask]], dim=0)

                euc_loss = supervised_contrastive_loss(z_euc, labels_for_contrast, cfg.temperature)
                hyp_loss = hyperbolic_contrastive_loss(z_hyp, labels_for_contrast, out["c"], cfg.temperature)
            else:
                euc_loss = torch.tensor(0.0, device=device)
                hyp_loss = torch.tensor(0.0, device=device)

            # small pseudo classification regularization
            if pseudo_mask.sum() > 0:
                pseudo_cls_loss = F.cross_entropy(out["logits1"][pseudo_mask], contrast_labels[pseudo_mask])
            else:
                pseudo_cls_loss = torch.tensor(0.0, device=device)

            loss = (
                cfg.lambda_cls * cls_loss
                + cfg.lambda_euc * euc_loss
                + cfg.lambda_hyp * hyp_loss
                + cfg.lambda_pseudo * pseudo_cls_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_meter += loss.item()

        avg_loss = total_loss_meter / len(train_loader)
        curvature_value = model.curvature().item()

        train_acc, train_nmi, train_ari = evaluate_clustering(model, eval_train_loader, device)
        test_acc, test_nmi, test_ari = evaluate_clustering(model, test_loader, device)

        print(
            f"\nEpoch [{epoch+1}/{cfg.epochs}] "
            f"Loss={avg_loss:.4f} "
            f"Curvature={curvature_value:.4f}"
        )
        print(f"Train: ACC={train_acc:.4f} NMI={train_nmi:.4f} ARI={train_ari:.4f}")
        print(f"Test : ACC={test_acc:.4f} NMI={test_nmi:.4f} ARI={test_ari:.4f}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()

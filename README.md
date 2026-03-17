# 🚀 Improved Hyperbolic Category Discovery (HypCD)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![License](https://img.shields.io/badge/License-Educational-green)
![Status](https://img.shields.io/badge/Status-Active-success)

An improved implementation of **Hyperbolic Category Discovery** for discovering known and unknown classes using **hyperbolic geometry + contrastive learning**.

---

## 📌 Overview

This project enhances the original HypCD framework by introducing:

- ✅ Adaptive curvature learning  
- ✅ Hybrid Euclidean + Hyperbolic contrastive learning  
- ✅ Pseudo-label refinement  
- ✅ Strong data augmentation  
- ✅ Training visualization (loss & metrics)  
- ✅ t-SNE embedding visualization  

---

## 🧠 Key Idea

- **Euclidean space** → captures local similarity  
- **Hyperbolic space** → captures hierarchical relationships  

👉 Combining both improves category discovery performance.

---

## 🏗️ Project Structure

```
improved-hypcd/
│
├── improved_hypcd_v2.py   # Main training script
├── outputs/               # Generated plots & results
├── data/                  # Dataset (auto-downloaded)
└── README.md
```

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install torch torchvision scikit-learn scipy matplotlib
```

---

## ▶️ How to Run

```bash
python improved_hypcd_v2.py
```

---

## 📊 Features

### 🔹 Adaptive Curvature
Learns curvature of hyperbolic space dynamically for better hierarchy modeling.

### 🔹 Hybrid Learning
- Euclidean contrastive loss  
- Hyperbolic contrastive loss  

### 🔹 Pseudo-Label Refinement
Improves learning from unlabeled data using clustering.

### 🔹 Data Augmentation
- Random crop  
- Flip  
- Color jitter  
- Grayscale  

### 🔹 Visualization
- 📉 Loss curve  
- 📊 Metrics (ACC, NMI, ARI)  
- 🔍 t-SNE plots  

---

## 📈 Evaluation Metrics

- **ACC (Clustering Accuracy)**
- **NMI (Normalized Mutual Information)**
- **ARI (Adjusted Rand Index)**

---

## 📸 Results (Example)

Add your screenshots in the `outputs/` folder and update paths below.

```
outputs/
 ├── loss_plot.png
 ├── metrics_plot.png
 └── tsne_plot.png
```

Example display:

```markdown
![Loss](outputs/loss_plot.png)
![Metrics](outputs/metrics_plot.png)
![t-SNE](outputs/tsne_plot.png)
```

---

## 🧪 Dataset

- CIFAR-10 (automatically downloaded)
- Split into:
  - Known classes (labeled)
  - Unknown classes (unlabeled)

---

## 🔄 Training Pipeline

1. Load dataset & apply augmentations  
2. Extract features using ResNet18  
3. Project to embedding space  
4. Map to hyperbolic space  
5. Apply hybrid losses  
6. Generate pseudo-labels  
7. Train with refined labels  
8. Evaluate using clustering  
9. Visualize results  

---

## 📌 Improvements Over Original HypCD

| Feature | Original | Improved |
|--------|--------|--------|
| Hyperbolic Learning | ✅ | ✅ |
| Adaptive Curvature | ❌ | ✅ |
| Hybrid Learning | ❌ | ✅ |
| Pseudo-label refinement | ❌ | ✅ |
| Visualization | ❌ | ✅ |

---

## 🎯 Use Cases

- Open-set recognition  
- Semi-supervised learning  
- Image clustering  
- Hierarchical representation learning  

---

## 🔮 Future Work

- Transformer backbone (ViT / DINO)
- Larger datasets (ImageNet, iNaturalist)
- Advanced clustering methods
- Poincaré disk visualization

---

## 👨‍💻 Author

**Ansh Kumar Sharma**

---

## ⭐ Support

If you find this useful, consider giving it a ⭐ on GitHub!

---

## 📜 License

This project is intended for **educational and research purposes**.

---

## 🔥 Pro Tip

For best results:
- Use GPU  
- Increase epochs (30+)  
- Use full dataset instead of subset  

# Product Recognition & Classification

Two computer-vision projects on the [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset) by Markus Klasson: a collection of **81 fine-grained product classes** grouped into **43 coarse categories** (fruits, dairy, juices, …).

## 1. Product Classification with CNN

**Directory:** `product_classification_with_CNN/`  
**Notebook (report):** [`Product_Classification_with_CNN.ipynb`](Product_Classification_with_CNN.ipynb)

Supervised classification trained end-to-end on the Grocery Store Dataset. The project is split into two tasks:

| Task       | Models          | Description                                                                                                                                     |
| ---------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task 1** | Model 1–5       | Custom CNNs of increasing complexity — from a plain 3-block network to a multi-task head that jointly predicts the 43 coarse and 81 fine labels |
| **Task 2** | ResNet-18 (a/b) | Transfer learning: (a) frozen backbone used as feature extractor, (b) full fine-tuning                                                          |

**Run Task 1:**

```bash
python product_classification_with_CNN/task1/main_1.py
```

**Run Task 2:**

```bash
python product_classification_with_CNN/task2/main_2.py
```

## 2. Product Recognition with Nearest Neighbour Search

**Directory:** `product_recognition_with_NNS/`  
**Notebook (report):** [`Product_Recognition_with_NNS.ipynb`](Product_Recognition_with_NNS.ipynb)

Few-shot recognition pipeline: given a single prototype image per class (gallery), classify query images via embedding-space nearest-neighbour search — no training required.

**Backbones evaluated:**

- ResNet-18 (512-D embeddings)
- ResNet-50 (2048-D embeddings)
- DINOv2-Small (384-D, CLS token)
- DINOv3-Small (384-D, CLS token)

**Distance metrics:**

- Cosine similarity
- Mahalanobis distance with PCA dimensionality reduction (50 components)

**Gallery modes:** `all` (all prototype embeddings) · `mean` (per-class average)

**Run:**

```bash
python product_recognition_with_NNS/main.py
```

## Dataset

[Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset) — 81 fine-grained product classes organised in 43 coarse categories.  
Pre-split into `train` / `val` / `test` sets; class mappings in `product_classification_with_CNN/dataset/classes.csv`.

## Repository Structure

```
ProgettoCV/
├── Product_Classification_with_CNN.ipynb   ← full report for project 1
├── Product_Recognition_with_NNS.ipynb      ← full report for project 2
├── product_classification_with_CNN/
│   ├── task1/          (main_1.py, models_1.py, config_1.py)
│   ├── task2/          (main_2.py, models_2.py, config_2.py)
│   ├── dataset/
│   ├── logs.py
│   └── plotting.py
├── product_recognition_with_NNS/
│   ├── main.py
│   ├── models.py
│   ├── config.py
│   ├── logs.py
│   └── nn_search_dataset/  (prototypes/, queries/)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

> GPU acceleration requires CUDA 12.x and cuDNN 9.x (pinned in `requirements.txt`).

Main dependencies: TensorFlow 2.21, Keras 3.14, keras-hub, scikit-learn, NumPy, Pandas, Matplotlib.

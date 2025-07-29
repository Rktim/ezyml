<div align="center">

# 📦 ezyml 🚀

From raw data to a trained model — in just one line of code.

<a href="https://pypi.org/project/ezyml/">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/ezyml?color=blue&label=PyPI&logo=pypi">
</a>
<a href="https://github.com/Rktim/ezyml/blob/main/LICENSE">
  <img alt="License" src="https://img.shields.io/github/license/Rktim/ezyml?color=blue">
</a>
<img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/ezyml?logo=python&logoColor=white">
  
  
[![PyPI Downloads](https://static.pepy.tech/badge/ezyml)](https://pepy.tech/projects/ezyml)

</div>




---

## 🌟 Why ezyml?

**ezyml** is a lightweight, high-level Python library and CLI tool that automates the most tedious parts of your ML pipeline — so you can focus on what matters. Whether you're building a classifier, a regressor, or just exploring data, ezyml does the heavy lifting.

### ✅ Key Features

* 🪄 **Auto-Pilot Mode** – Detects task type (classification, regression, etc.) automatically.
* 🧹 **Smart Preprocessing** – Handles missing values, encodes categories, and scales features out of the box.
* 🧰 **20+ Models** – Pre-integrated models from `scikit-learn` and `xgboost`.
* 💾 **One-Line Export** – Save your model as `.pkl` and performance report as `.json`.
* 📉 **Dimensionality Reduction** – Easily visualize data using PCA or t-SNE.
* 🧪 **Dual Interface** – Use as a Python package *or* from the command line.

---

## 📦 Installation

Install via pip:

```bash
pip install ezyml
```

---

## 🚀 CLI Quickstart

### 🧠 Train a Classifier

```bash
ezyml train \
  --data titanic.csv \
  --target Survived \
  --model extra_trees \
  --output titanic_model.pkl
```

### 📈 Train a Regressor

```bash
ezyml train \
  --data housing.csv \
  --target price \
  --model ridge \
  --output house_price_model.pkl
```

### 📉 Run PCA

```bash
ezyml reduce \
  --data features.csv \
  --model pca \
  --components 2 \
  --output pca_data.csv
```

---

## 🧪 Python API Example

### ▶️ Classification

```python
from ezyml import EZTrainer

# 1. Initialize
trainer = EZTrainer(data='heart.csv', target='label', model='naive_bayes')

# 2. Train
trainer.train()

# 3. Save Results
trainer.save_model('heart_model.pkl')
trainer.save_report('heart_report.json')
```

### 🔍 Dimensionality Reduction (PCA)

```python
from ezyml import EZTrainer

pca_trainer = EZTrainer(
    data='high_dim.csv',
    model='pca',
    task='dim_reduction',
    n_components=2
)

pca_trainer.train()
pca_trainer.save_transformed('pca_output.csv')
```

---

## 🧰 Supported Models

| Task                         | Models                                                                                                             |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Classification**           | `logistic_regression`, `random_forest`, `xgboost`, `svm`, `naive_bayes`, `gradient_boosting`, `extra_trees`, `knn` |
| **Regression**               | `linear_regression`, `ridge`, `lasso`, `elasticnet`, `random_forest`, `xgboost`, `svr`, `gradient_boosting`        |
| **Clustering**               | `kmeans`, `dbscan`, `agglo` (Agglomerative Clustering)                                                             |
| **Dimensionality Reduction** | `pca`, `tsne`                                                                                                      |

---

## 📜 License

MIT License – [View License](https://github.com/Rktim/ezyml/blob/main/LICENSE)

---

## 👨‍💻 Author

Built with ❤️ by [Raktim Kalita](https://github.com/Rktim)

---

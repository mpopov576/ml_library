````markdown
# ml_library

A modular **machine learning library** built from scratch using **NumPy** — no `scikit-learn` under the hood.

## 🚀 Implemented Algorithms

### Supervised Learning

- **Linear Regression**
  - Supports regularization techniques.
  - Suitable for continuous value prediction.

- **Logistic Regression**
  - Supports both binary and multiclass classification.
  - Uses gradient-based optimization.

- **K-Nearest Neighbors (KNN)**
  - Configurable distance metrics.
  - Simple instance-based learning algorithm.

- **Naive Bayes**
  - Probabilistic classifier based on Bayes' theorem.
  - Efficient for high-dimensional datasets.

- **Support Vector Machine (SVM)**
  - Supports custom kernel functions.
  - Effective for linear and non-linear classification.

- **Decision Tree**
  - Recursive feature splitting.
  - Configurable split criteria and stopping conditions.

- **Random Forest**
  - Ensemble of multiple decision trees.
  - Improves accuracy and reduces overfitting.

- **AdaBoost**
  - Adaptive boosting ensemble method.
  - Sequentially focuses on difficult training examples.

---

### Unsupervised Learning

- **K-Means**
  - Centroid-based clustering algorithm.
  - Iteratively minimizes within-cluster variance.

- **Hierarchical Clustering**
  - Agglomerative clustering approach.
  - Builds a hierarchy of clusters.

---

## 🛠 Utilities

- **`kernels.py`**
  - Custom kernel functions for SVM and other kernel-based algorithms.

- **`metrics.py`**
  - Evaluation metrics for classification and regression.

- **`model_selection.py`**
  - Cross-validation and hyperparameter tuning utilities.

- **`feature_extraction/text.py`**
  - TF-IDF vectorization for NLP tasks.

- **`stats.py`**
  - Statistical helper functions used throughout the library.

---

## 📦 Installation

```bash
git clone https://github.com/mpopov576/ml_library.git
cd ml_library
````

```
```

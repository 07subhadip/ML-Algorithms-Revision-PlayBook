# 📘 ML Algorithms Revision PlayBook

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-success)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)

Welcome to the **ML Algorithms Revision PlayBook**. This repository is designed as a **comprehensive guide** for mastering Machine Learning algorithms, providing a structured approach to theoretical understanding and practical implementation.

The goal is to facilitate an **end-to-end revision** by applying each algorithm on **random Kaggle datasets**, ensuring a balance between **conceptual clarity** and **hands-on proficiency**.

---

## � Table of Contents

- [How to Use This PlayBook](#-how-to-use-this-playbook)
- [Repository Structure](#-repository-structure)
- [1. Supervised Learning – Regression](#1-supervised-learning--regression-algorithms)
- [2. Supervised Learning – Classification](#2-supervised-learning--classification-algorithms)
- [3. Unsupervised Learning](#3-unsupervised-learning-algorithms)
- [4. Semi-Supervised Learning](#4-semi-supervised-learning)
- [5. Anomaly & Outlier Detection](#5-anomaly--outlier-detection)
- [6. Time Series Algorithms](#6-time-series-algorithms)
- [7. Model Evaluation & Validation](#7-model-evaluation--validation-techniques)
- [8. Feature Engineering & Preprocessing](#8-feature-engineering--preprocessing)
- [9. Model Explainability & Interpretation](#9-model-explainability--interpretation)
- [Practical Workflow](#-practical-workflow-for-each-algorithm)
- [Prerequisites](#-prerequisites)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📌 How to Use This PlayBook

For **each algorithm**, we follow a rigorous practice regime:

1.  **Theory**: Understand core intuition and mathematical formulation.
2.  **Assumptions**: Know the strengths, weaknesses, and underlying assumptions.
3.  **Scratch Implementation**: Code the algorithm from scratch (where feasible) to deepen understanding.
4.  **Library Implementation**: Implement using standard libraries like `scikit-learn`.
5.  **Evaluation**: Assess performance using appropriate metrics.
6.  **Real-world Application**: Apply on at least **one real-world Kaggle dataset**.

---

## 📂 Repository Structure

The repository is organized to keep algorithms and projects structured:

```text
ML-Algorithms-Revision-PlayBook/
├── data/                        # Datasets (gitignored if large)
├── notebooks/                   # Main working directory for notebooks
│   ├── 01_Regression/           # Linear, Ridge, Lasso, etc.
│   ├── 02_Classification/       # Logistic, KNN, SVM, etc.
│   ├── 03_Clustering/           # K-Means, DBSCAN, etc.
│   ├── 04_DimensionalityRed/    # PCA, t-SNE, etc.
│   ├── 05_TimeSeries/           # ARIMA, Prophet, etc.
│   └── 06_DeepLearning/         # Neural Networks (optional/future)
├── scripts/                     # Helper python scripts (data_loader.py, etc.)
├── images/                      # Images for README and notebooks
├── .gitignore                   # Files to ignore (e.g., venv, large data)
├── LICENSE                      # Apache 2.0 License
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

---

## 1️⃣ Supervised Learning – Regression Algorithms

### 1.1 Linear Models

- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)
- Elastic Net Regression

**🔑 Key Concepts:**

- Cost functions (MSE, MAE)
- Gradient Descent
- Bias–Variance Tradeoff
- Regularization intuition

### 1.2 Tree-Based Regression

- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor

**🔑 Key Concepts:**

- Entropy vs Variance Reduction
- Overfitting in trees
- Feature importance

### 1.3 Ensemble & Boosting (Regression)

- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

**🔑 Key Concepts:**

- Boosting vs Bagging
- Learning rate vs number of estimators
- Overfitting control

---

## 2️⃣ Supervised Learning – Classification Algorithms

### 2.1 Linear & Probabilistic Models

- Logistic Regression
- Naive Bayes (Gaussian, Multinomial, Bernoulli)

**🔑 Key Concepts:**

- Sigmoid & Softmax functions
- Log Loss
- Bayes Theorem assumptions

### 2.2 Distance-Based & Margin-Based Models

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
  - Linear SVM
  - Kernel SVM (RBF, Polynomial)

**🔑 Key Concepts:**

- Distance metrics (Euclidean, Manhattan)
- Curse of dimensionality
- Kernel trick
- Margin maximization

### 2.3 Tree-Based Classification

- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier

**🔑 Key Concepts:**

- Gini Index vs Entropy
- Feature selection
- Tree depth control

### 2.4 Ensemble & Boosting (Classification)

- AdaBoost
- Gradient Boosting Classifier
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier

**🔑 Key Concepts:**

- Weak learners vs Strong learners
- Bias reduction
- Handling class imbalance

---

## 3️⃣ Unsupervised Learning Algorithms

### 3.1 Clustering

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Mean Shift

**🔑 Key Concepts:**

- Elbow method & Silhouette Score
- Distance metrics
- Density-based clustering

### 3.2 Dimensionality Reduction

- Principal Component Analysis (PCA)
- Kernel PCA
- Linear Discriminant Analysis (LDA)
- t-SNE
- UMAP

**🔑 Key Concepts:**

- Variance maximization
- Eigenvalues & eigenvectors
- Visualization vs feature reduction

### 3.3 Association Rule Learning

- Apriori Algorithm
- FP-Growth Algorithm

**🔑 Key Concepts:**

- Support, Confidence, Lift

---

## 4️⃣ Semi-Supervised Learning

- Label Propagation
- Label Spreading

**🔑 Key Concepts:**

- Graph-based learning
- Handling small labeled datasets

---

## 5️⃣ Anomaly & Outlier Detection

- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)
- Elliptic Envelope

**🔑 Key Concepts:**

- Contamination factor
- Real-world anomaly scenarios

---

## 6️⃣ Time Series Algorithms

- AR (AutoRegressive)
- MA (Moving Average)
- ARMA
- ARIMA
- SARIMA
- Prophet

**🔑 Key Concepts:**

- Stationarity (Dickey-Fuller test)
- Seasonality & Trend
- ACF & PACF plots

---

## 7️⃣ Model Evaluation & Validation Techniques

### 7.1 Evaluation Metrics

**Regression:**

- MSE, RMSE, MAE
- R² Score, Adjusted R²

**Classification:**

- Accuracy
- Precision, Recall, F1 Score
- ROC-AUC, Confusion Matrix

### 7.2 Validation Techniques

- Train-Test Split
- K-Fold Cross Validation
- Stratified K-Fold
- Time Series Split

---

## 8️⃣ Feature Engineering & Preprocessing

- **Missing Value Handling:** Mean/Median/Mode imputation, KNN Imputer
- **Outlier Treatment:** IQR, Z-score
- **Encoding:** Label Encoding, One-Hot Encoding, Target Encoding
- **Feature Scaling:** StandardScaler, MinMaxScaler, RobustScaler
- **Feature Selection:** Filter, Wrapper, Embedded methods

---

## 9️⃣ Model Explainability & Interpretation

- Feature Importance
- Permutation Importance
- SHAP (SHapley Additive exPlanations)
- LIME
- Partial Dependence Plot (PDP)

---

## 🔟 Practical Workflow for Each Algorithm

For every project in this playbook, we adhere to the following workflow:

1.  **Dataset Understanding**: Domain knowledge and data dictionary.
2.  **EDA (Exploratory Data Analysis)**: Visualizations and statistical summaries.
3.  **Feature Engineering**: Creation and transformation of variables.
4.  **Model Training**: Baseline and advanced models.
5.  **Hyperparameter Tuning**: GridSearch, RandomizedSearch, or Optuna.
6.  **Evaluation**: Comprehensive metric analysis.
7.  **Interpretation**: Why does the model predict what it predicts?
8.  **Final Conclusion**: Business or research insights.

---

## � Prerequisites

To run the notebooks and examples in this repository, you'll need the following installed:

- Python 3.8+
- Jupyter Notebook / Lab
- Common ML Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib` & `seaborn`
  - `scikit-learn`
  - `xgboost`, `lightgbm`, `catboost`
  - `statsmodels`

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost statsmodels
```

## 🤝 Contributing

Contributions are welcome! If you'd like to add an algorithm, fix a bug, or improve documentation:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

⭐ **If you find this repository useful for your ML revision, please give it a star!**

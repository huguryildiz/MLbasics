# Lab 2 - Classification Analysis

## 📌 Overview
This repository contains **Lab 2 - Classification Analysis**, which implements **logistic regression** and **regularized classification models** using **synthetic data**.

## 📂 Files
- **`Lab2-Classification.ipynb`** → Jupyter Notebook implementing:
  - **Data Generation**: Creating synthetic classification data using `numpy.random.rand()`
  - **Data Preprocessing**: Normalization and splitting into train/test sets
  - **Logistic Regression**: Using `sklearn.linear_model.LogisticRegression`
  - **Regularization (L2)**: Applying Ridge regularization (`sklearn.linear_model.RidgeClassifier`)
  - **Performance Evaluation**: Accuracy, Confusion Matrix, and Decision Boundary

## 🚀 Usage
Run the notebook to generate and visualize synthetic classification data.
Train and compare unregularized vs. regularized logistic regression models.
Adjust the regularization strength (C for Logistic Regression, alpha for Ridge) to observe its effect.

## 📊 Key Results
Logistic Regression without Regularization may overfit.
Regularized Logistic Regression (C < 1 or alpha > 0) improves generalization.
Optimal Regularization minimizes test error.

## 🤝 Contributing
Feel free to fork this repository, make improvements, and submit a pull request! 🚀

## 🔧 Setup & Requirements
To run the notebook, install the required dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
Then, open the notebook:
jupyter notebook Lab2-Classification.ipynb

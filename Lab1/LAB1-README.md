# Lab 1 - Regression Analysis

## 📌 Overview
This repository contains **Lab 1 - Regression Analysis**, which explores **linear regression** and **regularized regression (Ridge)** using **synthetic data**.

## 📂 Files
- **`Lab1-Regression.ipynb`** → Jupyter Notebook implementing:
  - **Data Generation**: Creating synthetic data using `numpy.random.rand()`
  - **Data Preprocessing**: Normalization and splitting into train/test sets
  - **Linear Regression**: Using `sklearn.linear_model.LinearRegression`
  - **Ridge Regression**: Applying L2 regularization using `sklearn.linear_model.Ridge`
  - **Performance Evaluation**: Mean Squared Error (MSE) comparison
    
## 🚀 Usage
Run the notebook to generate and visualize synthetic data.
Train and compare unregularized vs. regularized regression models.
Adjust the regularization strength (alpha) in Ridge regression to observe its effect.

##  📊 Key Results
Unregularized Linear Regression may overfit.
Ridge Regression (alpha > 0) controls overfitting and improves generalization.
Optimal alpha minimizes test error.

## 🤝 Contributing
Feel free to fork this repository, make improvements, and submit a pull request! 🚀

## 🔧 Setup & Requirements
To run the notebook, install the required dependencies:
```bash

pip install numpy pandas scikit-learn matplotlib

# Machine Learning Labs - Regression, Classification & Neural Networks

## 📌 Overview  
This repository contains **five Jupyter Notebooks** covering **Regression, Classification, Neural Networks, Decision Trees, and Machine Learning Best Practices**. Each lab provides **hands-on experiments** with practical implementations and evaluations.

### **Labs Overview:**
- **Lab 1: Regression Analysis** → Linear & Ridge Regression  
- **Lab 2: Classification Analysis** → Logistic Regression & Regularization  
- **Lab 3: Neural Networks** → Multi-layer Feedforward Neural Networks (MLP)  
- **Lab 4: ML Advice & Best Practices** → Bias-Variance Tradeoff, Regularization, and Model Optimization  
- **Lab 5: Decision Trees & Random Forests** → Tree-based classifiers, feature importance, and hyperparameter tuning  

Each lab demonstrates:  
✔ **Data preprocessing & feature engineering**  
✔ **Model training and evaluation**  
✔ **Regularization & hyperparameter tuning**  
✔ **Visualization of results and decision boundaries**  
✔ **Performance analysis using metrics**  

---

## 📂 Files  

### **Lab 1 - Regression Analysis**
📄 **`Lab1-Regression.ipynb`**  
Implements:
- **Linear Regression** (`LinearRegression`)  
- **Ridge Regression** (`Ridge`)  
- **Effect of Regularization (L1 & L2)**  
- **Mean Squared Error (MSE) and Residual Analysis**  

### **Lab 2 - Classification Analysis**
📄 **`Lab2-Classification.ipynb`**  
Implements:
- **Logistic Regression** (`LogisticRegression`)  
- **Ridge Classifier** (`RidgeClassifier`)  
- **Decision boundary visualization**  
- **Performance Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix**  

### **Lab 3 - Neural Networks**
📄 **`Lab3-NeuralNetworks.ipynb`**  
Implements:
- **Multi-layer Feedforward Neural Networks (MLP) with TensorFlow/Keras**  
- **Activation Functions: ReLU, Sigmoid, Softmax**  
- **Backpropagation and Gradient Descent Optimization**  
- **Regularization: L2, Dropout**  
- **Hyperparameter tuning using Grid Search**  

### **Lab 4 - Machine Learning Advice & Best Practices**
📄 **`Lab4-Advice.ipynb`**  
Implements:
- **Bias-Variance Tradeoff & Overfitting Detection**  
- **Learning Curves & Model Generalization**  
- **Regularization Techniques (L1 & L2)**  
- **Hyperparameter tuning for better model performance**  
- **Error Analysis: Diagnosing model weaknesses**  

### **Lab 5 - Decision Trees & Random Forests**
📄 **`Lab5-DecisionTrees.ipynb`**  
Implements:
- **Decision Tree Classifier (`DecisionTreeClassifier`)**  
- **Splitting Criteria: Gini vs. Entropy**  
- **Tree Visualization (`plot_tree`)**  
- **Feature Importance Analysis**  
- **Hyperparameter Tuning for Decision Trees**  
- **Random Forest (`RandomForestClassifier`) vs. Single Tree Performance**  

---

## 🔧 Setup & Requirements  
To run these notebooks, install the required dependencies:  
```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow xgboost

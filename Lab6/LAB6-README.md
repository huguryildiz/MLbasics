# Lab 6 - Unsupervised Learning & Deep Reinforcement Learning

## 📌 Overview  
This repository contains **Lab 6 - Unsupervised Learning and Deep Reinforcement Learning**, where we explore key unsupervised machine learning techniques like **clustering** and **dimensionality reduction**, alongside implementing a **Deep Q-Network (DQN)** agent to solve a simple 1D grid environment.

---

## 📂 Files  
**Lab6_Unsupervised_Learning.ipynb** → Jupyter Notebook implementing:  
- **K-Means Clustering**: Automatically grouping similar data points  
- **Elbow Method**: Choosing the optimal number of clusters  
- **Silhouette Analysis**: Evaluating clustering quality  
- **Principal Component Analysis (PCA)**: Reducing feature dimensions  
- **t-SNE Visualization**: Mapping high-dimensional data to 2D for interpretation  
- **Deep Q-Network (DQN)**: Neural network-based Q-learning agent  
- **Training a DQN Agent**: Learning optimal actions in a simple 1D grid world  
- **ε-Greedy Policy**: Balancing exploration and exploitation during learning  
- **Bellman Equation Targets**: Updating Q-values using deep learning  
- **Reward Tracking and Performance Visualization**

---

## 🚀 Usage  
Run the notebook to:  
- **Cluster unlabeled data** and visualize high-dimensional structures.  
- **Train a DQN agent** to efficiently navigate a simple environment using learned Q-values.  
- **Visualize agent's learning progress** episode-by-episode.

---

## 📊 Key Results  
- **K-Means successfully identifies natural groups** within data without supervision.  
- **PCA and t-SNE reveal meaningful low-dimensional structures** hidden in high-dimensional data.  
- **Deep Q-Network enables the agent to learn optimal navigation strategies** through reinforcement and neural function approximation.  
- **Training curves show increasing cumulative rewards over episodes**, indicating learning convergence.

---

## 🤝 Contributing  
Feel free to **fork** this repository, suggest improvements, and submit a **pull request**! 🚀

---

## 🔧 Setup & Requirements  
To run the notebook, install the required dependencies:  
```bash
pip install numpy pandas scikit-learn matplotlib seaborn torch

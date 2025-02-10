# GRADIENT DESCENT - SIMPLE EXAMPLE

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate synthetic data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 3 * x + 7 + np.random.randn(100) * 2  # y = 3x + 7 + noise

# Initialize parameters
w = np.random.randn()
b = np.random.randn()

# Hyperparameters
learning_rate = 0.01
iterations = 1000

# Gradient Descent
cost_history = []
w_history = []
b_history = []
for i in range(iterations):
    y_pred = w * x + b
    error = y_pred - y
    cost = np.mean(error ** 2)  # Mean Squared Error
    cost_history.append(cost)
    w_history.append(w)
    b_history.append(b)

    # Compute gradients
    dw = (2 / len(x)) * np.sum(error * x)
    db = (2 / len(x)) * np.sum(error)

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    # Print status every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}: Cost = {cost:.4f}, w = {w:.4f}, b = {b:.4f}")

# Final parameter values
print(f"Cost = {cost:.4f}, Final parameters: w = {w:.4f}, b = {b:.4f}")

# Plot cost function in 3D
w_values = np.linspace(w - 2, w + 2, 50)
b_values = np.linspace(b - 2, b + 2, 50)
J_values = np.zeros((50, 50))

for i in range(50):
    for j in range(50):
        w_temp = w_values[i]
        b_temp = b_values[j]
        y_pred_temp = w_temp * x + b_temp
        J_values[i, j] = np.mean((y_pred_temp - y) ** 2)

W, B = np.meshgrid(w_values, b_values)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, J_values, cmap='turbo', alpha=1)
ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('Cost J(w,b)')
ax.set_title('Cost Function Surface')
ax.view_init(elev=25, azim=135)  # Adjust these values for a better perspective
# Plot gradient descent path
ax.scatter(w_history, b_history, cost_history, color='red', marker='o', s=5, label='Gradient Descent Path')
ax.legend()

plt.show()

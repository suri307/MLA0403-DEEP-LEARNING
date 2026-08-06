import numpy as np
import matplotlib.pyplot as plt

# Generate Sample Data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add Bias Term
X_b = np.c_[np.ones((100, 1)), X]

# Initialize Parameters
theta = np.random.randn(2, 1)
learning_rate = 0.1
iterations = 1000
m = len(X)

# Gradient Descent
for i in range(iterations):
    gradients = (2 / m) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

# Predictions
y_pred = X_b.dot(theta)

# Print Parameters
print("Intercept:", theta[0][0])
print("Slope:", theta[1][0])

# Plot Results
plt.scatter(X, y, color="blue", label="Data Points")
plt.plot(X, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()

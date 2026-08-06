import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Sample Data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X, y)

# Generate Values for Sigmoid Curve
X_new = np.linspace(1, 10, 200).reshape(-1, 1)
y_prob = model.predict_proba(X_new)[:, 1]

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', label='Data Points')
plt.plot(X_new, y_prob, color='blue', linewidth=2, label='Sigmoid Curve')
plt.xlabel("Input")
plt.ylabel("Probability")
plt.title("Logistic Regression - Sigmoid Function")
plt.legend()
plt.grid(True)
plt.show()

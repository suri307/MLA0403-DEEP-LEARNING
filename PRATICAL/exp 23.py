import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate spiral data
n = 500

theta = np.linspace(0, 4 * np.pi, n)
r = np.linspace(0.2, 2.5, n)

# Class 0 spiral
x1 = r * np.cos(theta)
y1 = r * np.sin(theta)

# Class 1 spiral
x2 = r * np.cos(theta + np.pi)
y2 = r * np.sin(theta + np.pi)

# Combine the data
X = np.vstack((
    np.column_stack((x1, y1)),
    np.column_stack((x2, y2))
))

y = np.vstack((
    np.zeros((n, 1)),
    np.ones((n, 1))
))

# Add small noise
X = X + np.random.randn(*X.shape) * 0.05

# Shuffle data
index = np.random.permutation(len(X))

X = X[index]
y = y[index]

# Split into training and testing data
split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# Neural Network
W1 = np.random.randn(2, 32) * 0.5
b1 = np.zeros((1, 32))

W2 = np.random.randn(32, 16) * 0.5
b2 = np.zeros((1, 16))

W3 = np.random.randn(16, 1) * 0.5
b3 = np.zeros((1, 1))

learning_rate = 0.5
epochs = 1500

loss_history = []


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# Training
for epoch in range(epochs):

    # Forward propagation
    Z1 = X_train @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    Z3 = A2 @ W3 + b3
    A3 = sigmoid(Z3)

    # Binary Cross-Entropy loss
    loss = -np.mean(
        y_train * np.log(A3 + 1e-8) +
        (1 - y_train) * np.log(1 - A3 + 1e-8)
    )

    loss_history.append(loss)

    # Backpropagation
    dZ3 = A3 - y_train

    dW3 = A2.T @ dZ3 / len(X_train)
    db3 = np.mean(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * A2 * (1 - A2)

    dW2 = A1.T @ dZ2 / len(X_train)
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * A1 * (1 - A1)

    dW1 = X_train.T @ dZ1 / len(X_train)
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # Update weights
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


# Testing
A1 = sigmoid(X_test @ W1 + b1)
A2 = sigmoid(A1 @ W2 + b2)
output = sigmoid(A2 @ W3 + b3)

# Convert probabilities to class labels
predicted = (output >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predicted == y_test)

print("Experiment 23")
print("Neural Network Analysis for Spiral Data")
print("Activation Function: Sigmoid")
print("Test Accuracy:", accuracy * 100, "%")


# Plot loss
plt.plot(loss_history)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Spiral Data - Sigmoid Activation")

plt.show()


# Plot spiral classification
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test.ravel()
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Spiral Data Classification")

plt.show()

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate three-class data
n = 300

# Class 1
X1 = np.random.randn(n, 2) * 0.6 + np.array([-2, -1])

# Class 2
X2 = np.random.randn(n, 2) * 0.6 + np.array([2, -1])

# Class 3
X3 = np.random.randn(n, 2) * 0.6 + np.array([0, 2])

# Combine data
X = np.vstack((X1, X2, X3))

# One-hot encoded labels
y = np.vstack((
    np.tile([1, 0, 0], (n, 1)),
    np.tile([0, 1, 0], (n, 1)),
    np.tile([0, 0, 1], (n, 1))
))

# Shuffle data
index = np.random.permutation(len(X))

X = X[index]
y = y[index]

# Split into training and testing
split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# Neural Network
W1 = np.random.randn(2, 24) * 0.5
b1 = np.zeros((1, 24))

W2 = np.random.randn(24, 12) * 0.5
b2 = np.zeros((1, 12))

W3 = np.random.randn(12, 3) * 0.5
b3 = np.zeros((1, 3))

learning_rate = 0.5
epochs = 1000

loss_history = []


# Sigmoid activation
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

    # Mean Squared Error
    loss = np.mean((A3 - y_train) ** 2)

    loss_history.append(loss)

    # Backpropagation
    dA3 = 2 * (A3 - y_train) / len(X_train)

    dZ3 = dA3 * A3 * (1 - A3)

    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * A2 * (1 - A2)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * A1 * (1 - A1)

    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

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

# Get predicted and actual classes
predicted = np.argmax(output, axis=1)
actual = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted == actual)

print("Experiment 24")
print("Neural Network Analysis for Multi-Class Data")
print("Activation Function: Sigmoid")
print("Number of Classes: 3")
print("Test Accuracy:", accuracy * 100, "%")


# Plot loss
plt.plot(loss_history)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Multi-Class Data - Sigmoid Activation")

plt.show()


# Plot classification
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=actual
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Multi-Class Data Classification")

plt.show()

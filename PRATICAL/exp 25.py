import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate circular data
n = 1000

X = np.random.uniform(-1.5, 1.5, (n, 2))

distance = X[:, 0]**2 + X[:, 1]**2

# Create two classes
y = (distance > 0.5).astype(int).reshape(-1, 1)

# Shuffle data
index = np.random.permutation(n)

X = X[index]
y = y[index]

# Split into training and testing data
split = 800

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# Neural Network
W1 = np.random.randn(2, 16) * 0.5
b1 = np.zeros((1, 16))

W2 = np.random.randn(16, 8) * 0.5
b2 = np.zeros((1, 8))

W3 = np.random.randn(8, 1) * 0.5
b3 = np.zeros((1, 1))

learning_rate = 0.05
epochs = 1000

loss_history = []


# Tanh activation function
def tanh(x):
    return np.tanh(x)


# Tanh derivative
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# Sigmoid activation for output
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# Training
for epoch in range(epochs):

    # Forward propagation
    Z1 = X_train @ W1 + b1
    A1 = tanh(Z1)

    Z2 = A1 @ W2 + b2
    A2 = tanh(Z2)

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
    dZ2 = dA2 * tanh_derivative(Z2)

    dW2 = A1.T @ dZ2 / len(X_train)
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * tanh_derivative(Z1)

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
A1 = tanh(X_test @ W1 + b1)
A2 = tanh(A1 @ W2 + b2)
output = sigmoid(A2 @ W3 + b3)

# Convert probability to class
predicted = (output >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predicted == y_test)

print("Experiment 25")
print("Neural Network Analysis for Circular Data")
print("Activation Function: Tanh")
print("Test Accuracy:", accuracy * 100, "%")


# Plot loss
plt.plot(loss_history)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Circular Data - Tanh Activation")

plt.show()


# Plot classification
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test.ravel()
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Circular Data Classification")

plt.show()

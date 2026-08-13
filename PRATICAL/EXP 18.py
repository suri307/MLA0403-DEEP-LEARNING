import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X = np.array([[1,1],[2,1],[1,2],[5,5],[6,5],[5,6]])
y = np.array([0,0,0,1,1,1])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = MLPClassifier(
    hidden_layer_sizes=(3,),
    activation='identity',
    solver='lbfgs',
    max_iter=2000,
    random_state=42
)

model.fit(X_scaled, y)

plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Neural Network Analysis for Two-Class Data")
plt.show()

print("Accuracy:", model.score(X_scaled, y))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

X, y = make_circles(n_samples=500,
                    noise=0.1,
                    factor=0.5)

model = MLPClassifier(hidden_layer_sizes=(10,),
                      activation='identity',
                      max_iter=1000)

model.fit(X,y)

plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Circular Data")
plt.show()

print("Accuracy:", model.score(X,y))

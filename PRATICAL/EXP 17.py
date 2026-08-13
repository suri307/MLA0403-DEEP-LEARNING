import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

np.random.seed(0)

X1 = np.random.randn(50,2) + [2,2]
X2 = np.random.randn(50,2) + [-2,-2]

X = np.vstack((X1,X2))
y = np.array([0]*50 + [1]*50)

model = LogisticRegression()
model.fit(X,y)

plt.scatter(X[:,0], X[:,1], c=y)

x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

xx, yy = np.meshgrid(np.linspace(x_min,x_max,100),
                     np.linspace(y_min,y_max,100))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0.5])
plt.title("Linear Separability")
plt.show()

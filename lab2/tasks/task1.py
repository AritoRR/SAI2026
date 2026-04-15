import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('../source/nn_1.csv', delimiter=",")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.4, stratify=y, random_state=42
)


class Neuron_default:
    def __init__(self):
        self.w0 = np.random.normal(size=1)[0]
        self.w1 = np.random.normal(size=1)[0]
        self.w2 = np.random.normal(size=1)[0]
        self.nu = 0.05

    def predict(self, x1, x2):
        return 1 if 1 * self.w0 + x1 * self.w1 + x2 * self.w2 > 0 else -1

    def fix(self, x1, x2, y_pred, y):
        self.w0 = self.w0 + self.nu * (y - y_pred) * 1
        self.w1 = self.w1 + self.nu * (y - y_pred) * x1
        self.w2 = self.w2 + self.nu * (y - y_pred) * x2

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            for i in range(len(X)):
                y_pred = self.predict(X[i][0], X[i][1])
                self.fix(X[i][0], X[i][1], y_pred, y[i])


class Neuron_sigmoid:
    def __init__(self):
        self.w0 = np.random.normal(size=1)[0] * 0.1
        self.w1 = np.random.normal(size=1)[0] * 0.1
        self.w2 = np.random.normal(size=1)[0] * 0.1
        self.nu = 0.05

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x1, x2):
        return self.activate(1 * self.w0 + x1 * self.w1 + x2 * self.w2)

    def predict(self, x1, x2):
        return 1 if self.forward(x1, x2) > 0.5 else -1

    def fix(self, x1, x2, y_pred_raw, y):
        y_binary = 1 if y == 1 else 0
        error = y_binary - y_pred_raw
        derivative = y_pred_raw * (1 - y_pred_raw)
        delta = error * derivative
        self.w0 += self.nu * delta * 1
        self.w1 += self.nu * delta * x1
        self.w2 += self.nu * delta * x2

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            for i in range(len(X)):
                y_pred = self.forward(X[i][0], X[i][1])
                self.fix(X[i][0], X[i][1], y_pred, y[i])


class Neuron_bisigmoid:
    def __init__(self):
        self.w0 = np.random.normal(size=1)[0] * 0.1
        self.w1 = np.random.normal(size=1)[0] * 0.1
        self.w2 = np.random.normal(size=1)[0] * 0.1
        self.nu = 0.05

    def activate(self, x):
        return np.tanh(x)

    def forward(self, x1, x2):
        return self.activate(1 * self.w0 + x1 * self.w1 + x2 * self.w2)

    def predict(self, x1, x2):
        # Для предсказания преобразуем в -1 или 1
        return 1 if self.forward(x1, x2) > 0 else -1

    def fix(self, x1, x2, y_pred_raw, y):
        error = y - y_pred_raw
        derivative = 1 - y_pred_raw ** 2
        delta = error * derivative

        self.w0 += self.nu * delta * 1
        self.w1 += self.nu * delta * x1
        self.w2 += self.nu * delta * x2

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            for i in range(len(X)):
                y_pred = self.forward(X[i][0], X[i][1])
                self.fix(X[i][0], X[i][1], y_pred, y[i])

clf_default = Neuron_default()
clf_sigmoid = Neuron_sigmoid()
clf_bisigmoid = Neuron_bisigmoid()

clf_default.fit(X_train, y_train, 20)
clf_sigmoid.fit(X_train, y_train, 10)
clf_bisigmoid.fit(X_train, y_train, 100)

y_pred = list()
for i in range(len(X_test)):
    y_pred.append(clf_default.predict(X_test[i][0], X_test[i][1]))

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

y_pred = list()
for i in range(len(X_test)):
    y_pred.append(clf_sigmoid.predict(X_test[i][0], X_test[i][1]))

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
y_pred = list()
for i in range(len(X_test)):
    y_pred.append(clf_bisigmoid.predict(X_test[i][0], X_test[i][1]))

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('../source/nn_1.csv', delimiter=",")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.6, stratify=y, random_state=42
)


class NeuronNetwork:
    def __init__(self):
        self.W1 = np.random.randn(2, 4) * 0.1
        self.b1 = np.zeros(4)
        self.W2 = np.random.randn(4, 1) * 0.1
        self.b2 = np.zeros(1)

        self.lr = 0.1

    def sigmoid_activate(self, X):
        return 1 / (1 + np.exp(-X))

    def forward(self, X):
        HL = self.sigmoid_activate(np.dot(X, self.W1) + self.b1)
        out = self.sigmoid_activate(np.dot(HL, self.W2) + self.b2)
        return HL, out[0]

    def predict(self, X):
        return 1 if self.forward(X)[1] > 0.5 else -1

    def backward(self, X, HL, y_predict, y):
        X = X.reshape(1, 2)
        HL = HL.reshape(1, 4)
        y_binary = 1 if y == 1 else 0

        dW2 = np.dot(HL.T, y_predict - y_binary)
        db2 = y_predict - y_binary
        dZ2 = y_predict - y_binary
        dZ1 = np.dot(dZ2, self.W2.T) * (HL * (1 - HL))
        dW1 = np.dot(X.T, dZ1)
        db1 = dZ1.flatten()

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1.reshape(-1)

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            for i in indices:
                HL, y_predict = self.forward(X[i])
                self.backward(X[i], HL, y_predict, y[i])


clf = NeuronNetwork()
clf.fit(X_train, y_train, 150)

y_predicts = list()
for i in range(len(X_test)):
    y_predicts.append(clf.predict(X_test[i]))

accuracy = accuracy_score(y_test, y_predicts)

print(accuracy, f1_score(y_test, y_predicts))

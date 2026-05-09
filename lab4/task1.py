from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
import pandas as pd
import numpy as np
from statistics import mode

df = pd.read_csv('src/glass.csv', delimiter=',')

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=52
)


class EnsembleClassifier:
    def __init__(self, classifier_class, n_estimators=10, random_state=None):
        self.classifier_class = classifier_class
        self.n_estimators = n_estimators
        if random_state is not None:
            np.random.seed(random_state)
        self.estimators = []

    def bootstrap(self, X, y):
        N = X.shape[0]
        indices = (np.random.choice(N, size=N, replace=True))
        return X.iloc[indices], y.iloc[indices]

    def fit(self, X, y):
        for i in range(self.n_estimators):
            estimator = clone(self.classifier_class)
            X_train, y_train = self.bootstrap(X, y)
            estimator.fit(X_train, y_train)
            self.estimators.append(estimator)
        return self

    def predict(self, X):
        predictions = [estimator.predict(X) for estimator in self.estimators]
        votes = []
        for i in range(len(X)):
            predictions_current = [predict[i] for predict in predictions]
            votes.append(mode(predictions_current))
        return votes


param = {
    'GaussianNativBayes': {
        'estimator': GaussianNB(),
        'n_estimators_vs': [10, 10]
    },
    'KNeighborsClassifier': {
        'estimator': KNeighborsClassifier(),
        'n_estimators_vs': [30, 30]
    },
    'SVC': {
        'estimator': svm.SVC(),
        'n_estimators_vs': [3, 5]
    },
    'DecisionTreeClassifier': {
        'estimator': DecisionTreeClassifier(),
        'n_estimators_vs': [30, 30]
    },
}
n_estimators_range = [1, 3, 5, 10, 20, 30, 50]

for estimator in param:
    print(f"{estimator}: ")
    ensemble = EnsembleClassifier(
        param[estimator]['estimator'],
        n_estimators=param[estimator]['n_estimators_vs'][0],
        random_state=52)
    ensemble.fit(X_train, y_train)
    y_predict = ensemble.predict(X_test)

    print("My  accuracy = ", accuracy_score(y_predict, y_test))

    ensemble = BaggingClassifier(
        estimator=param[estimator]['estimator'],
        n_estimators=param[estimator]['n_estimators_vs'][1],
        random_state=52)
    ensemble.fit(X_train, y_train)
    y_predict = ensemble.predict(X_test)

    print("Skl accuracy = ", accuracy_score(y_predict, y_test), '\n')

results = {name: [] for name in param}

for estimator in param:
    for n in n_estimators_range:
        ensemble = EnsembleClassifier(
            param[estimator]['estimator'],
            n_estimators=n,
            random_state=52)
        ensemble.fit(X_train, y_train)
        y_predict = ensemble.predict(X_test)
        results[estimator].append(accuracy_score(y_predict, y_test))

plt.figure(figsize=(12, 8))
for i, (name, _) in enumerate(param.items(), 1):
    plt.subplot(2, 2, i)
    print(name, n_estimators_range,  results[name])
    plt.plot(n_estimators_range, results[name], 'o-')
    plt.title(name)
    plt.xlabel('Число классификаторов')
    plt.ylabel('Accuracy на тесте')
    plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

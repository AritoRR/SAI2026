import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('src/vehicle.csv', delimiter=',')
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=52
)

models = {
    'GaussianNativBayes': {
        'estimator': GaussianNB(),
    },
    'Perceptron': {
        'estimator': Perceptron(),
    },
    'DecisionTreeClassifier_max_depth_1': {
        'estimator': DecisionTreeClassifier(max_depth=1),
    },

    'DecisionTreeClassifier_max_depth_8': {
        'estimator': DecisionTreeClassifier(max_depth=8),
    }
}

n_estimators_range = [1, 3, 5, 10, 20, 30, 50]
results = {name: [] for name in models}

for estimator in models:
    for n in n_estimators_range:
        ensemble = AdaBoostClassifier(
            models[estimator]['estimator'],
            n_estimators=n,
            random_state=52)
        ensemble.fit(X_train, y_train)
        y_predict = ensemble.predict(X_test)
        results[estimator].append(accuracy_score(y_predict, y_test))

plt.figure(figsize=(12, 8))
for i, (name, _) in enumerate(models.items(), 1):
    plt.subplot(2, 2, i)
    print(name, n_estimators_range, results[name])
    plt.plot(n_estimators_range, results[name], 'o-')
    plt.title(name)
    plt.xlabel('Число классификаторов')
    plt.ylabel('Accuracy на тесте')
    plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

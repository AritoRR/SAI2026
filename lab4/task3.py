import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier


def preprocess(df, age_median=None, embarked_mode=None, fare_median=None):
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features].copy()

    if age_median is None:
        age_median = X['Age'].median()
    if embarked_mode is None:
        embarked_mode = X['Embarked'].mode()[0]
    if fare_median is None:
        fare_median = X['Fare'].median()

    X['Age'] = X['Age'].fillna(age_median)
    X['Embarked'] = X['Embarked'].fillna(embarked_mode)
    X['Fare'] = X['Fare'].fillna(fare_median)
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    X['Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    X['Embarked'] = X['Embarked'].fillna(0)

    if 'Survived' in df.columns:
        y = df['Survived'].values
        return X.values.astype(float), y
    else:
        return X.values.astype(float)


class MetaClassifier:
    def __init__(self, base_model1, base_model2, meta_model):
        self.base_model1 = base_model1
        self.base_model2 = base_model2
        self.meta_model = meta_model

    def fit(self, X, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        predicts_1 = np.zeros(len(y))
        predicts_2 = np.zeros(len(y))

        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            m1 = clone(self.base_model1)
            m1.fit(X_train, y_train)
            predicts_1[val_idx] = m1.predict(X_val)

            m2 = clone(self.base_model2)
            m2.fit(X_train, y_train)
            predicts_2[val_idx] = m2.predict(X_val)

        X_meta = np.column_stack([predicts_1, predicts_2])
        self.meta_model.fit(X_meta, y)

        self.base_model1.fit(X, y)
        self.base_model2.fit(X, y)

    def predict(self, X):
        pred1 = self.base_model1.predict(X)
        pred2 = self.base_model2.predict(X)
        X_meta = np.column_stack([pred1, pred2])
        return self.meta_model.predict(X_meta)


train_df = pd.read_csv('src/titanic_train.csv')
test_df = pd.read_csv('src/titanic_test.csv')

X_train, y_train = preprocess(train_df)
X_test = preprocess(test_df)

clf = MetaClassifier(KNeighborsClassifier(), DecisionTreeClassifier(), LogisticRegression())
clf.fit(X_train, y_train)

estimators = [
    ('knn', KNeighborsClassifier()),
    ('dtree', DecisionTreeClassifier())
]

sk_clf = StackingClassifier(estimators, LogisticRegression())
sk_clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
print("Accuracy on train:", accuracy_score(y_train, y_pred_train))

y_pred_test = clf.predict(X_test)
y_pred_sk_test = sk_clf.predict(X_test)
print(accuracy_score(y_pred_test, y_pred_sk_test))

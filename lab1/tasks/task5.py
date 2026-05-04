import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

data_frame_glass = pd.read_csv('../src/glass.csv')
X = data_frame_glass.iloc[:, 1:-1]
y = data_frame_glass.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'max_leaf_nodes': [None, 10, 20, 30],
    'class_weight': [None, 'balanced']
}

dt = tree.DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Лучшие параметры:")
print(grid_search.best_params_)
print(f"Лучшая точность на кросс-валидации: {grid_search.best_score_:.4f}")
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)
print(f"Точность на тесте: {accuracy_score(y_test, y_pred):.4f}")


plt.figure(figsize=(20, 10))
tree.plot_tree(best_dt, feature_names=X.columns, class_names=[str(c) for c in sorted(np.unique(y))],
               filled=True, rounded=True, fontsize=10)
plt.title(f"Дерево решений (глубина={best_dt.get_depth()}, листьев={best_dt.get_n_leaves()})")
plt.tight_layout()
plt.show()


df_spam7 = pd.read_csv('../src/spam7.csv')
X = df_spam7.drop('yesno', axis=1)
y = df_spam7['yesno'].map({'y': 1, 'n': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

dt = tree.DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    dt, param_grid, cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("Лучшие параметры:")
print(grid_search.best_params_)
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

plt.figure(figsize=(20, 10))
tree.plot_tree(best_dt, feature_names=X.columns, class_names=[str(c) for c in sorted(np.unique(y))],
               filled=True, rounded=True, fontsize=10)
plt.title(f"Дерево решений (глубина={best_dt.get_depth()}, листьев={best_dt.get_n_leaves()})")
plt.tight_layout()
plt.show()
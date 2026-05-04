import pandas as pd
from sklearn import tree, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../src/bank_scoring_train.csv', delimiter="\t")
X_train = df.iloc[:, 1:].values
y_train = df.iloc[:, 0].values

test_df = pd.read_csv('../src/bank_scoring_test.csv', delimiter="\t")
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

print(f"\nТочность: {accuracy_score(y_test, y_predict):.4f}"
      f"\nF1: {f1_score(y_test, y_predict):.4f}"
      f"\nROC-AUC: {roc_auc_score(y_test, y_predict):.4f}")


tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
print(f"FP (ошибка 1 рода): {fp}")
print(f"FN (ошибка 2 рода): {fn}")

dt = tree.DecisionTreeClassifier(max_depth=50)
dt.fit(X_train, y_train)
y_predict = dt.predict(X_test)

print(f"\nТочность: {accuracy_score(y_test, y_predict):.4f}"
      f"\nF1: {f1_score(y_test, y_predict):.4f}"
      f"\nROC-AUC: {roc_auc_score(y_test, y_predict):.4f}")


tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
print(f"FP (ошибка 1 рода): {fp}")
print(f"FN (ошибка 2 рода): {fn}")

log_reg = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42
    )
)

# Обучаем
log_reg.fit(X_train, y_train)
y_predict = log_reg.predict(X_test)

print(f"\nТочность: {accuracy_score(y_test, y_predict):.4f}"
      f"\nF1: {f1_score(y_test, y_predict):.4f}"
      f"\nROC-AUC: {roc_auc_score(y_test, y_predict):.4f}")



tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
print(f"FP (ошибка 1 рода): {fp}")
print(f"FN (ошибка 2 рода): {fn}")
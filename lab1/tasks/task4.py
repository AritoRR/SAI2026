import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_decision_regions


def get_data_frame(path_train, path_test):
    df = pd.read_csv(path_train, delimiter="\t")
    X_train = df.iloc[:, :-1].values
    y_train = (df.iloc[:, -1] == "red").astype(int).values
    test_df = pd.read_csv(path_test, delimiter="\t")
    X_test = test_df.iloc[:, :-1].values
    y_test = (test_df.iloc[:, -1] == "red").astype(int).values
    return X_train, y_train, X_test, y_test


def decision_boundary_plot(name, X, y, clf):
    plot_decision_regions(X=X, y=y, clf=clf)
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title(name)
    plt.show()


def print_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")


print("SVMDATA A:")

X_train, y_train, X_test, y_test = get_data_frame(
    '../source/data_4/svmdata_a.txt',
    '../source/data_4/svmdata_a_test.txt')

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)
print(f"\nТочность: {accuracy:.4f}")

decision_boundary_plot('SVM train', X_train, y_train, clf)
decision_boundary_plot('SVM test', X_test, y_test, clf)

print(f"\nКоличество опорных векторов: {len(clf.support_vectors_)}")

print("\nconfusion matrix (train):")
print_cm(y_train, y_predict)
print("\nconfusion matrix (test):")
print_cm(y_test, y_predict)

print("SVMDATA B:")

X_train, y_train, X_test, y_test = get_data_frame(
    '../source/data_4/svmdata_b.txt',
    '../source/data_4/svmdata_b_test.txt')

clf = svm.SVC(kernel='linear', C=1.0, random_state=0)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)
print(f"\nТочность: {accuracy:.4f}")

decision_boundary_plot('SVM train', X_train, y_train, clf)
decision_boundary_plot('SVM test', X_test, y_test, clf)


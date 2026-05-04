import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
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


def decision_boundary_plot(name, clf, X, y, ax=None):
    plot_decision_regions(X=X, y=y, clf=clf, ax=ax)
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title(name)
    if ax is None:
        plt.show()


def print_cm(y_true, y_pred, printf=True):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * ((precision * recall) / (precision + recall))
    if printf:
        print(f"\nTP: {tp}; FP: {fp}\n"
              f"FN: {fn}; TN: {tn}\n"
              f"F1: {f1}")
    return f1


def clf_test_kernel(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"{clf.kernel}: Точность: {accuracy:.4f}")
    decision_boundary_plot(f'{clf.kernel}', clf, X_test, y_test)

def clf_test_gamma(kernel, X_train, y_train, X_test, y_test, degree=None):
    for gamma in [0.01, 0.1, 1, 10, 100]:
        if kernel == 'poly':
            clf = svm.SVC(kernel=kernel, degree=degree, gamma=gamma)
        else:
            clf = svm.SVC(kernel=kernel, gamma=gamma)
        clf.fit(X_train, y_train)
        if degree:
            print(f"{kernel}-{degree}: ")
        else:
            print(f"{kernel}: ")

        pre_train = clf.predict(X_train)
        accuracy_train = accuracy_score(y_train, pre_train)
        f1_train = print_cm(y_train, pre_train, False)
        print(f"train; gamma={gamma}:  "
              f"Количество опорных векторов: {len(clf.support_vectors_)};"
              f" Точность: {accuracy_train:.4f};"
              f" F1: {f1_train}")

        pre_test = clf.predict(X_test)
        accuracy_test = accuracy_score(y_test, pre_test)
        f1_test = print_cm(y_test, pre_test, False)
        print(f"test; gamma={gamma}:   "
              f"Количество опорных векторов: {len(clf.support_vectors_)};"
              f" Точность: {accuracy_test:.4f};"
              f" F1: {f1_test}")


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        decision_boundary_plot(f"{kernel}-SVM train", clf, X_train, y_train, ax=ax1)
        decision_boundary_plot(f"{kernel}-SVM test", clf, X_test, y_test, ax=ax2)

        plt.tight_layout()
        plt.show()


print("SVMDATA A:")

X_train, y_train, X_test, y_test = get_data_frame(
    '../src/data_4/svmdata_a.txt',
    '../src/data_4/svmdata_a_test.txt')

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)
print(f"\nТочность: {accuracy:.4f}")

decision_boundary_plot('SVM train', clf, X_train, y_train)
decision_boundary_plot('SVM test', clf, X_test, y_test)

print(f"\nКоличество опорных векторов: {len(clf.support_vectors_)}")

print("\nconfusion matrix (train):")
print_cm(y_train, y_predict)
print("\nconfusion matrix (test):")
print_cm(y_test, y_predict)

print("SVMDATA B:")

X_train, y_train, X_test, y_test = get_data_frame(
    '../src/data_4/svmdata_b.txt',
    '../src/data_4/svmdata_b_test.txt')

print("Train:")
for C in [1000, 100, 10, 1.0, 0.1, 0.01]:
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_train)
    accuracy = accuracy_score(y_train, y_predict)

    print(f"C={C}: "
          f"Количество опорных векторов: {len(clf.support_vectors_)};"
          f" Точность: {accuracy:.4f};"
          f" F1: {print_cm(y_train, y_predict, False)}")

print("Test:")
for C in [1000, 100, 10, 1.0, 0.1, 0.01]:
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)

    print(f"C={C}: "
          f"Количество опорных векторов: {len(clf.support_vectors_)};"
          f" Точность: {accuracy:.4f};"
          f" F1: { print_cm(y_test, y_predict, False)}")

print("SVMDATA С:")

X_train, y_train, X_test, y_test = get_data_frame(
    '../src/data_4/svmdata_c.txt',
    '../src/data_4/svmdata_c_test.txt')

for kernel in ['linear', 'poly', 'sigmoid', 'rbf']:
    if kernel == 'poly':
        for degree in range(1, 6):
            clf = svm.SVC(kernel=kernel, degree=degree, random_state=0)
            clf_test_kernel(clf, X_train, y_train, X_test, y_test)
    else:
        clf = svm.SVC(kernel=kernel, random_state=0)
        clf_test_kernel(clf, X_train, y_train, X_test, y_test)


print("SVMDATA D:")

X_train, y_train, X_test, y_test = get_data_frame(
    '../src/data_4/svmdata_d.txt',
    '../src/data_4/svmdata_d_test.txt')

for kernel in ['poly', 'sigmoid', 'rbf']:
    if kernel == 'poly':
        for degree in range(1, 6):
            clf = svm.SVC(kernel=kernel, degree=degree, random_state=0)
            clf_test_kernel(clf, X_train, y_train, X_test, y_test)
    else:
        clf = svm.SVC(kernel=kernel, random_state=0)
        clf_test_kernel(clf, X_train, y_train, X_test, y_test)


print("SVMDATA E:")

X_train, y_train, X_test, y_test = get_data_frame(
    '../src/data_4/svmdata_e.txt',
    '../src/data_4/svmdata_e_test.txt')

for kernel in ['poly', 'sigmoid', 'rbf']:
    if kernel == 'poly':
        for degree in range(1, 5):
            clf_test_gamma(kernel, X_train, y_train, X_test, y_test, degree)
    else:
        clf_test_gamma(kernel, X_train, y_train, X_test, y_test)
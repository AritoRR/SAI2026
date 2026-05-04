import pandas as pd
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def plot(sizes, train_scores, test_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, train_scores, label='Train')
    plt.plot(sizes, test_scores, label='Test')
    plt.xlabel('Размер выборки обучающей выборки')
    plt.ylabel('Точность')
    plt.title('Кривые обучения')
    plt.legend()
    plt.grid(True)
    plt.show()


def clf_do(X, y, size, NB_class):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=size, stratify=y, random_state=42
    )
    clf = NB_class()
    clf.fit(X_train, y_train)
    return accuracy_score(y_train, clf.predict(X_train)), accuracy_score(y_test, clf.predict(X_test))


df_ttt = pd.read_csv('../src/tic_tac_toe.txt', delimiter=",", header=None)
df_spam = pd.read_csv('../src/spam.csv')

X_ttt = (df_ttt.iloc[:, :-1] == 'x').astype(int)
y_ttt = (df_ttt.iloc[:, -1] == 'positive').astype(int)

X_spam = df_spam.iloc[:, 1:-1]
y_spam = (df_spam['type'] == 'spam').astype(int)

train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
train_scores_ttt, train_scores_spam = [], []
test_scores_ttt, test_scores_spam = [], []

for size in train_sizes:
    train_score, test_score = clf_do(X_ttt, y_ttt, size, BernoulliNB)
    train_scores_ttt.append(train_score)
    test_scores_ttt.append(test_score)

    train_score, test_score = clf_do(X_spam, y_spam, size, MultinomialNB)
    train_scores_spam.append(train_score)
    test_scores_spam.append(test_score)


plot(train_sizes, train_scores_ttt, test_scores_ttt)
plot(train_sizes, train_scores_spam, test_scores_spam)

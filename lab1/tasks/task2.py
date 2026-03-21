import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


np.random.seed(67)

loc = {
    -1: (13, 14),
    1: (20, 18)
}

scale = {
    -1: (4, 4),
    1: (5, 5)
}

size = {
    -1: 10,
    1: 90
}

X_class_first = np.random.normal(
    loc=loc[-1],
    scale=scale[-1],
    size=(size[-1], 2)
)

X_class_second = np.random.normal(
    loc=loc[1],
    scale=scale[1],
    size=(size[1], 2)
)

X = np.vstack([X_class_first, X_class_second])
y = np.array([-1] * size[-1] + [1] * size[1])

plt.figure(figsize=(10, 8))

plt.scatter(X_class_first[:, 0], X_class_first[:, 1], label=f'Класс -1 (n={len(X_class_first)})')
plt.scatter(X_class_second[:, 0], X_class_second[:, 1], label=f'Класс 1 (n={len(X_class_second)})')

plt.xlabel('Признак X₁')
plt.ylabel('Признак X₂')

plt.legend(fontsize=12)

plt.show()


X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.3, stratify=y, random_state=42
    )

clf = GaussianNB()
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)

print(f"Accuracy: {accuracy}")

cm = confusion_matrix(y_test, y_predict, labels=[-1, 1])

tn, fp, fn, tp = cm.ravel()
print(f"\nTN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

y_test_binary = (y_test == 1).astype(int)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
baseline_pr = np.mean(y_test_binary)

precision, recall, pr_thresholds = precision_recall_curve(y_test_binary, y_pred_proba)
pr_auc = auc(recall, precision)

min_len = min(len(precision), len(recall))
precision_cut = precision[:min_len]
recall_cut = recall[:min_len]

f1_scores = 2 * (precision_cut * recall_cut) / (precision_cut + recall_cut + 1e-10)
best_f1_idx = np.argmax(f1_scores)
best_precision = precision_cut[best_f1_idx]
best_recall = recall_cut[best_f1_idx]
best_f1 = f1_scores[best_f1_idx]

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR кривая')
plt.axhline(y=baseline_pr, color='r', linestyle='--', label=f'Precision')

plt.xlabel('Recall')
plt.ylabel('Precision')

plt.legend(fontsize=12)

plt.plot(best_recall, best_precision, 'go')
plt.show()

print(f"\nPR-AUC: {pr_auc:.4f}")
print(f"Максимальный F1-score: {best_f1:.4f}")
print(f"При Precision = {best_precision:.4f}, Recall = {best_recall:.4f}")
print(f"Baseline Precision: {baseline_pr:.4f}")
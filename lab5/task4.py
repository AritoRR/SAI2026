import csv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('src/longley.csv', sep=',')

data = data.drop('Population', axis=1)
X = data.drop('Employed', axis=1)
y = data['Employed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

results = []

lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)

mse_train_lr = round(mean_squared_error(y_train, y_train_pred_lr), 4)
mse_test_lr = round(mean_squared_error(y_test, y_test_pred_lr), 4)

results.append({
    'Model': 'Linear Regression',
    'MSE_Train': mse_train_lr,
    'MSE_Test': mse_test_lr,
    'Lambda': '-',
    'Coefficient': '-'
})

lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

train_errors = []
test_errors = []
coefficients = []

for lam in lambdas:
    ridge = Ridge(alpha=lam)
    ridge.fit(X_train, y_train)

    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)

    mse_train = round(mean_squared_error(y_train, y_train_pred), 4)
    mse_test = round(mean_squared_error(y_test, y_test_pred), 4)

    results.append({
        'Model': 'Ridge Regression',
        'MSE_Train': mse_train,
        'MSE_Test': mse_test,
        'Lambda': lam,
        'Coefficient': [round(i, 4) for i in ridge.coef_]
    })
    train_errors.append(mse_train)
    test_errors.append(mse_test)
    coefficients.append(ridge.coef_)

with open("result4.csv", "w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Тип Модели', 'MSE_Train', 'MSE_Test', 'Lambda', 'Coefficient'])

    for row in results:
        writer.writerow([
            row['Model'],
            row['MSE_Train'],
            row['MSE_Test'],
            row['Lambda'],
            row['Coefficient']
        ])

plt.figure(figsize=(8, 5))

plt.plot(lambdas, train_errors, 'b-o', label='Train MSE Ridge', linewidth=2)
plt.plot(lambdas, test_errors, 'r-s', label='Test MSE Ridge', linewidth=2)
plt.axhline(y=mse_train_lr, color='b', linestyle='--', alpha=0.5, label='Train MSE Linear')
plt.axhline(y=mse_test_lr, color='r', linestyle='--', alpha=0.5, label='Train MSE Linear')
plt.xscale('log')
plt.xlabel('λ')
plt.ylabel('MSE')
plt.title('Ошибка на train и test')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

import csv

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

data = pd.read_csv('src/reglab1.txt', sep='\t')


# Список моделей для тестирования
models = [
    # z как зависимая
    {'name': 'z ~ x', 'Y': 'z', 'X': ['x']},
    {'name': 'z ~ y', 'Y': 'z', 'X': ['y']},
    {'name': 'z ~ x + y', 'Y': 'z', 'X': ['x', 'y']},

    # x как зависимая
    {'name': 'x ~ z', 'Y': 'x', 'X': ['z']},
    {'name': 'x ~ y', 'Y': 'x', 'X': ['y']},
    {'name': 'x ~ z + y', 'Y': 'x', 'X': ['z', 'y']},

    # y как зависимая
    {'name': 'y ~ z', 'Y': 'y', 'X': ['z']},
    {'name': 'y ~ x', 'Y': 'y', 'X': ['x']},
    {'name': 'y ~ z + x', 'Y': 'y', 'X': ['z', 'x']},
]

results = []

for m in models:
    X = data[m['X']]
    y = data[m['Y']]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    results.append({
        'Модель': m['name'],
        'R²': round(r2_score(y, y_pred), 4),
        'RMSE': round(root_mean_squared_error(y, y_pred), 4),
        'Коэффициенты': dict(zip(m['X'], np.round(model.coef_, 4))),
        'Свободный член': round(model.intercept_, 4)
    })

# Выводим все результаты
results_df = pd.DataFrame(results)

with open("result1.csv", "w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Заголовки
    writer.writerow(['Модель', 'R²', 'RMSE', 'Коэффициенты', 'Свободный член'])

    # Данные построчно
    for _, row in results_df.iterrows():
        writer.writerow([
            row['Модель'],
            row['R²'],
            row['RMSE'],
            row['Коэффициенты'],
            row['Свободный член']
        ])
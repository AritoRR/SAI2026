import csv
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('src/reglab.txt', sep='\t')

features = ['x1', 'x2', 'x3', 'x4']


def calc_rss(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    rss = np.sum((y - y_pred) ** 2)
    return rss, model


results = []
total_features = len(features)
y = data['y'].values

for k in range(1, total_features):
    for subset in combinations(features, k):
        X = data[list(subset)].values
        rss, model = calc_rss(X, y)
        results.append({
            'k': k,
            'Признаки': ', '.join(subset),
            'RSS': round(rss, 6),
        })

with open("result2.csv", "w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['k', 'Признаки', 'RSS'])

    # Данные построчно
    for row in results:
        writer.writerow([
            row['k'],
            row['Признаки'],
            row['RSS']
        ])

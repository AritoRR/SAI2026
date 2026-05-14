import csv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('src/eustock.csv', sep=',')
data['Day'] = np.arange(len(data))

exchanges = ['DAX', 'SMI', 'CAC', 'FTSE']
results = []

X = data[['Day']].values

all_values = []
all_days = []

for ex in exchanges:
    all_values.extend(data[ex].values)
    all_days.extend(data['Day'].values)

    y = data[ex].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Дневная динамика
    daily_change = model.coef_[0]
    # Общее изменение за период
    total_change = daily_change * len(data)

    results.append({
        'Биржа': ex,
        'Изменение в день': daily_change,
        'Общее изменение': total_change,
        'R²': r2,
        'Intercept': model.intercept_
    })

X_all = np.array(all_days).reshape(-1, 1)
y_all = np.array(all_values)

model_all = LinearRegression()
model_all.fit(X_all, y_all)
y_all_pred = model_all.predict(X_all)
r2_all = r2_score(y_all, y_all_pred)

results.append({
        'Биржа': 'All',
        'Изменение в день': model_all.coef_[0],
        'Общее изменение': model_all.coef_[0]*len(data),
        'R²': r2_all,
        'Intercept': model_all.intercept_
    })

with open("result5.csv", "w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Биржа', 'Изменение в день', 'Общее изменение', 'R²', 'Intercept'])
    for row in results:
        writer.writerow([
            row['Биржа'],
            row['Изменение в день'],
            row['Общее изменение'],
            row['R²'],
            row['Intercept']
        ])

plt.figure(figsize=(8, 4))

colors = ['blue', 'green', 'red', 'orange']
for i, ex in enumerate(exchanges):
    plt.plot(data['Day'], data[ex], color=colors[i], label=ex, linewidth=0.8)

plt.xlabel('День', fontsize=12)
plt.ylabel('Котировки', fontsize=12)
plt.title('Кривые изменения котировок во времени', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)


plt.show()
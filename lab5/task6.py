import csv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


data = pd.read_csv('src/JohnsonJohnson.csv', sep=',')
data['Year'] = data['index'].str.split(' ').str[0].astype(int)
data['Quarter'] = data['index'].str.split(' ').str[1]

quarters = ['Q1', 'Q2', 'Q3', 'Q4']
results = []

for q in quarters:
    q_data = data[data['Quarter'] == q]

    X = q_data[['Year']].values
    y = q_data['value'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    quarter_change = model.coef_[0]
    total_change = quarter_change * len(q_data)

    year_2016 = np.array([[2016]])
    forecast_q = model.predict(year_2016)[0]

    results.append({
        'Квартал': q,
        'Изменение в квартал': quarter_change,
        'Общее изменение': total_change,
        'R²': r2,
        'Intercept': model.intercept_,
        'Прогноз': forecast_q
    })

data['TimeIndex'] = range(len(data))
X_all = data[['TimeIndex']].values
y_all = data['value'].values

model_all = LinearRegression()
model_all.fit(X_all, y_all)
y_all_pred = model_all.predict(X_all)
r2_all = r2_score(y_all, y_all_pred)

last_year = data['Year'].max()
last_time_index = data[data['Year'] == last_year]['TimeIndex'].max()
years_diff = 2016 - last_year
quarters_diff = years_diff * 4
time_index_2016 = last_time_index + quarters_diff

forecast_2016_q1 = model_all.predict([[time_index_2016 + 1]])[0]
forecast_2016_q2 = model_all.predict([[time_index_2016 + 2]])[0]
forecast_2016_q3 = model_all.predict([[time_index_2016 + 3]])[0]
forecast_2016_q4 = model_all.predict([[time_index_2016 + 4]])[0]

avg_forecast_2016 = (forecast_2016_q1 + forecast_2016_q2 + forecast_2016_q3 + forecast_2016_q4) / 4

results.append({
    'Квартал': 'All',
    'Изменение в квартал': model_all.coef_[0],
    'Общее изменение': model_all.coef_[0] * len(data),
    'R²': r2_all,
    'Intercept': model_all.intercept_,
    'Прогноз': avg_forecast_2016
})

with open("result6.csv", "w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Квартал', 'Изменение в квартал', 'Общее изменение', 'R²', 'Intercept', 'Прогноз'])
    for row in results:
        writer.writerow([
            row['Квартал'],
            row['Изменение в квартал'],
            row['Общее изменение'],
            row['R²'],
            row['Intercept'],
            row['Прогноз']
        ])

plt.figure(figsize=(8, 4))

colors = ['blue', 'green', 'red', 'orange']
for q, color in zip(quarters, colors):
    q_data = data[data['Quarter'] == q]
    plt.plot(q_data['Year'], q_data['value'], 'o-', color=color,
                label=q, linewidth=1.5, markersize=4)

plt.xlabel('Год', fontsize=12)
plt.ylabel('Прибыль', fontsize=12)
plt.title('Поквартальная прибыль', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.show()

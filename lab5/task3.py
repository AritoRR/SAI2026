import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

data = pd.read_csv('src/cygage.txt', sep='\t')

X, y = data[['Depth']].values, data['calAge'].values
results = []

model_with_weight = LinearRegression()
model_with_weight.fit(X, y, sample_weight=data['Weight'])
y_pred = model_with_weight.predict(X)
r2 = r2_score(y, y_pred)
RMSE = root_mean_squared_error(y, y_pred)
results.append({
            'model_type': 'with_weight',
            'R2': r2,
            'RMSE': RMSE,
        })

model_without_weight = LinearRegression()
model_without_weight.fit(X, y)
y_pred = model_without_weight.predict(X)
r2 = r2_score(y, y_pred)
RMSE = root_mean_squared_error(y, y_pred)
results.append({
            'model_type': 'without_weight',
            'R2': r2,
            'RMSE': RMSE,
        })


with open("result3.csv", "w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Тип Модели', 'R2', 'RMSE'])

    for row in results:
        writer.writerow([
            row['model_type'],
            row['R2'],
            row['RMSE']
        ])

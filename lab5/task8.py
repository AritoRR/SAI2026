import csv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

data = pd.read_csv('src/svmdata6.txt', delimiter='\t')
X = data[['X']].values
y = data['Y'].values

epsilons = [0, 0.05, 0.1, 0.5, 1, 2, 5, 10]
results = []

for epsilon in epsilons:
    model = SVR(kernel='rbf', C=1, epsilon=epsilon)
    model.fit(X, y)
    mse = mean_squared_error(y, model.predict(X))
    results.append({
        'Epsilon': epsilon,
        'MSE': mse
    })

with open("result8.csv", "w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Epsilon', 'MSE'])
    for row in results:
        writer.writerow([
            row['Epsilon'],
            row['MSE']
        ])

fig, ax = plt.subplots(4, 1, figsize=(5, 10))

ax[0].set_title('Зависимость mse от ε', fontsize=14)

ax[0].plot([i['Epsilon'] for i in results[:3]], [i['MSE'] for i in results[:3]], 'o-', color='red',
         linewidth=1.5, markersize=4)

ax[0].set_xlabel('MSE', fontsize=12)
ax[0].set_ylabel('Epsilon', fontsize=12)
ax[0].grid(True, alpha=0.3)


ax[1].plot([i['Epsilon'] for i in results[2:5]], [i['MSE'] for i in results[2:5]], 'o-', color='red',
         linewidth=1.5, markersize=4)

ax[1].set_xlabel('MSE', fontsize=12)
ax[1].set_ylabel('Epsilon', fontsize=12)
ax[1].grid(True, alpha=0.3)


ax[2].plot([i['Epsilon'] for i in results[5:]], [i['MSE'] for i in results[5:]], 'o-', color='red',
         linewidth=1.5, markersize=4)

ax[2].set_xlabel('MSE', fontsize=12)
ax[2].set_ylabel('Epsilon', fontsize=12)
ax[2].grid(True, alpha=0.3)

ax[3].plot([i['Epsilon'] for i in results], [i['MSE'] for i in results], 'o-', color='red',
         linewidth=1.5, markersize=4)

ax[3].set_xlabel('MSE', fontsize=12)
ax[3].set_ylabel('Epsilon', fontsize=12)
ax[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

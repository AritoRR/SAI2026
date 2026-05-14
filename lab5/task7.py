import csv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('src/cars.csv', sep=',')

X = data[['speed']].values
y = data['dist'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print("R2  predict:\n", r2, model.predict(np.array([[40]]))[0])


results = []

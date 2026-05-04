import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('src/pluton.csv', delimiter=",", header=None)[1:].values
scaler = StandardScaler()
scaler.fit(df)

for i in range(1, 10):
    print(f"max_iter = {i}:")
    kmeans = KMeans(n_clusters=3, max_iter=i, random_state=67)
    kmeans.fit(df)
    print(f"Non Standard with real_iter = {kmeans.n_iter_} : inertia = {kmeans.inertia_}")
    kmeans.fit(scaler.transform(df))
    print(f"Standard with real_iter = {kmeans.n_iter_} : inertia = {kmeans.inertia_}\n")


import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def silhouette_safe(data, labels):
    if len(set(labels)) > 1 and -1 in labels:
        # только не шум
        mask = labels != -1
        return silhouette_score(data[mask], labels[mask])
    elif len(set(labels)) > 1:
        return silhouette_score(data, labels)
    else:
        return -1  # один кластер или только шум


def kmeans_cubit(data):
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=67)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertia, marker='o')
    plt.title("Метод локтя")
    plt.xlabel("Число кластеров (k)")
    plt.ylabel("Сумма внутрикластерных расстояний")
    plt.show()


def dbs_cubit(data):
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(data)
    distances, _ = neighbors_fit.kneighbors(data)
    distances = np.sort(distances[:, -1])
    plt.plot(distances)
    plt.title("График k-расстояний (k=5)")
    plt.xlabel("Точки")
    plt.ylabel("Расстояние до 5-го соседа")
    plt.show()


def kmeans_clustering_test(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=67)
    kmeans.fit(data)
    print(
        f"silhouette_score of KMeans(n_clusters=2) = {silhouette_score(data, kmeans.fit_predict(data))}")
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, s=10, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, label='Центры кластеров')
    plt.title("Результаты кластеризации методом k-means")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.legend()
    plt.show()

    scaler = StandardScaler()
    scaler.fit(data)
    clustering_1_standard = scaler.transform(data)

    kmeans_standard = KMeans(n_clusters=n_clusters, random_state=67)
    kmeans_standard.fit(clustering_1_standard)
    print(
        f"silhouette_score of standard KMeans(n_clusters=2) = {silhouette_score(clustering_1_standard, kmeans.fit_predict(clustering_1_standard))}")
    plt.scatter(clustering_1_standard[:, 0], clustering_1_standard[:, 1], c=kmeans.labels_, s=10, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, label='Центры кластеров')
    plt.title("Результаты кластеризации методом k-means")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.legend()
    plt.show()



def dbs_clustering_test(data, eps, min_samples):
    dbs = DBSCAN(eps=eps, min_samples=min_samples)
    dbs.fit(data)

    unique_labels = set(dbs.labels_)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(8, 6))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'black'
            marker = 'x'
            label_name = 'Шум'
        else:
            marker = 'o'
            label_name = f'Кластер {label}'

        mask = (dbs.labels_ == label)
        plt.scatter(data[mask, 0], data[mask, 1],
                    c=color, s=30, marker=marker, label=label_name, alpha=0.7)

    plt.title("Результаты кластеризации методом DBSCAN")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    print(f"DBSCAN:  {silhouette_safe(data, dbs.labels_):.3f}")


def agg_clustering_test(data, n_clusters):
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg.fit(data)

    plt.scatter(data[:, 0], data[:, 1], c=agg.labels_, s=10, cmap='viridis')
    plt.title("Результаты кластеризации методом AgglomerativeClustering")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.show()
    print(f"Hierarchical: {silhouette_safe(data, agg.labels_):.3f}")


# print("clustering_1")
# clustering_1 = pd.read_csv('src/clustering_1.csv', delimiter='\t', header=None).values
# kmeans_clustering_test(clustering_1, 2)
# dbs_clustering_test(clustering_1, 0.2, 15)
# agg_clustering_test(clustering_1, 2)

print("clustering_2")
clustering_2 = pd.read_csv('src/clustering_2.csv', delimiter='\t', header=None).values
kmeans_clustering_test(clustering_2, 3)
dbs_clustering_test(clustering_2, 0.3, 8)
agg_clustering_test(clustering_2, 3)

print("clustering_3")
clustering_3 = pd.read_csv('src/clustering_3.csv', delimiter='\t', header=None).values
kmeans_clustering_test(clustering_3, 2)
dbs_clustering_test(clustering_3, 0.31, 9)
agg_clustering_test(clustering_3, 2)


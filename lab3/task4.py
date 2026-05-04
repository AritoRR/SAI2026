import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Загружаем данные (первая колонка — названия штатов)
df = pd.read_csv('src/votes.csv', index_col=0)

# Проверяем, есть ли пропуски
print("Пропуски в данных:")
print(df.isnull().sum().sum(), "пропущенных значений")

# Если есть NaN, можно заполнить средним по столбцу (году)
if df.isnull().sum().sum() > 0:
    df = df.fillna(df.mean())

# Проверяем, есть ли столбцы с нулевой дисперсией (все значения одинаковые)
low_variance_cols = df.columns[df.std() == 0]
if len(low_variance_cols) > 0:
    print("Столбцы с нулевой дисперсией (все штаты одинаково):", list(low_variance_cols))
    # Удаляем такие столбцы, чтобы избежать NaN при стандартизации
    df = df.drop(columns=low_variance_cols)

# Стандартизация
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Дополнительная проверка на нефинитные значения после стандартизации
if not np.all(np.isfinite(df_scaled)):
    print("Внимание: после стандартизации остались нефинитные значения!")
    # Заменяем NaN и inf на 0 (на всякий случай)
    df_scaled = np.nan_to_num(df_scaled)

# Иерархическая кластеризация (метод Ward)
linked = linkage(df_scaled, method='ward')

# Построение дендрограммы
plt.figure(figsize=(12, 8))
dendrogram(linked,
           labels=df.index,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           leaf_rotation=90,
           leaf_font_size=10)
plt.title('Дендрограмма штатов по голосам за республиканцев (1856–1976)')
plt.xlabel('Штаты')
plt.ylabel('Евклидово расстояние (после стандартизации)')
plt.tight_layout()
plt.show()
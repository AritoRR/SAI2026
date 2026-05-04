import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

# Загружаем изображение (замени 'image.jpg' на свой файл)
image_path = 'src/image.jpg'
img = Image.open(image_path)
img_array = np.array(img)

# Сохраняем размеры
h, w, d = img_array.shape

# Преобразуем в двумерный массив пикселей (каждый пиксель - RGB)
pixels = img_array.reshape(-1, 3)

# Количество цветов в сжатой палитре
n_colors = 4  # можно менять

# K-means на цветах пикселей
kmeans = KMeans(n_clusters=n_colors, random_state=42)
kmeans.fit(pixels)

# Центры кластеров — новые цвета палитры
new_colors = kmeans.cluster_centers_.astype(int)

# Метки кластеров для каждого пикселя
labels = kmeans.labels_

# Восстанавливаем изображение: каждый пиксель заменяем на цвет своего кластера
compressed_pixels = new_colors[labels]
compressed_img_array = compressed_pixels.reshape(h, w, d)

# Создаём изображение из массива
compressed_img = Image.fromarray(compressed_img_array.astype('uint8'))

# Отображаем исходное и сжатое рядом
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].set_title('Исходное изображение')
axes[0].axis('off')

axes[1].imshow(compressed_img)
axes[1].set_title(f'Сжатое изображение ({n_colors} цветов)')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Сохраняем результат (опционально)
compressed_img.save('compressed_image.jpg')
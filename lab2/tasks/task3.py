import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 2. Модель
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(7 * 7 * 64, 10)

    def forward(self, X):
        x = self.pool(torch.relu(self.conv1(X)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def fit(self, train_loader, epochs):
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(epochs):
            total_loss = 0.0
            total_batches = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                # data: [batch_size, 1, 28, 28]
                # target: [batch_size]

                optimizer.zero_grad()
                output = self.forward(data)  # [batch_size, 10]
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

            avg_loss = total_loss / total_batches
            print(f'Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}')

    def evaluate(self, test_loader):
        self.eval()  # Переключаем модель в режим оценки (выключает Dropout, если бы он был)

        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = self.forward(data)  # [batch, 10]
                _, predicted = torch.max(output, 1)  # Индекс максимального значения
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100.0 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy
#
#
# # 3. Запуск
# clf = CNN()
# clf.fit(train_loader, epochs=3)
#
# # Оценка
# clf.evaluate(test_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# Если у тебя есть сохранённые веса, раскомментируй следующую строку:
# model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
# Если нет, просто визуализируем необученные (случайные) фильтры.

model.eval()  # переводим в режим оценки


# ----------------------------
# 4. Функция визуализации фильтров (весов) свёрточного слоя
# ----------------------------
def visualize_filters(weights, title="Filters", n_cols=8):
    """
    weights: тензор формы [out_channels, in_channels, H, W]
    """
    # Нормализация в [0,1]
    w = weights.detach().cpu().numpy()
    w_min, w_max = w.min(), w.max()
    w_norm = (w - w_min) / (w_max - w_min + 1e-8)

    out_channels, in_channels, h, w = w.shape
    n_filters = out_channels

    n_rows = int(np.ceil(n_filters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = axes.flatten()

    for i in range(n_filters):
        # Показываем первый канал каждого фильтра
        filt = w_norm[i, 0, :, :]
        axes[i].imshow(filt, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'{i}')

    for j in range(n_filters, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ----------------------------
# 5. Функция извлечения карт признаков (feature maps) для одного изображения
# ----------------------------
def get_feature_maps(model, x):
    """
    x: тензор [1, 1, 28, 28]
    Возвращает словарь с выходами каждого интересующего этапа.
    """
    model.eval()
    with torch.no_grad():
        # Перемещаем на то же устройство, что и модель
        x = x.to(next(model.parameters()).device)

        # Пропускаем через слои вручную, сохраняя промежуточные результаты
        conv1_out = model.conv1(x)
        conv1_relu = F.relu(conv1_out)
        pool1_out = model.pool(conv1_relu)

        conv2_out = model.conv2(pool1_out)
        conv2_relu = F.relu(conv2_out)
        pool2_out = model.pool(conv2_relu)

    return {
        'conv1': conv1_out,
        'conv1_relu': conv1_relu,
        'pool1': pool1_out,
        'conv2': conv2_out,
        'conv2_relu': conv2_relu,
        'pool2': pool2_out
    }


# ----------------------------
# 6. Визуализация карт признаков
# ----------------------------
def visualize_feature_maps(feature_maps_dict, layer_name, max_maps=16, n_cols=4):
    """
    Отображает до max_maps карт признаков из указанного слоя.
    """
    maps = feature_maps_dict[layer_name][0].cpu().numpy()  # [C, H, W]
    n_maps = min(maps.shape[0], max_maps)
    n_rows = int(np.ceil(n_maps / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    axes = axes.flatten()

    for i in range(n_maps):
        axes[i].imshow(maps[i], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'{i}')

    for j in range(n_maps, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Feature Maps: {layer_name}', fontsize=14)
    plt.tight_layout()
    plt.show()


# ----------------------------
# 7. Визуализация фильтров модели
# ----------------------------
print("Визуализация фильтров conv1 (32 фильтра 3x3):")
visualize_filters(model.conv1.weight, title="Conv1 Filters (32 filters)", n_cols=8)

print("\nВизуализация фильтров conv2 (первые 16 из 64, показан только первый канал):")
# Берём только первый канал каждого фильтра conv2
conv2_weights = model.conv2.weight.data  # [64, 32, 3, 3]
# Усреднять по входным каналам или брать первый? Часто берут первый канал или среднее.
# Покажем первый канал:
first_channel_weights = conv2_weights[:, 0:1, :, :]  # сохраняем размерность [64, 1, 3, 3]
visualize_filters(first_channel_weights, title="Conv2 Filters (first channel of each)", n_cols=8)
# ----------------------------
# 8. Визуализация карт признаков для одного тестового изображения
# ----------------------------
# Создаём отдельный загрузчик с batch_size=1 для удобства
vis_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
data_iter = iter(vis_loader)
image, label = next(data_iter)  # image: [1, 1, 28, 28], label: [1]

# Показываем оригинал
plt.imshow(image[0, 0].cpu(), cmap='gray')
plt.title(f'Original Image (Label: {label.item()})')
plt.axis('off')
plt.show()

# Извлекаем feature maps
fmaps = get_feature_maps(model, image)

# Визуализируем для разных слоёв
visualize_feature_maps(fmaps, 'conv1', max_maps=16, n_cols=4)
visualize_feature_maps(fmaps, 'conv1_relu', max_maps=16, n_cols=4)
visualize_feature_maps(fmaps, 'pool1', max_maps=16, n_cols=4)
visualize_feature_maps(fmaps, 'conv2', max_maps=16, n_cols=4)
visualize_feature_maps(fmaps, 'conv2_relu', max_maps=16, n_cols=4)
visualize_feature_maps(fmaps, 'pool2', max_maps=16, n_cols=4)

print("Визуализация завершена.")
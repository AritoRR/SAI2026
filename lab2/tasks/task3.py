import torch.nn as nn
import torch.optim as optim
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


# 3. Запуск
clf = CNN()
clf.fit(train_loader, epochs=3)

# Оценка
clf.evaluate(test_loader)

import pandas as pd
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_frame_glass = pd.read_csv('../source/data_3/glass.csv', delimiter=",", usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

X_glass = data_frame_glass.iloc[:, :-1]
y_glass = data_frame_glass.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X_glass, y_glass, random_state=52
)


def calculate_distance(train_instance, test_instance, metric='euclidean', p=3):
    if metric == 'manhattan':
        return sum(abs(train_instance[i] - test_instance[i])
                   for i in range(len(train_instance)))

    elif metric == 'chebyshev':
        # Расстояние Чебышева (L-infinity)
        return max(abs(train_instance[i] - test_instance[i])
                   for i in range(len(train_instance)))

    elif metric == 'minkowski':
        # Расстояние Минковского (L-p)
        return (sum(abs(train_instance[i] - test_instance[i]) ** p
                    for i in range(len(train_instance)))) ** (1 / p)
    else:
        return math.sqrt(sum((train_instance[i] - test_instance[i]) ** 2
                             for i in range(len(train_instance))))


def get_neighbors(X_train, y_train, test_instance, metric, k=1):
    distances = []
    for i in range(len(X_train)):
        dist = calculate_distance(X_train.iloc[i].values, test_instance, metric)
        distances.append((y_train.iloc[i], dist))
    distances.sort(key=lambda elem: elem[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors


def prediction(neighbors):
    count = {}
    for instance in neighbors:
        if instance in count:
            count[instance] +=1
        else :
            count[instance] = 1
    target = max(count.items(), key=lambda x: x[1])[0]
    return target

def calculate_accuracy(test, test_prediction):
    correct = 0
    for i in range (len(test)):
        if test[i][-1] == test_prediction[i]:
            correct += 1
    return (correct / len(test))

def calculate_knn(k, metric):
    predictions = []
    for i in range(len(X_test)):
        neighbors = get_neighbors(X_train, y_train, X_test.iloc[i].values, metric, k)
        predictions.append(prediction(neighbors))
    correct = sum(1 for i in range(len(y_test)) if y_test.iloc[i] == predictions[i])
    accuracy = correct / len(y_test)
    error = 1 - accuracy
    return predictions, accuracy, error




# ПРОСТОЙ ГРАФИК ЗАВИСИМОСТИ ОШИБКИ ОТ K
def draw_plot(metric):
    k_values = range(1, 16)  # проверяем k от data_1 до 15
    test_errors = []

    for k in k_values:
        test_errors.append(calculate_knn(k, metric)[2])
    # Простой график
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, test_errors, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Количество соседей (k)')
    plt.ylabel('Ошибка классификации')
    plt.title('Зависимость ошибки от k')
    plt.grid(True)
    plt.xticks(k_values)
    plt.show()
    best_k = k_values[test_errors.index(min(test_errors))]
    print(f"{metric}: Лучшее k = {best_k} с ошибкой {min(test_errors):.3f}")

draw_plot('euclidean')
draw_plot('manhattan')
draw_plot('chebyshev')
draw_plot('minkowski')

new_glass = [1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]

neighbors = get_neighbors(X_train, y_train, new_glass, 'euclidean', 4)
print(prediction(neighbors))

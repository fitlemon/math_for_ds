import numpy as np


def predict(X, weights, bias):
    y_pred = bias

    for i in range(len(X)):
        y_pred += weights[i] * X[i]

    return y_pred


def linear_regression(X, y, learning_rate=0.01, num_iterations=10):
    n_samples, n_features = X.shape
    weights = [0] * n_features
    bias = 0

    for _ in range(num_iterations):
        # Инициализация градиентов
        dw = [0] * n_features
        db = 0

        """!!! Обратить внимание на этот цикл - по каждому признаку объекта
        происходит изменение соответствующего элемента вектора градиента
        """
        # Проход по каждому объекту
        for i in range(n_samples):
            # Предсказание
            y_pred = predict(X[i], weights, bias)

            # Обновление градиентов
            error = y_pred - y[i]
            for j in range(n_features):
                dw[j] += error * X[i][j]
            db += error

        # Обновление весов и смещения
        for j in range(n_features):
            weights[j] -= (learning_rate * dw[j]) / n_samples
        bias -= (learning_rate * db) / n_samples

    return weights, bias


# Пример данных
X = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8], [7, 8, 9, 10, 11]])
y = np.array([15, 30, 45])

# Обучение модели
weights, bias = linear_regression(X, y, num_iterations=100)

# Вывод результатов
print("Веса:", weights)
print("Смещение:", bias)

# Пример предсказания
x_test = np.array([10, 11, 12, 13, 14])
y_pred = predict(x_test, weights, bias)
print("Предсказанное значение:", y_pred)

from sklearn.base import BaseEstimator

from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, auc

from sklearn.model_selection import KFold
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=10000, n_features=10, n_informative=5, n_redundant=5, random_state=999
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=999
)


class MyLogisticRegression(BaseEstimator):
    def __init__(
        self,
        gd_type: str = "stochastic",
        tolerance: float = 1e-6,
        max_iter: int = 1000,
        eta: float = 1e-2,
        w0: np.array = None,
    ) -> None:
        """
        Аргументы:
          gd_type: Вид градиентного спуска ('full' или 'stochastic').

          tolerance: Порог для остановки градиетного спуска.

          max_iter: Максимальное количество шагов в градиентном спуске.

          eta: Скорость обучения (learning rate).

          w0: Массив размерности d (d — количество весов в оптимизации).
              Начальные веса.
        """
        self.gd_type = gd_type
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.eta = eta
        self.w0 = w0
        self.w = None
        self.loss_history = None

    def fit(self, X: np.array, y: np.array):
        """Обучаем модель на training данных. Сохраняйте значении функции потерь после каждой итерации.

        Аргументы:
          X: Данные на обучение.

          y: Целевая переменная.

        Возвращает:
          self: Обученный регрессор.
        """
        self.loss_history = []
        self.w0 = np.random.rand(X.shape[1])
        self.w = np.random.rand()
        for iter in range(self.max_iter):
            if self.gd_type == "stochastic":
                sample_indice = np.random.randint(0, len(y) + 1)
                loss = self.calc_loss(
                    np.array([X[sample_indice]]), np.array([y[sample_indice]])
                )
                grad = self.calc_gradient(
                    np.array([X[sample_indice]]), np.array([y[sample_indice]])
                )
                new_w0 = self.w0 - self.eta * grad
                new_w = self.w - self.eta * (
                    self.predict_proba(np.array([X[sample_indice]]))
                    - np.array([y[sample_indice]])
                )
            elif self.gd_type == "full":
                loss = self.calc_loss(X, y)
                grad = self.calc_gradient(X, y)
                new_w0 = self.w0 - self.eta * grad
                new_w = self.w - self.eta * (self.predict_proba(X) - y)
            else:
                raise Exception("Not valid GD type")
            self.loss_history.append(loss)
            if (
                np.linalg.norm(new_w0 - self.w0) < self.tolerance
                and np.linalg.norm(new_w - self.w) < self.tolerance
            ):
                break
            else:
                self.w0 = new_w0
                self.w = new_w
        return self

    def predict_proba(self, X: np.array) -> np.array:
        """Вычислите вероятности положительного и отрицательного классов для каждого наблюдения.

        Аргументы:
          X: Массив размером (n, d).
             Данные.

        Возвращает:
             Массив размером (n, 2).
             Предсказанные вероятности.
        """
        if self.w is None:
            raise Exception("Not trained yet")
        # your code
        if self.gd_type == "stochastic":
            z = X.dot(self.w0.T) + self.w
        elif self.gd_type == "full":
            z = X.dot(self.w0) + self.w
        else:
            raise Exception("Not valid GD type")
        p = 1 / (1 + np.exp(-z))
        return np.array([p])

    def predict(self, X: np.array) -> np.array:
        """Предсказание метки класса для каждого наблюдения.

        Аргументы:
          X: Массив размером (n, d).
             Данные.

        Возвращает:
             Массив размером (n,).
             Предсказанные метки классов.
        """
        if self.w is None:
            raise Exception("Not trained yet")
        # your code
        return self.predict_proba(X)

    def calc_gradient(self, X: np.array, y: np.array) -> np.array:
        """Вычислите градиент функции потерь после каждой итерации.

        Аргументы:
          X: Массив размером (n, d), n может быть равно 1, если выбран 'stochastic'.
          y: Массив размером (n,).

        Возвращает:
          Массив размером (d,).
          Градиент функции потерь после текущей итерации.
        """
        # your code
        p = self.predict_proba(X)
        print(p)
        print(y)
        print(X)
        grad = (p - y).dot(X) / len(y)
        return grad

    def calc_loss(self, X: np.array, y: np.array) -> float:
        """Вычислите значение функции потерь после каждой итерации.

        Аргументы:
          X: Массив размером (n, d).
          y: Массив размером (n,).

        Возвращает:
          Значение функции потерь после текущей итерации.
        """
        # your code
        p = self.predict_proba(X)
        loss = (-1) * sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / len(y)

        return loss


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

x_ = np.array([[1, 2, 3], [4, 5, 6]])
w0_ = np.array([0, 0, 1])
w_ = 0
y_ = np.array([0, 1])

z = x_.dot(w0_) + w_

# logregr = MyLogisticRegression(gd_type="full")
# model = logregr.fit(X_train, y_train)
# len(model.loss_history)
# print(model.loss_history)
# print(np.log(np.array([1.00000000e000, 1.01633791e-196])))
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer

scaler = Normalizer()

# logregr_full = MyLogisticRegression(gd_type="full", eta=1e-1)
# model_full = logregr_full.fit(scaler.fit_transform(X_train), y_train)
logregr_sgd = MyLogisticRegression(gd_type="stochastic", eta=1e-1)
model_sgd = logregr_sgd.fit(scaler.fit_transform(X_train), y_train)
fig = plt.figure(figsize=(10, 4))
axes = fig.add_axes([0, 0, 1, 1])
# axes.plot(model_full.loss_history, label="Full GD")
axes.plot(model_sgd.loss_history, label="SGD GD")
axes.set_title("Изменение Lost функции во время обучения", fontsize=16)

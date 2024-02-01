import numpy as np
from KDTree import KDTreeNode


class KNNclassifiller(object):

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", metric: str = 'minkowski'):
        # Число учитываемых соседей
        self.n_neighbors = n_neighbors
        # Вид метрики
        self.metric = metric
        # Способ вычисления весов
        self.weights = weights

    def fit(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        # Словарь уникальных классов
        self.uniq_y = dict(list([elem, 0]) for elem in np.unique(y_train))

    def predict(self, x_pred):
        # Список предсказаний
        ans = []
        for x in x_pred:
            # Массив [[расстояние, значение класса],...]
            distance = np.array(list([self.eval_metric(x, self.x[i]), self.y[i]] for i in range(len(self.y))))
            # Сортируем индексы по расстоянию
            sorted_index = np.argsort(distance[:, 0])
            # Список k ближайших соседей
            neighbors = distance[sorted_index][:self.n_neighbors][:, 1]

            temp_dict = self.uniq_y.copy()
            if self.weights == "uniform":
                for key in neighbors:
                    temp_dict[key] += 1
            if self.weights == "distance":
                for i in range(self.n_neighbors):
                    temp_dict[neighbors[i]] += (self.n_neighbors - i + 1) / self.n_neighbors
            # Поиск самого популярного соседа
            ans.append(sorted(temp_dict, key=temp_dict.__getitem__)[-1])
        return ans

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.sum(y_pred == y)/len(y)

    def eval_metric(self, x1, x2):
        return {"minkowski": np.sqrt(np.sum((x1 - x2) ** 2))
                }[self.metric]
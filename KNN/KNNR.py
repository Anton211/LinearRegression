import numpy as np
from KDTree import KDTreeNode


class KNNregressor(object):

    def __init__(self, n_neighbors: int = 5, h=1, weights: str = "uniform", metric: str = 'minkowski'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        # "Ширина окна"
        self.h = h

    def fit(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        # Словарь уникальных классов
        self.uniq_y = dict(list([elem, 0]) for elem in np.unique(y_train))

    def predict(self, x_pred):
        ans = []
        for x in x_pred:
            distance = np.array(list([self.eval_metric(x, self.x[i]), self.y[i]] for i in range(len(self.y))))
            sorted_index = np.argsort(distance[:, 0])
            neighbors = distance[sorted_index][:self.n_neighbors][:, 1]
            sort_distance = distance[sorted_index][:self.n_neighbors][:, 0]

            temp_dict = self.uniq_y.copy()
            if self.weights == "uniform":
                for key in neighbors:
                    temp_dict[key] += 1
            if self.weights == "distance":
                for i in range(self.n_neighbors):
                    temp_dict[neighbors[i]] += (self.n_neighbors - i + 1) / self.n_neighbors
            if self.weights == "Exp":
                for i in range(self.n_neighbors):
                    temp_dict[neighbors[i]] += 0.39894 * np.exp(-2 * (sort_distance[i] / self.h) ** 2)
            # Высчитываем среднее значение по соседям
            avg = 0
            count = 0
            for elem in list(temp_dict.items()):
                avg += float(elem[0]) * float(elem[1])
                count += float(elem[1])
            ans.append(avg / count)
        return ans

    def score(self, x, y):
        y_avg = np.average(y)
        return 1 - np.sum((y - self.predict(x)) ** 2) / np.sum((y - y_avg) ** 2)

    def eval_metric(self, x1, x2):
        return {"minkowski": np.sqrt(np.sum((x1 - x2) ** 2))
                }[self.metric]
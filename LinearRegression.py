import numpy as np


class LinearRegression(object):

    def __init__(self, method='stochastic'):
        self.method = method

    def fit(self, x_train, y_train):
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        if x_train.ndim == 1:
            x_train = np.concatenate((np.ones((len(x_train), 1)),
                                      x_train.reshape(len(x_train), 1)), axis=1)
            count_weight = len(x_train[0])
        else:
            x_train = np.concatenate((np.ones((len(x_train), 1)),
                                      x_train.reshape(len(x_train), len(x_train[0]))), axis=1)
            count_weight = len(x_train[0])

        w = np.zeros(count_weight) + np.random.normal(0, 1)
        grad = np.zeros(count_weight)

        if self.method == "classic":

            w_next = w + 0.0000001
            fit_rate = 0.001

            while np.linalg.norm(w_next - w) > 1e-8:

                err = y_train - w.dot(x_train.T)
                grad[0] = np.array([np.sum(-2 * err)])
                for i in range(count_weight - 1):
                    grad[i + 1] = -2 * x_train[:, i + 1].T.dot(err)

                w, w_next = w_next, w_next - fit_rate * grad

        elif self.method == "stochastic":

            w_next = w + 0.0000001
            fit_rate = 0.05
            batch_step = 5
            num_epoch = 100

            for _ in range(num_epoch):

                step_num = 0

                while batch_step * step_num <= len(x_train):

                    x_batch = x_train[step_num * batch_step: batch_step * (step_num + 1)]
                    y_batch = y_train[step_num * batch_step: batch_step * (step_num + 1)]

                    err = y_batch - w.dot(x_batch.T)
                    grad[0] = np.array([np.sum(-2 * err)]) / batch_step
                    for i in range(count_weight - 1):
                        grad[i + 1] = -2 * x_batch[:, i + 1].T.dot(err) / batch_step

                    w, w_next = w_next, w_next - fit_rate * grad

                    step_num += 1

        self.weights = w

    def predict(self, x):
        if x.ndim == 1:
            x = np.concatenate((np.ones((len(x), 1)), x.reshape(len(x), 1)), axis=1)
        else:
            x = np.concatenate((np.ones((len(x), 1)), x.reshape(len(x), len(x[0]))), axis=1)
        return self.weights.dot(x.T)

    def score(self, x, y):
        y_avg = np.average(y)
        return 1 - np.sum((y - self.predict(x)) ** 2) / np.sum((y - y_avg) ** 2)
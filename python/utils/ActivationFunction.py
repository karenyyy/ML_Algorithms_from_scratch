import numpy as np


## sigmoid
class Sigmoid:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


class Softmax:
    ## X.shape => (training_samples, n_features)
    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def gradient(self, x):
        p = self.softmax(x)
        return p * (1 - p)


class TanH:
    @staticmethod
    def tanh(x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.tanh(x), 2)


class ReLU:
    @staticmethod
    def relu(x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class LeakyReLU:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def leaky_relu(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)


class ELU:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def elu(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return np.where(x >= 0.0, 1, self.elu(x) + self.alpha)


class SoftPlus:
    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1 / (1 + np.exp(-x))

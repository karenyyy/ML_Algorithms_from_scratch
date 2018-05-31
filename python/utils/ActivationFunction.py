import numpy as np


## sigmoid
class Sigmoid:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

from __future__ import print_function, division
import numpy as np
import math

from utils.ActivationFunction import Sigmoid
from utils.utils import diagonal


class LogisticRegression:
    """ Logistic Regression classifier.
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If
        false then we use batch optimization by least squares.
    """

    def __init__(self, learning_rate=.1, gradient_descent=True):
        self.w = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.activate = Sigmoid()

    # instead of using the naive approach by initializing weights as 0,
    # a better way of initialization would be initializing as white-noise
    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (1, n_features))  # try to avoid using rank 1

    def fit(self, X, y, n_iterations=10000):
        self._initialize_parameters(X)
        for i in range(n_iterations):
            y_pred = self.activate.sigmoid(self.w.dot(X.T))
            if self.gradient_descent:
                # print (self.w.shape)
                # print (y_pred.shape, y.shape)
                y = y.reshape(1, -1)
                # print (y_pred.shape, y.shape)
                # print ((y_pred - y).dot(X).shape)
                self.w -= self.learning_rate * (y_pred - y).dot(X)

            else:
                # Make a diagonal matrix of the sigmoid gradient column vector
                diag_gradient = diagonal(self.activate.gradient(X.dot(self.w)))
                # Batch optimization:
                self.w = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(
                    diag_gradient.dot(X).dot(self.w) + y - y_pred)

    def predict(self, X):
        y_pred = np.round(self.activate.sigmoid(X.dot(self.w.T))).astype(int)
        return y_pred

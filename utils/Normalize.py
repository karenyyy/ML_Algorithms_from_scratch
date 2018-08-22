import numpy as np


def normalize(X, axis=1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


# print(normalize(np.array([[0, 3, 4], [4, 5, 6]])))

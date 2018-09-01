import numpy as np
from itertools import combinations_with_replacement

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def cal_entropy(y):
    # 越隨機的信源的熵越大
    log2 = lambda x: np.log(x) / np.log(2)
    entropy = 0
    for y_unique in np.unique(y):
        p = len(y[y == y_unique]) / len(y)
        entropy += -p * log2(p)
    return entropy


def diagonal(x):
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] = x[i]
    return m


def normalize(X, axis=1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    split_i = len(y) - int(len(y) * test_size)
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2), axis=0))


def cal_covariance_matrix(X):
    mean = np.ones(X.shape) * X.mean(axis=0)
    n_samples = X.shape[0]
    variance = (1 / n_samples) * (X - mean).T.dot(X - mean)
    return variance


def mean_squared_error(y_test, y_pred):
    return np.mean(np.sqrt(np.square(np.abs(y_test - y_pred))))


def polynomial_features(X, degree):
    n_samples, n_features = X.shape

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs

    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))

    for i, index_combs in enumerate(combinations):
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new

# if __name__ == '__main__':
#     print(euclidean_distance(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])))

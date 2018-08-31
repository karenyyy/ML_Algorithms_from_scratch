import numpy as np
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error


def load_dataset(all_features=True, dim=2):
    diabetes = datasets.load_diabetes()
    y = diabetes.target
    if all_features:
        X = diabetes.data
    else:
        X = diabetes.data[:, np.newaxis, dim]
    return X, y


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def plot_scatter(X_train, y_train, X_test, y_test):
    cmap = plt.get_cmap('viridis')
    m1 = plt.scatter(x=X_train, y=y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(x=X_test, y=y_test, color=cmap(0.5), s=10)
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    return plt


class Regression(object):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.regression_model = None

    def fit(self):
        self.regression_model.fit(self.X_train, self.y_train)

    def predict(self):
        return self.regression_model.predict(self.X_test)

    def loss(self, y_real, y_pred, metric):
        if metric == 'mse':
            error = mean_squared_error(y_true=y_real, y_pred=y_pred)
        elif metric == 'mse_log':
            error = mean_squared_log_error(y_true=y_real, y_pred=y_pred)
        elif metric == 'mse_abs':
            error = mean_absolute_error(y_true=y_real, y_pred=y_pred)
        return error

    def plot_line(self, X, y_pred, plt, error, metric):
        return NotImplemented


class LinearRegression(Regression):
    def __init__(self, X_train, X_test, y_train, y_test):
        super(LinearRegression, self).__init__(X_train, X_test, y_train, y_test)
        self.regression_model = linear_model.LinearRegression()

    def fit(self):
        super(LinearRegression, self).fit()

    def predict(self):
        return super(LinearRegression, self).predict()

    def plot_line(self, X, y_pred, plt, error, metric):
        plt.plot(X,
                 y_pred,
                 color='black',
                 linewidth=2,
                 label="Prediction")
        plt.title("linear-" + metric + ": {0:.7g}\n".format(error), fontsize=10)


class KNNRegression(Regression):
    def __init__(self, X_train, X_test, y_train, y_test,
                 dist='euclidean',
                 neighbors=20,
                 weights='uniform',
                 algorithm='auto'):
        super(KNNRegression, self).__init__(X_train, X_test, y_train, y_test)
        self.dist = dist
        self.neighbors = neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.regression_model = KNeighborsRegressor(n_neighbors=self.neighbors,
                                                    weights=self.weights,
                                                    algorithm=self.algorithm,
                                                    metric=self.dist)

    def fit(self):
        super(KNNRegression, self).fit()

    def predict(self):
        return super(KNNRegression, self).predict()

    def plot_line(self, X, y_pred, plt, error, metric):
        plt.plot(X,
                 y_pred,
                 color='red',
                 linewidth=0.1,
                 label="Prediction")
        plt.title("knn-" + metric + ": {0:.7g}\n".format(error), fontsize=10)


def grid_search_comparison(lr_err, y_test, metric='mse', neighbors=200):
    kr_errs = []
    for neighbor in range(3, neighbors):
        kr = KNNRegression(neighbors=neighbor)
        kr.fit()
        y_pred = kr.predict()
        kr_err = kr.loss(y_real=y_test, y_pred=y_pred, metric=metric)
        kr_errs.append(kr_err)

    fig = plt.figure(figsize=(8, 8))
    plt.plot(range(3, neighbors), kr_errs)
    plt.hlines(lr_err, xmin=1, xmax=neighbors, colors='red')
    plt.title('Loss: {} Neighbors: {}'.format(metric, neighbors))
    return fig


def test_all_features():
    metric = ['mse', 'mse_log', 'mse_abs']
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    lr = LinearRegression()
    lr.fit()
    lr_y_pred = lr.predict()

    lr_mse = lr.loss(y_real=y_test, y_pred=lr_y_pred, metric=metric[0])
    lr_mse_log = lr.loss(y_real=y_test, y_pred=lr_y_pred, metric=metric[1])
    lr_mse_abs = lr.loss(y_real=y_test, y_pred=lr_y_pred, metric=metric[2])
    lr_err = [lr_mse, lr_mse_log, lr_mse_abs]

    for idx, m in enumerate(metric):
        fig = grid_search_comparison(lr_err=lr_err[idx], y_test=y_test, metric=m)
        fig.savefig(fname='plots/{}.png'.format(m))


def test_single_feature():
    n_features = 10
    metric = ['mse', 'mse_log', 'mse_abs']

    for d in range(2, n_features):
        fig = plt.figure(figsize=(8, 8))

        X, y = load_dataset(all_features=False, dim=d)
        X_train, X_test, y_train, y_test = split_dataset(X, y)
        plt_ = plot_scatter(X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test)

        lr = LinearRegression(X_train, X_test, y_train, y_test)
        lr.fit()
        lr_y_pred = lr.predict()

        kr = KNNRegression(X_train, X_test, y_train, y_test, neighbors=20)
        kr.fit()
        kr_y_pred = kr.predict()

        for me in metric:
            lr_err = lr.loss(y_real=y_test, y_pred=lr_y_pred, metric=me)
            kr_err = kr.loss(y_real=y_test, y_pred=kr_y_pred, metric=me)
            lr.plot_line(X=X_test, y_pred=lr_y_pred, plt=plt_, metric=me, error=lr_err)
            kr.plot_line(X=X_test, y_pred=kr_y_pred, plt=plt_, metric=me, error=kr_err)
            fig.savefig(fname='out_pred_plots/' + me + '-feature{}.png'.format(d))


if __name__ == '__main__':
    test_single_feature()

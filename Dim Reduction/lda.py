import numpy as np
from utils.util import cal_covariance_matrix, train_test_split, accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class LDA(object):
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        X1 = X[y == 0]
        X2 = X[y == 1]

        cov1 = cal_covariance_matrix(X1)
        cov2 = cal_covariance_matrix(X2)
        cov_tot = cov1 + cov2

        mean1 = X1.mean(0)
        mean2 = X2.mean(0)
        mean_diff = np.atleast_1d(mean1 - mean2)

        # Determine the vector which when X is projected onto it best separates the
        # data by class. w = (mean1 - mean2) / (cov1 + cov2)
        self.w = np.linalg.pinv(cov_tot).dot(mean_diff)


    def predict(self, X):
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred


def main():
    data = load_iris()
    X = data.data
    y = data.target

    X = X[y != 2]
    y = y[y != 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    fig = plt.figure(figsize=(8,8))
    plt.plot(X_test, y_pred)
    plt.title('LDA')
    plt.show()
    fig.savefig(fname='lda_iris_test.png')


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, make_regression
from sklearn import linear_model

from utils.util import mean_squared_error, train_test_split
from supervised.Regression.regression import LinearRegression


def main():
    diabetes = load_diabetes()

    X, y = diabetes.data[:, np.newaxis, 2], diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    n_samples, n_features = np.shape(X)

    model = linear_model.LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean squared error: %s" % mse)

    y_pred_line = model.predict(X)

    cmap = plt.get_cmap('viridis')

    print(X_train.shape, y_train.shape)
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.suptitle("Linear Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()

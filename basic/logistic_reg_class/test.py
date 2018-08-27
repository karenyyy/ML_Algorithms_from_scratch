from sklearn import datasets

from basic.logistic_reg_class.LogisticReg import LogisticRegression
from utils.Accuracy import accuracy_score
# Import helper functions
from utils.Normalize import normalize
from utils.Split import train_test_split


# train_test_split, accuracy_score, Plot


def main():
    # Load dataset
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_percent=0.33, seed=1)

    clf = LogisticRegression(gradient_descent=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    assert(accuracy.shape == y_test.shape)
    print("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    # Plot().plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)


if __name__ == "__main__":
    main()

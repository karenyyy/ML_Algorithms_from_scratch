import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from utils.util import euclidean_distance
from utils.visualize import plot

class KNN():
    def __init__(self, k=5):
        self.k = k

    def _vote(self, neighbor_labels):
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()

    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])
        for i, test_sample in enumerate(X_test):
            idx = np.argsort([euclidean_distance(test_sample, x) for x in X_train])[:self.k]
            k_nearest_neighbors = np.array([y_train[i] for i in idx])
            y_pred[i] = self._vote(k_nearest_neighbors)

        return y_pred



def main():
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = KNN(k=5)
    y_pred = clf.predict(X_test, X_train, y_train)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    # Reduce dimensions to 2d using pca and plot the results
    plot(X=X_test, y=y_pred, title='KNN (dimension reduction by PCA)', accuracy=accuracy,
         legend_labels=data.target_names)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

def PCA(X, n):
    diff = X - X.mean(axis=0)
    covariance_matrix = diff.T.dot(diff)/(X.shape[0]-1)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort the eigenvalues and corresponding eigenvectors descending and select the first n
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx][:, :n]

    # Project the data onto principal components
    X_transformed = X.dot(eigenvectors)
    return X_transformed


def main():
    data = datasets.load_digits()
    X = data.data
    y = data.target
    print(X.shape, y.shape)

    # Project the data onto the 2 pcs
    X_trans = PCA(X, 2)

    # pc1
    x1 = X_trans[:, 0]
    print('pc1: ', x1)
    # pc2
    x2 = X_trans[:, 1]
    print('pc2: ', x2)

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

    class_distr = []
    # Plot the different class distributions
    for i, l in enumerate(np.unique(y)):
        print(l, y)
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

    plt.legend(class_distr, y, loc=1)

    # Axis labels
    plt.title("PCA MNIST 0-9")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


if __name__ == "__main__":
    main()
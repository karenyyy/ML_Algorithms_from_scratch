import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from utils.util import cal_entropy, accuracy_score
from supervised.DecisionTree.DecisionTree import DecisionTree


class DTClassifier(DecisionTree):
    def __init__(self):
        super(DTClassifier, self).__init__()

    def info_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        entropy = cal_entropy(y)
        info_gain = entropy - p * \
                              cal_entropy(y1) - (1 - p) * \
                                                cal_entropy(y2)

        return info_gain

    def majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self.criteria_value = self.info_gain
        self.leaf_value = self.majority_vote
        super(DTClassifier, self).fit(X, y)


if __name__ == '__main__':


    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = DTClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

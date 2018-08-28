import numpy as np
from supervised.DecisionTree.SplitNode import SplitNode


class DecisionTree(object):
    def __init__(self, min_samples_split=2, 
                       min_criteria_value=1e-7,
                       max_depth=float("inf")):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_criteria_value = min_criteria_value
        self.max_depth = max_depth
        self.criteria_value = None
        self.leaf_value = None

    def build_tree(self, X, y, current_depth=0):
        largest_criteria_value = 0
        best_criteria = None
        best_sets = None

        # in case y=(n_samples, )
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        X_y = np.concatenate((X, y), axis=1)

        n_samples, n_features = X.shape

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the criteria_value for each feature
            for feature_index in range(n_features):
                print('feature_index: ', feature_index)
                feature_values = np.expand_dims(X[:, feature_index], axis=1)
                print('feature_values: ', feature_values)
                unique_values = np.unique(feature_values)
                for threshold in unique_values:
                    # X_y_left = X_y[feature_index] >= threshold
                    # X_y_right = X_y[feature_index] >= threshold
                    X_y_left, X_y_right = self.feature_divide(X_y, feature_index, threshold)
                    print(X_y_left, X_y_right)
                    if len(X_y_left) > 0 and len(X_y_right) > 0:
                        y_left = X_y_left[:, n_features:]
                        y_right = X_y_right[:, n_features:]
                        # criteria_value calculation here depends on what the DTree is used for
                        # if it is used for classification, then it is info_gain
                        # if it is used for regression, then it is variance_reduction
                        criteria_value = self.criteria_value(y, y_left, y_right)

                        # get the max criteria value
                        if criteria_value > largest_criteria_value:
                            # update the criteria_value value
                            largest_criteria_value = criteria_value
                            best_criteria = {"feature_index": feature_index,
                                             "threshold": threshold}
                            best_sets = {
                                "leftX": X_y_left[:, :n_features],  # X of left subtree
                                "lefty": X_y_left[:, n_features:],  # y of left subtree
                                "rightX": X_y_right[:, :n_features],  # X of right subtree
                                "righty": X_y_right[:, n_features:]  # y of right subtree
                            }

        if largest_criteria_value > self.min_criteria_value:
            # Build subtrees recursively
            true_branch = self.build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self.build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)

            return SplitNode(feature_index=best_criteria["feature_index"],
                             threshold=best_criteria["threshold"],
                             true_branch=true_branch,
                             false_branch=false_branch)

        else:
            leaf_value = self.leaf_value(y)
            return SplitNode(value=leaf_value)

    def feature_divide(self, X, feature_index, threshold):
        split_rule = lambda sample: sample[feature_index] >= threshold
        X_1 = np.array([sample for sample in X if split_rule(sample)])
        X_2 = np.array([sample for sample in X if not split_rule(sample)])
        return np.array([X_1, X_2])

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_helper(self, x, node=None):
        if node is None:
            node = self.root

        # If we have a value (i.e we're at a leaf), then directly return value as the prediction
        if node.value is not None:
            return node.value

        feature_value = x[node.feature_index]

        # start from the false branch again and predict recursively util reaching the leaf node
        current_branch = node.false_branch
        if feature_value >= node.threshold:
            current_branch = node.true_branch

        # Test subtree
        return self.predict_helper(x, current_branch)

    def predict(self, X):
        y_pred = [self.predict_helper(sample) for sample in X]
        return y_pred

    def print_tree(self, node=None, indent=" "):
        if not node:
            node = self.root

        # If we're at leaf => print the label
        if node.value is not None:
            print(node.value)
        else:
            # Print test
            print("%s:%s? " % (node.feature_index, node.threshold))
            # Print the true scenario
            print("%sT->" % indent, end="")
            self.print_tree(node.true_branch, indent + indent)
            # Print the false scenario
            print("%sF->" % indent, end="")
            self.print_tree(node.false_branch, indent + indent)

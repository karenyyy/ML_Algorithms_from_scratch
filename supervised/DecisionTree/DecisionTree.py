import numpy as np
from DecisionTree.utils import divide_on_feature

from supervised.DecisionTree.DecisionNode import DecisionNode


class DecisionTree(object):
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.impurity = None
        self.leaf_value = None
        self.one_dim = None
        self.loss = loss

    def build_tree(self, X, y, current_depth=0):

        largest_impurity = 0
        best_criteria = None
        best_sets = None

        # in case y=(n_samples, )
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = X.shape

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_index in range(n_features):
                print(feature_index)
                feature_values = np.expand_dims(X[:, feature_index], axis=1)
                print(feature_values)
                unique_values = np.unique(feature_values)
                for threshold in unique_values:
                    Xy1, Xy2 = divide_on_feature(Xy, feature_index, threshold)
                    print(Xy1, Xy2)
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]
                        # impurity calculation here depends on what the DTree is used for
                        # if it is used for classification, then it is info_gain
                        # if it is used for regression, then it is variance_reduction
                        impurity = self.impurity(y, y1, y2)
                        if impurity > largest_impurity:
                            # update the impurity value
                            largest_impurity = impurity
                            best_criteria = {"feature_index": feature_index,
                                             "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],  # X of left subtree
                                "lefty": Xy1[:, n_features:],  # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]  # y of right subtree
                            }

        if largest_impurity > self.min_impurity:
            # Build subtrees recursively
            true_branch = self.build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self.build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)

            return DecisionNode(feature_index=best_criteria["feature_index"],
                                threshold=best_criteria["threshold"],
                                true_branch=true_branch,
                                false_branch=false_branch)

        else:
            leaf_value = self.leaf_value(y)
            return DecisionNode(value=leaf_value)

    def predict_helper(self, x, tree=None):
        if tree is None:
            tree = self.root

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = x[tree.feature_index]

        # Determine if we will follow left or right branch
        current_branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                current_branch = tree.true_branch
        elif feature_value == tree.threshold:
            current_branch = tree.true_branch

        # Test subtree
        return self.predict_helper(x, current_branch)

    def predict(self, X):
        y_pred = [self.predict_helper(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value is not None:
            print(tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print("%s:%s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)
class DecisionNode(object):
    """Class that represents a decision node or leaf in the decision tree

    Parameters:
    -----------
    feature_index: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_index against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """

    def __init__(self, feature_index=None,
                 threshold=None,
                 value=None,
                 true_branch=None,
                 false_branch=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

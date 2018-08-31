class SplitNode(object):
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

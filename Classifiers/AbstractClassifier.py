"""Abstract Classifer.
All Classifiers will inherit this class.
"""


class AbstractClassifier:

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def unfit(self):
        pass

    class TreeNode:
        def __init__(self):
            self.Sons = {}
            self.attribute = None
            self.value = None
            self.leaf = False
            self.node_classifier = None
            self.element_num_in_node = None
            self.parent = None

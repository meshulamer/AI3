from .AbstractClassifier import AbstractClassifier
import pandas as pd


class CSTDIDT(AbstractClassifier):
    # Pass positive/negative that you would like to minimize to higher_loss, and certainty_ratio would be the scale.
    # For example, if getting a false negative is 10 times worse then getting a false positive then pass 10
    def __init__(self, att_choice_func, higher_loss, threshold):
        # Needs to receive att_choice_func(features,objects) -> classifier
        # Classifier(object)->int that will dictate its group
        self.att_choice_func = att_choice_func
        self.root = self.TreeNode()
        self.M = 1
        self.higher_loss = higher_loss
        self.threshold = threshold

    def fit(self, features, objects, m=0):
        self.M = m
        if len(objects) == 0:
            self.root.value = None
            self.root.leaf = True
        positive = objects[0][-1]
        negative = None
        for e in objects:
            if positive != e[-1]:
                negative = e[-1]
                break
        if negative is None:
            self.root.value = positive
            self.root.leaf = True
            return
        self.build_tree_recursively(self.root, features, objects, None, positive, negative)

    def predict(self, x):
        current = self.root
        while not current.leaf:
            current = current.Sons[current.node_classifier(x)]
        return current.value

    def reset(self, att_choice_func, higher_loss, threshold):
        self.att_choice_func = att_choice_func
        self.root = self.TreeNode()
        self.M = 1
        self.higher_loss = higher_loss
        self.threshold = threshold

    # This Function recieves the father node, needs to build the sons recursivly
    def build_tree_recursively(self, current, features, objects, default, positive, negative):

        # First we will check Leaf Cases:
        if len(objects) == 0:
            current.value = default
            current.leaf = True
            return

        # Tree Slice
        if len(objects) <= self.M:
            current.leaf = True
            current.value = default
            return

        current.element_num_in_node = len(objects)
        num_of_pos = 0
        num_of_neg = 0
        for e in objects:
            if e[-1] == positive:
                num_of_pos += 1
            else:
                num_of_neg += 1
        if num_of_neg == 0 or num_of_pos == 0:
            current.leaf = True
            if num_of_pos == 0:
                if self.higher_loss != negative:
                    current.value = negative
                else:
                    current.value = default
            else:
                if self.higher_loss != positive:
                    current.value = positive
                else:
                    current.value = default
            return

        # Now we will recursively build the node:
        if num_of_pos < num_of_neg:
            if self.higher_loss != negative:
                current.value = negative
            elif num_of_neg / num_of_pos >= self.threshold:
                current.value = negative
            else:
                current.value = positive

        else:
            if self.higher_loss != positive:
                current.value = positive
            elif num_of_pos / num_of_neg >= self.threshold:
                current.value = positive
            else:
                current.value = negative
        node_details = [num_of_neg, num_of_pos, negative, positive]
        classifier = self.att_choice_func.choose(self, features=features, objects=objects, node_details=node_details)
        current.node_classifier = classifier
        new_groups = {}
        for e in objects:
            group = classifier(e)
            if group in new_groups:
                new_groups[group].append(e)
            else:
                new_groups[group] = []
                new_groups[group].append(e)
        for group_num, sub_objects in new_groups.items():
            temp_node = self.TreeNode()
            self.build_tree_recursively(temp_node, features, sub_objects, current.value, positive, negative)
            current.Sons[group_num] = temp_node

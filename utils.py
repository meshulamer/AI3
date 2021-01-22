import numpy as np
import random
import math


class DiagnosisClassifier:

    def __init__(self, feature, separator, classes):
        self.feature_index = feature
        self.limit = separator
        self.classes = classes

    def __call__(self, obj):
        if obj[self.feature_index] < self.limit:
            return 0
        else:
            return 1

    def ClassifyRange(self):
        return range(0, self.classes)

    def GetIndex(self):
        return self.feature_index


class RandomAttributeChoiceFunction:

    def __init__(self):
        pass

    def choose(self, features, objects, node_details):
        selected_feature = random.choice(features)
        feature_index = features.index(selected_feature)
        objects.sort(key=lambda e: e[feature_index])
        selected_sep = objects[math.floor(len(objects) / 2)][feature_index]
        classifier = DiagnosisClassifier(feature_index, selected_sep, 2)
        return classifier


class MinEntropyChooser:

    def __init__(self):
        pass

    def choose(self, features, objects, node_details, features_to_ignore=None):
        max_information_gain = 0
        best_feature_index = None
        seperator = None
        for i in range(0, len(features)):
            if features_to_ignore is not None and features[i] in features_to_ignore:
                continue
            feature_information_gain, temp_seperator = information_gain(i, objects, node_details)
            if feature_information_gain >= max_information_gain:
                max_information_gain = feature_information_gain
                best_feature_index = i
                seperator = temp_seperator
        classifier = DiagnosisClassifier(best_feature_index, seperator, 2)
        return classifier


class MinEntropyChooserCS:

    def __init__(self):
        pass

    def choose(self, features, objects, node_details, features_to_ignore=None):
        max_information_gain = 0
        best_feature_index = None
        seperator = None
        for i in range(0, len(features)):
            if features_to_ignore is not None and features[i] in features_to_ignore:
                continue
            feature_information_gain, temp_seperator = information_gain(i, objects, node_details)
            if feature_information_gain >= max_information_gain:
                max_information_gain = feature_information_gain
                best_feature_index = i
                seperator = temp_seperator
        #Found best seperator. Now we be careful by extending the seperator to minimize error margin
        classifier = DiagnosisClassifier(best_feature_index, seperator, 2)
        return classifier


def information_gain(feature_index, objects, node_details, continuous=True):
    if continuous:
        objects.sort(key=lambda e: e[feature_index])
        t_prob = node_details[1] / len(objects)
        h = H(t_prob)
        best_information_gain_k = None
        max_information_gain = 0
        pos_in_left_subgroup = 0
        for i in range(0, len(objects) - 1):
            k_value = (objects[i + 1][feature_index] + objects[i][feature_index]) / 2
            if objects[i][-1] == node_details[3]:
                pos_in_left_subgroup += 1
            total_elements_in_node = node_details[0] + node_details[1]
            true_in_right_subgroup = node_details[1] - pos_in_left_subgroup
            positive_prob_left = pos_in_left_subgroup / (i + 1)
            positive_prob_right = true_in_right_subgroup / (total_elements_in_node - (i + 1))
            htemp = ((i + 1) * H(positive_prob_left) + (total_elements_in_node - (i + 1)) * H(positive_prob_right)) /\
                    len(objects)
            if (h - htemp) > max_information_gain:
                max_information_gain = h - htemp
                best_information_gain_k = k_value
        return max_information_gain, best_information_gain_k


def H(prob):
    if prob == 0 or prob == 1:
        if prob == 0:
            return -(1 - prob) * math.log2(1 - prob)
        return -prob * math.log2(prob)
    return -(prob * math.log2(prob) + (1 - prob) * math.log2(1 - prob))
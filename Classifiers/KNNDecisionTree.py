from .AbstractClassifier import AbstractClassifier
import math
from copy import deepcopy as dc


class KNN(AbstractClassifier):

    def __init__(self, ignored_features_indexes=None):
        self.samples = None
        self.location_average = None
        if ignored_features_indexes is None:
            self.ignored_feature_indexes = []
        else:
            self.ignored_feature_indexes = ignored_features_indexes
        self.k = 1
        self.positive = None
        self.negative = None

    def fit(self, features, objects, k=7):
        self.k = k
        self.samples = dc(objects)
        self.ignored_feature_indexes.append(len(objects[0]) - 1)
        positive = self.samples[0][-1]
        negative = None
        for e in self.samples:
            if positive != e[-1]:
                negative = e[-1]
                break
        self.positive = positive
        self.negative = negative
        self.location_average = self.calculate_location_average()

    def reset(self, ignored_features_indexes=None):
        self.__init__(ignored_features_indexes)


    def predict(self, x):
        self.samples.sort(key=self.DistanceCalculator(x, self.ignored_feature_indexes))
        positive_votes = 0
        negative_votes = 0
        for i in range(0, self.k):
            if self.samples[i][-1] == self.positive:
                positive_votes += 1
            else:
                negative_votes += 1
        if positive_votes > negative_votes:
            return self.positive
        else:
            return self.negative

    def calculate_location_average(self):
        average_vec = []
        for feature in range(0, len(self.samples[0]) - 1):
            sum = 0
            for e in self.samples:
                sum += e[feature]
            feat_average = sum / len(self.samples)
            average_vec.append(feat_average)
        return average_vec

    def coordinates(self):
        return self.location_average



    class DistanceCalculator:
        def __init__(self, x, removed_feats):
            self.test_sample = x
            self.removed_feats = removed_feats

        def __call__(self, x):
            distance_sqr = 0
            for i in range(0, len(self.test_sample) - 1):
                if i in self.removed_feats:
                    continue
                distance_sqr += (x[i] - self.test_sample[i])**2
            return math.sqrt(distance_sqr)

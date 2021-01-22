from .AbstractClassifier import AbstractClassifier
from .KNNDecisionTree import KNN
import random
import math


class KNNForest(AbstractClassifier):

    def __init__(self, N, K, k=7, ignored_features=None):
        self.k = k
        self.K = K
        self.N = N
        self.forest = []
        self.positive = None
        self.negative = None
        self.ignored_features = ignored_features
        if self.ignored_features is None:
            self.ignored_features = []
        self.location_average = None

    def fit(self, features, objects):
        positive = objects[0][-1]
        negative = None
        for e in objects:
            if positive != e[-1]:
                negative = e[-1]
                break
        self.positive = positive
        self.negative = negative
        for i in range(0, self.N):
            subgroup_size = int(len(objects) * random.uniform(0.3, 0.7))
            random.shuffle(objects)
            idx = random.randint(0, len(objects) - subgroup_size)
            subgroup = objects[idx:idx + subgroup_size]
            tree = KNN()
            tree.fit(features, subgroup, self.k)
            self.forest.append(tree)
        self.location_average = self.calculate_location_average()

    def predict(self, x):
        self.forest.sort(key=self.DistanceCalculator(x, self.ignored_features))
        positive_votes = 0
        negative_votes = 0
        for i in range(0, self.K):
            if self.forest[i].predict(x) == self.positive:
                positive_votes += 1
            else:
                negative_votes += 1
        if positive_votes > negative_votes:
            return self.positive
        else:
            return self.negative

    def reset(self, N, K, k=7, ignored_features=None):
        self.__init__(N, K, k, ignored_features)

    def calculate_location_average(self):
        average_vec = None
        for tree in self.forest:
            tree_coords = tree.coordinates()
            if average_vec is None:
                average_vec = tree_coords
                continue
            average_vec = list(map(sum, zip(average_vec,tree_coords)))
        for i in range(0, len(average_vec)):
            average_vec[i] = average_vec[i]/len(average_vec)
        return average_vec



    def coordinates(self):
        return self.location_average

    class DistanceCalculator:
        def __init__(self, x, removed_feats=None):
            self.test_sample = x
            self.removed_feats = removed_feats

        def __call__(self, x):
            distance_sqr = 0
            for i in range(0, len(self.test_sample) - 1):
                if i in self.removed_feats:
                    continue
                distance_sqr += (x.coordinates()[i] - self.test_sample[i]) ** 2
            return math.sqrt(distance_sqr)

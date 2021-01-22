from .AbstractClassifier import AbstractClassifier
from .KNNForest import KNNForest
from copy import deepcopy as dc
from .ID3 import ID3
from .ID3 import kfold_validate_prune
import random
import math


def mark_ignored_features(min_features, features, objects, m=1, param=0.002):
    current_feature_group = dc(features)
    current_removed_features = []
    while len(current_removed_features) < len(features):
        most_useless_feature_contender = None
        worst_accuracy = 0
        current_accuracy = kfold_validate_prune(objects, features, m, current_removed_features)
        for feature in current_feature_group:
            current_removed_features.append(feature)
            accuracy_without_feature = kfold_validate_prune(objects, features, m, current_removed_features)
            if accuracy_without_feature > worst_accuracy:
                worst_accuracy = accuracy_without_feature
                most_useless_feature_contender = feature
            current_removed_features.remove(feature)
        if worst_accuracy > current_accuracy or worst_accuracy > current_accuracy * (1 - param):
            current_removed_features.append(most_useless_feature_contender)
            current_feature_group.remove(most_useless_feature_contender)
        else:
            return current_removed_features


class ImprovedKNNForest(AbstractClassifier):

    def __init__(self, N, K, Nc, Kc, k=7, max_loss=0.005):
        self.k = k
        self.K = K
        self.N = N
        self.forest_committee = []
        self.positive = None
        self.negative = None
        self.ignored_features = None
        self.max_loss = max_loss
        self.N_committee = Nc
        self.K_committee = Kc

    def fit(self, features, objects):
        #self.ignored_features = mark_ignored_features(self.max_loss, features, objects)
        #best ignored features for testing purpses
        self.ignored_features = ['concave points_worst', 'area_worst', 'radius_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'texture_mean', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'smoothness_worst', 'fractal_dimension_worst', 'fractal_dimension_se', 'radius_worst']
        positive = objects[0][-1]
        negative = None
        for e in objects:
            if positive != e[-1]:
                negative = e[-1]
                break
        self.positive = positive
        self.negative = negative
        #OPTION 1
        # for i in range(0, self.N_committee):
        #     subgroup_size = int(len(objects) * random.uniform(0.3, 0.7))
        #     random.shuffle(objects)
        #     idx = random.randint(0, len(objects) - subgroup_size)
        #     subgroup = objects[idx:idx + subgroup_size]
        #     tree = KNNForest(self.N, self.K, self.k, self.ignored_features)
        #     tree.fit(features, subgroup)
        #     self.forest_committee.append(tree)
        # OPTION2
        for i in range(0,self.N_committee):
            tree = KNNForest(self.N, self.K, self.k, self.ignored_features)
            tree.fit(features, objects)
            self.forest_committee.append(tree)


    def predict(self, x):
        #OPTION1
        # self.forest_committee.sort(key=self.DistanceCalculator(x, self.ignored_features))
        # positive_votes = 0
        # negative_votes = 0
        # for i in range(0, self.K_committee):
        #     if self.forest_committee[i].predict(x) == self.positive:
        #         positive_votes += 1
        #     else:
        #         negative_votes += 1
        # if positive_votes > negative_votes:
        #     return self.positive
        # else:
        #     return self.negative
        #OPTION 2
        positive_votes = 0
        negative_votes = 0
        for i in range(0, self.N_committee):
            if self.forest_committee[i].predict(x) == self.positive:
                positive_votes += 1
            else:
                negative_votes += 1
        if positive_votes > negative_votes:
            return self.positive
        else:
            return self.negative

    def reset(self, N, K, Nc, Kc, k=7, max_loss=0.005):
        self.__init__(N, K, Nc, Kc, k, max_loss)

    class DistanceCalculator:
        def __init__(self, x, removed_feats):
            self.test_sample = x
            self.removed_feats = removed_feats
            if removed_feats is None:
                self.removed_feats = []

        def __call__(self, x):
            distance_sqr = 0
            for i in range(0, len(self.test_sample) - 1):
                if i in self.removed_feats:
                    continue
                distance_sqr += (x.coordinates()[i] - self.test_sample[i]) ** 2
            return math.sqrt(distance_sqr)

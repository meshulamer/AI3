from .AbstractClassifier import AbstractClassifier
from .TDIDT import TDIDT
import numpy as np
from utils import MinEntropyChooser
from sklearn.model_selection import KFold
import time


def kfold_validate_prune(objects, features, m=1, ignored_features=None):
    index = 0
    k_fold_testing_objs = np.array(objects)
    kf = KFold(n_splits=5, shuffle=True, random_state=205949886)
    prediction_scores = []
    for train_index, test_index in kf.split(k_fold_testing_objs):
        current_train, current_test = k_fold_testing_objs[train_index], k_fold_testing_objs[test_index]
        current_test = np.ndarray.tolist(current_test)
        current_train = np.ndarray.tolist(current_train)
        for e in current_train:
            for i in range(0, len(e) - 1):
                e[i] = float(e[i])
        for e in current_test:
            for i in range(0, len(e) - 1):
                e[i] = float(e[i])
        iterative_classifier = ID3(ignored_features)
        iterative_classifier.fit(features, current_train, m)
        success_counter = 0
        for e in current_test:
            res = iterative_classifier.predict(e)
            if res == e[-1]:
                success_counter += 1
        prediction_scores.append(success_counter / len(current_test))
        index += 1
    return sum(prediction_scores) / len(prediction_scores)


def kfold_loss_calc(objects, features, m, positive, fp_prob, fn_prob):
    k_fold_testing_objs = np.array(objects)
    kf = KFold(n_splits=5, shuffle=True, random_state=205949886)
    prediction_scores = []
    for train_index, test_index in kf.split(k_fold_testing_objs):
        current_train, current_test = k_fold_testing_objs[train_index], k_fold_testing_objs[test_index]
        current_test = np.ndarray.tolist(current_test)
        current_train = np.ndarray.tolist(current_train)
        for e in current_train:
            for i in range(0, len(e) - 1):
                e[i] = float(e[i])
        for e in current_test:
            for i in range(0, len(e) - 1):
                e[i] = float(e[i])
        iterative_classifier = ID3()
        iterative_classifier.fit(features, current_train, m)
        false_positives = 0
        false_negatives = 0
        for e in current_test:
            res = iterative_classifier.predict(e)
            if res != e[-1]:
                if res == positive:
                    false_positives += 1
                else:
                    false_negatives += 1
        prediction_scores.append((false_positives * fp_prob + false_negatives * fn_prob) / len(current_test))
    return sum(prediction_scores) / len(prediction_scores)


class ID3(AbstractClassifier):

    def __init__(self, features_to_ignore=None):
        self.tdidt_classifier = TDIDT(MinEntropyChooser, features_to_ignore)
        self.features_to_ignore = features_to_ignore
        if features_to_ignore is None:
            self.features_to_ignore = []

    def fit(self, features, objects, m=1):
        self.tdidt_classifier.fit(features, objects, m)

    def predict(self, x):
        return self.tdidt_classifier.predict(x)

    def reset(self, features_to_ignore=None):
        self.tdidt_classifier = TDIDT(MinEntropyChooser, features_to_ignore)
        self.features_to_ignore = features_to_ignore
        if features_to_ignore is None:
            self.features_to_ignore = []
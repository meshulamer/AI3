from .AbstractClassifier import AbstractClassifier
import numpy as np
from utils import MinEntropyChooser
from sklearn.model_selection import KFold
from .CSTDIDT import CSTDIDT
import time


def kfold_loss_calc(objects, features, m, positive, higher_loss, threshold, fp_prob, fn_prob):
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
        iterative_classifier = CSTDIDT(MinEntropyChooser, higher_loss, threshold)
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


class CostSensitiveID3(AbstractClassifier):

    def __init__(self, positive, high_loss, threshold, fp_var, fn_var):
        self.cs_classifier = CSTDIDT(MinEntropyChooserCS, high_loss, threshold)
        self.fp_var = fp_var
        self.fn_var = fn_var
        self.positive = positive
        self.high_loss = high_loss
        self.threshold = threshold

    # Higher Loss is the classification with higher loss score. threshold is how much worse it is to classify the higher loss
    # wrong as oppposed to classifying a lower loss.In the ex example, the threshold is 10.
    def fit(self, features, objects, m=0):
        m_opt = m
        # loss_with_m = []
        # m_paramater = []
        # for i in range(1, 100):
        #     loss_with_m.append(kfold_loss_calc(objects, features, i, self.positive, self.high_loss, self.threshold,
        #                                        self.fp_var, self.fn_var))
        #     m_paramater.append(m)
        # min_val = min(loss_with_m)
        # m_opt = m_paramater[loss_with_m.index(min_val)]
        self.cs_classifier.fit(features, objects, m_opt)

    def predict(self, x):
        return self.cs_classifier.predict(x)

    def reset(self, positive, high_loss, threshold, fp_var, fn_var):
        self.__init__(positive, high_loss, threshold, fp_var, fn_var)

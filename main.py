from CSVread import get_data
from Classifiers.ID3 import ID3
import time
from Classifiers.ID3 import kfold_validate_prune
from Classifiers.ID3 import kfold_loss_calc
from Classifiers.CostSensitiveID3 import CostSensitiveID3
import matplotlib.pyplot as plt
from Classifiers.CostSensitiveID3 import CostSensitiveID3
from Classifiers.KNNDecisionTree import KNN
from Classifiers.KNNForest import KNNForest
from Classifiers.ImprovedKNNForest import ImprovedKNNForest
# Reading info and preparing to run algorithm




"""Fitting ID3 tree and calculating accuracy on the test group"""
# # Fitting to ID3 Tree
# objects, features = get_data("train.csv")
# Classifier = ID3()
# Classifier.fit(features, objects)
#
# # Checking %Success
# objects, features = get_data("test.csv")
# success_counter = 0
# for e in objects:
#     res = Classifier.predict(e)
#     if res == e[-1]:
#         success_counter += 1
# print(success_counter/len(objects))




"""Testing param M to discover best value on test group. Prints Graph"""
# accuracy_with_m = []
# m_paramater = []
# for m in range(0, 31):
#     objects, features = get_data("train.csv")
#     Classifier = ID3()
#     Classifier.fit(features, objects, m)
#     objects, features = get_data("test.csv")
#     success_counter = 0
#     for e in objects:
#         res = Classifier.predict(e)
#         if res == e[-1]:
#             success_counter += 1
#     accuracy_with_m.append(success_counter / len(objects))
#     m_paramater.append(m)
# # plotting the graph
# plt.plot(m_paramater, accuracy_with_m)
# plt.xlabel('M')
# plt.ylabel('Accuracy')
# plt.title('Effect of M parameter on accuracy')
# plt.show()

"""Testing param M to discover best value. Prints Graph"""
# objects, features = get_data("train.csv")
# accuracy_with_m = []
# m_paramater = []
# for m in range(0, 31):
#     accuracy_with_m.append(kfold_validate_prune(objects, features, m))
#     m_paramater.append(m)
# # plotting the graph
# plt.plot(m_paramater, accuracy_with_m)
# plt.xlabel('M')
# plt.ylabel('Average K-fold Accuracy')
# plt.title('Effect of M parameter on K-Fold accuracy')
# plt.show()


"""Testing param M to discover Loss Values. Prints Graph"""
# objects, features = get_data("train.csv")
# loss_with_m = []
# m_paramater = []
# for m in range(1, 31):
#     loss_with_m.append(kfold_loss_calc(objects, features, m, positive='M', fp_prob=0.1, fn_prob=1))
#     m_paramater.append(m)
# # plotting the graph
# plt.plot(m_paramater, loss_with_m)
# plt.xlabel('M')
# plt.ylabel('Average K-Fold Loss')
# plt.title('Effect of M parameter on K-Fold loss')
# plt.show()

"""Testing ID3 Loss Value""" #  0.021238938053097345
#Fitting to ID3 Tree
objects, features = get_data("train.csv")
Classifier = ID3()
Classifier.fit(features, objects)

# Checking %loss
objects, features = get_data("test.csv")
false_positives = 0
false_negatives = 0
for e in objects:
    res = Classifier.predict(e)
    if res != e[-1]:
        if res == 'M':
            false_positives += 1
        else:
            false_negatives += 1
print((false_positives * 0.1 + false_negatives) / len(objects))


"""Testing CSID3 Loss Value"""
# #Fitting to ID3 Tree
# Classifier = CostSensitiveID3('M', 'B', 10, 0.1, 1)
# Classifier.fit(features, objects)
#
# # Checking %loss
# objects, features = get_data("test.csv")
# false_positives = 0
# false_negatives = 0
# for e in objects:
#     res = Classifier.predict(e)
#     if res != e[-1]:
#         if res == 'M':
#             false_positives += 1
#         else:
#             false_negatives += 1
# print((false_positives * 0.1 + false_negatives) / len(objects))
#

'Running KNN tree for testing'
# Classifier = KNN()
# Classifier.fit(features, objects, 7)
# # Checking %Success
# objects, features = get_data("test.csv")
# success_counter = 0
# for e in objects:
#     res = Classifier.predict(e)
#     if res == e[-1]:
#         success_counter += 1
# print(success_counter/len(objects))
#

'Running KNN forest for testing'
# sum = 0
# for i in range(0,100):
#     objects, features = get_data("train.csv")
#     Classifier = KNNForest(11,7)
#     Classifier.fit(features, objects)
#     # Checking %Success
#     objects, features = get_data("test.csv")
#     success_counter = 0
#     for e in objects:
#         res = Classifier.predict(e)
#         if res == e[-1]:
#             success_counter += 1
#     sum += success_counter/len(objects)
# objects, features = get_data("test.csv")
# sum /= len(objects)
# print('KNN over average 100 tests:')
# print(sum)

"Running ImprovedKNN forest for testing"
# def check_params(N, K, Nc, Kc, k, max_loss):
#     sum = 0
#     for i in range(0,100):
#         objects, features = get_data("train.csv")
#         Classifier = ImprovedKNNForest(N, K, Nc, Kc, k)
#         Classifier.fit(features, objects)
#         # Checking %Success
#         objects, features = get_data("test.csv")
#         success_counter = 0
#         for e in objects:
#             res = Classifier.predict(e)
#             if res == e[-1]:
#                 success_counter += 1
#         sum += success_counter/len(objects)
#     objects, features = get_data("test.csv")
#     sum /= len(objects)
#     print(f'Improved KNN, params{N},{K},{Nc},{k} over average 100 tests:')
#     print(sum)











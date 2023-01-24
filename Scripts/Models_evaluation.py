'''
Author = Shreyas Pawar
Description = Script for analysing confusion matrices for different models
'''


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

# All confusion matrices were obtained from the best performing models on the test datsets.
regression_matrix = np.array([[13088, 0, 0, 1748, 0, 5, 0, 2],
  [1598, 91658, 0, 55232, 1, 204, 0, 264],
  [283, 7360, 0, 6486, 0,  30, 0, 132],
  [2506, 65427, 0, 158740, 1, 7981, 0, 5280],
  [17,183, 0, 2430, 0,  38, 0,  39],
  [233, 18, 0, 17804, 1, 12182, 0, 1293],
  [864, 19, 0, 1875, 1, 254, 0, 522],
  [1568, 5085, 0, 24918, 3, 197, 0, 38291]])

ann_matrix = np.array([[13898, 75, 3, 599, 0, 8, 62, 198],
 [301, 106574, 502, 39559, 15, 181, 19, 1806],
 [116, 2902, 3563, 7473, 3, 25, 4, 205],
 [486, 16417, 97, 212410, 101, 7423, 599, 2402],
 [1, 115, 0, 1217, 1225, 72, 1, 76],
 [57, 321, 0, 12421, 19, 18289, 102, 322],
 [14, 3, 1, 831, 0, 155, 2347, 184],
 [350, 1059, 44, 1899, 47, 239, 933, 65491]])

decision_matrix = np.array([[14781, 1, 0, 1, 0, 0, 0, 0],
 [0, 129166, 196, 19487, 2, 0, 0, 10],
 [0, 479, 8341, 5332, 0, 1, 0, 0],
 [0, 5480, 246, 856351,  18, 1180, 4,  96],
 [0, 11, 1, 253, 2375, 2, 0, 0],
 [1, 4, 0, 2757, 0, 28752, 0, 0],
 [0, 1, 0,  29, 0, 2, 3456, 0],
 [2, 19, 2, 208, 1, 0, 2, 69958]])

bayes_matrix = np.array([[2812410, 100036,	0,	0,	0,	0,	1, 0],
[417911, 86423,	0,	0,	0,	0,	13,	0],
[239808,	69,	0,	0,	0,	0,	0,	0],
[49422,	0,	0,	0,	0,	0,	0,	0],
[107637,	3,	0,	0,	0,	0,	0,	0],
[12039,	0,	0,	0,	0,	0,	0,	0],
[35338,	13358,	0,	0,	0,	0,	8,	0],
[9347,	7,	0,	0,	0,	0,	0,	0]])

svm_matrix = np.array([[2858105, 17038, 7016, 0, 30224, 57, 3, 4],
[502128, 1449, 356, 0, 401, 13, 0, 0],
[234479, 3239, 1444, 0, 686, 29, 0, 0],
[41108, 2268, 6046, 0, 0, 0, 0, 0],
[104076, 332, 448, 0, 2779, 3, 0, 2],
[10659, 466, 553, 0, 361, 0, 0, 0],
[48572, 97, 14, 0, 21, 0, 0, 0],
[8991, 142, 75, 0, 146, 0, 0, 0]])

knn_matrix = np.array([[ 1375, 7, 4, 120, 0, 0, 1, 23],
 [26, 8348, 94, 6946, 0, 10, 0, 95],
 [10, 170, 355, 911,  1,  5,  0,  6],
 [185, 2040, 42, 87060, 15, 489, 66, 398],
 [1, 0, 0, 166, 106, 7, 0, 2],
 [7, 21, 1, 2453, 3, 855, 1, 9],
 [0,  1,  0, 195, 0, 4, 154, 35],
 [38,   107,  3,   644,  4, 17, 24,  6340]])

# A function to calculate the ratio of (number of correctly predicted categories samples / Total samples)
def get_total_score(matrix):
  total = sum(sum(matrix))
  correct = sum(np.diagonal(matrix))
  return correct/total

def plot_graphs(x,y):
  plt.plot(x, y, color='red', linestyle='dashed', linewidth = 2,
          marker='o', markerfacecolor='blue', markersize=10)
  plt.xlabel('Models')
  plt.ylabel('Scores')
  plt.title('Model Performances')
  plt.show()


models = [regression_matrix, ann_matrix, decision_matrix, bayes_matrix, svm_matrix, knn_matrix]
model_names = ["Regression","ANN","Decision Tree","Bayesian","SVM","KNN"]
scores = []
for model in models:
  scores.append(get_total_score(model))
plot_graphs(model_names, scores)


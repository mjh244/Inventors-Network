import pandas as pd
import csv
from numpy import array
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from pandas import read_csv
#from seaborn import scatterplot
from matplotlib import pyplot as plt

# Initializes  lists to store features and labels
features = []
labels = []

with open('datasets/inventor-patent-tickers-dates-prices-centrality-to-numbers.csv','r') as file:
  for line in csv.reader(file):
      #print(len(line))
      if (line[0] != "Firstname"):
        features.append(line[0:16] +line[17:20])
        labels.append(line[16])

# Prints out length of entries and some lables
#print(len(features))
#print(features[0])
#print(labels[0:25])

# Turns features and labels to floats to perform ML
features = np.asarray(features)
features = features.astype(np.float64)
labels = np.asarray(labels)
labels = labels.astype(np.float64)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

param_dist = {'min_samples_split': np.arange(6,9)}
model = RandomForestClassifier(n_estimators = 100, bootstrap= False, min_samples_leaf = 7, max_depth = 7, max_features= 'sqrt')
#model = GradientBoostingClassifier(n_estimators=20, learning_rate=1, max_features=2, max_depth=2, random_state=0)
#model = SVC(kernel="linear")
#model = GaussianNB()

"""
cv = GridSearchCV(model, param_grid= param_dist, cv=5, scoring="accuracy")
cv.fit(features_train, labels_train)
print("results from grid search")
print("The best estimator across ALL searched params:\n",cv.best_estimator_)
print("The best score across ALL searched params:\n",cv.best_score_)
print("The best parameters across ALL searched params:\n",cv.best_params_)
"""

para_range = np.arange(4, 11)

model.fit(features_train, labels_train)
prediction = model.predict(features_test)
print(accuracy_score(labels_test, prediction))


"""
train_scores, validation_scores = validation_curve(model, X=features, y=labels, param_name="min_samples_leaf",
                                                   param_range=para_range, scoring="accuracy")
mean_train_score = np.mean(train_scores, axis = 1)
std_train_score = np.std(train_scores, axis = 1)
mean_test_score = np.mean(validation_scores, axis = 1)
std_test_score = np.std(validation_scores, axis = 1)

plt.plot(para_range, mean_train_score, label="Training Score", color='b')
plt.plot(para_range, mean_test_score, label="Cross Validation Score", color='g')

plt.ylabel("Accuracy")
plt.xlabel("min_samples_split")
plt.title("Validation scores for tree")
plt.tight_layout()
plt.legend(loc = 'best')
#plt.savefig('figures/Validation_scores_for_tree.png')
plt.show()
print("finished validation analysis")
"""
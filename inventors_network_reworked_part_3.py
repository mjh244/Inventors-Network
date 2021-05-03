# Authors: Group 7 - McKenzie Hawkins, Alexander Mazon, Haolin Hu

# Imports
import pandas as pd
import csv
from numpy import array
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier

# Much of the code for the ML models in this section was reused from a McKenzie's project in a different class
# The link to that project is below
# https://colab.research.google.com/drive/1bmFoikf-GIq7JGQZRdhSqwoXnPkZMSsn?authuser=1#scrollTo=qkXa_6SYKhEf

# Initializes  lists to store features and labels
features = []
labels = []

with open('datasets/inventor-patent-tickers-dates-prices-centrality-to-numbers.csv','r') as file:
  for line in csv.reader(file):
      #print(len(line))
      if (line[0] != "Firstname"):
        features.append(line[1:16]+line[17:20])
        labels.append(line[16])

# Prints out length of entries and some lables
print(len(features))
print(labels[0:25])

# Turns features and labels to floats to perform ML
features = np.asarray(features)
features = features.astype(np.float64)
labels = np.asarray(labels)
labels = labels.astype(np.float64)

# Initializes accuracy values
gnb_acc = 0
svc_acc = 0
dt_acc = 0
clf_acc = 0

# Trains/tests the models 100 times and sums results to average at the end
for i in range(100):

  # Splits data into 80% training and 20% testing
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

  # Decision Tree
  classifierTree = tree.DecisionTreeClassifier(max_depth=9)
  classifierTree = classifierTree.fit(features_train, labels_train)
  dt_predictions = classifierTree.predict(features_test)
  dt_acc += accuracy_score(labels_test, dt_predictions)

"""
  # Gaussian Naive Bayes
  classifierNB = GaussianNB()
  classifierNB = classifierNB.fit(features_train, labels_train)
  gnb_predictions = classifierNB.predict(features_test)
  gnb_acc += accuracy_score(labels_test, gnb_predictions)
  
  # Neural Net
  clf = MLPClassifier(solver='sgd')
  clf = clf.fit(features_train, labels_train)
  clf_predictions = clf.predict(features_test)
  clf_acc += accuracy_score(labels_test, clf_predictions)

# SMV SVC

  classifierSVC = svm.SVC(kernel = 'linear', class_weight='balanced')
  classifierSVC = classifierSVC.fit(features_train, labels_train)
  svc_predictions = classifierSVC.predict(features_test)
  svc_acc += accuracy_score(labels_test, svc_predictions)
"""


gnb_acc = gnb_acc / 100
#svc_acc = svc_acc / 100
dt_acc = dt_acc / 100
#clf_acc = clf_acc / 100
print("GNB Accuracy:", gnb_acc)
#print("SVC Accuracy:", svc_acc)
print("DT Accuracy:", dt_acc)
print("NN Accuracy:", clf_acc)
print()

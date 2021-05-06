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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve

from pandas import read_csv
from seaborn import scatterplot
from matplotlib import pyplot as plt


# Much of the code for the ML models in this section was reused from a McKenzie's project in a different class
# The link to that project is below
# https://colab.research.google.com/drive/1bmFoikf-GIq7JGQZRdhSqwoXnPkZMSsn?authuser=1#scrollTo=qkXa_6SYKhEf
#Some data visualization
'''
url = 'datasets/inventor-patent-tickers-dates-prices-centrality-to-numbers.csv'
dataset = read_csv(url, header=0)
prev_month_change=[]
for index, row in dataset.iterrows():
  diff = row["Price a Month Before"] - row["Price the Day of"]
  prev_month_change.append(diff)

dataset["Previous Month Change"]= prev_month_change
# create scatter plot
scatterplot(x="Previous Month Change", y="Change", data=dataset)
# show plot
plt.show()
'''
# Initializes  lists to store features and labels
features = []
labels = []

with open('datasets/inventor-patent-tickers-dates-prices-centrality-to-numbers.csv','r') as file:
  for line in csv.reader(file):
      #print(len(line))
      if (line[0] != "Firstname"):
        features.append(line[0:16]+line[17:20])
        labels.append(line[16])

# Prints out length of entries and some lables
print(len(features))
print(features[0])
print(labels[0:25])

# Turns features and labels to floats to perform ML
features = np.asarray(features)
features = features.astype(np.float64)
labels = np.asarray(labels)
labels = labels.astype(np.float64)
#analysis on tree depth
print("params")
classifierTree = tree.DecisionTreeClassifier()

print(classifierTree.get_params())
para_range = np.arange(1,20)
train_scores, validation_scores = validation_curve(classifierTree, X=features, y=labels, param_name="max_depth",
                                                   param_range=para_range, scoring="accuracy")
mean_train_score = np.mean(train_scores, axis = 1)
std_train_score = np.std(train_scores, axis = 1)
mean_test_score = np.mean(validation_scores, axis = 1)
std_test_score = np.std(validation_scores, axis = 1)

plt.plot(para_range, mean_train_score, label="Training Score", color='b')
plt.plot(para_range, mean_test_score, label="Cross Validation Score", color='g')



plt.ylabel("Accuracy")
plt.xlabel("Depth of Tree")
plt.title("Validation scores for tree")
plt.tight_layout()
plt.legend(loc = 'best')
plt.savefig('figures/Validation_scores_for_tree.png')
plt.show()
print("finished validation analysis")
# Initializes accuracy values
gnb_acc = 0
svc_acc = 0
dt_acc_arr = []
dt_prec_arr = []
dt_rec_arr = []
clf_acc = 0


# Trains/tests the models 5 times and sums results to average at the end

for j in range(1,20):
  dt_acc = 0
  dt_prec = 0
  dt_rec = 0
  for i in range(5):

    # Splits data into 80% training and 20% testing
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)


    # Decision Tree
    classifierTree = tree.DecisionTreeClassifier(max_depth=j)
    classifierTree = classifierTree.fit(features_train, labels_train)
    dt_predictions = classifierTree.predict(features_test)
    dt_acc += accuracy_score(labels_test, dt_predictions)
    dt_prec += precision_score(labels_test, dt_predictions)
    #percent of positives we got right
    dt_rec += recall_score(labels_test, dt_predictions)
  dt_acc = dt_acc / 5
  dt_prec = dt_prec / 5
  dt_rec = dt_rec / 5
  dt_acc_arr.append(dt_acc)
  dt_rec_arr.append(dt_rec)
  dt_prec_arr.append(dt_prec)
depths=np.arange(1, 20)
plt.plot(depths, dt_acc_arr)
plt.plot(depths, dt_rec_arr)
plt.plot(depths, dt_prec_arr)
plt.xlabel('Tree depths')
plt.legend(["Accuracy scores", "Recall scores", "Precision scores"])
plt.title("Classification metrics as a function of tree depth")
plt.savefig('figures/metrics_vs_tree_depth.png')
plt.show()



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
svc_acc = svc_acc / 100

clf_acc = clf_acc / 100
print("GNB Accuracy:", gnb_acc)
#print("SVC Accuracy:", svc_acc)
print("DT Accuracy:", dt_acc)
print("DT Precision:", dt_prec)
print("DT Recall:", dt_rec)
print("NN Accuracy:", clf_acc)
print()
# load the dataset


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import csv
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from numpy import array
from numpy import argmax
#from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('inventor-patent-stock-centrality.csv')
next_month = df["Closing Price Next Month"]
current_month = df["Closing Price"]
last_month = df["Closing Price Last Month"]

change = [next_month-current_month > 0]
change = np.array(change)
change = change.astype(int)
print(change[0])

df["Change"] = change[0]

str_cols = df.keys()
str_cols = str_cols[1:]
print(str_cols)

# Map nominal features to numbers
for col in str_cols:
  if ((col == 'Firstname') or (col == 'Lastname') or (col == 'City') or (col == 'State') or (col == 'Country')
  or (col == 'Zipcode') or (col == 'Patent') or (col == 'AppYear') or (col == 'GYear') or (col == 'AppDate')
  or (col == 'Assignee') or (col == 'Tickers')):
    df[col] = pd.Categorical(df[col], ordered=True).codes
    pd.to_numeric(df[col], downcast='float')
#df = df.apply(pd.to_numeric)
#print(df)


df = df.drop(['Closing Price Next Month'], axis=1)
pd.set_option('max_columns', None)
print(df)

df.to_csv('inventor-patent-stock-centrality-to-numbers', index=False)




# Much of the code for the ML models in this section was reused from a McKenzie's project in a different class
# The link to that project is below
# https://colab.research.google.com/drive/1bmFoikf-GIq7JGQZRdhSqwoXnPkZMSsn?authuser=1#scrollTo=qkXa_6SYKhEf

# Method to compute percent of true positives, false positives, etc.
def measure(actual, predictions):
  tpr = 0
  tnr = 0
  tpSum = 0
  fpSum = 0
  tnSum = 0
  fnSum = 0
  for i in range(len(actual)):
    if ((actual[i] == 1) and (predictions[i] == 1)):
      tpSum += 1
    elif ((actual[i] == 0) and (predictions[i] == 1)):
      fpSum += 1
    elif ((actual[i] == 0) and (predictions[i] == 0)):
      tnSum += 1
    elif ((actual[i] == 1) and (predictions[i] == 0)):
      fnSum += 1
  tpr = tpSum / (tpSum + fnSum)
  tnr = tnSum / (tnSum + fpSum)
  return tpr, tnr

features = []
labels = []
data = []

with open('inventor-patent-stock-centrality-to-numbers','r') as file:
  for line in csv.reader(file):
      if (line[0] != "Unnamed: 0"):
        features.append(line[1:19])
        labels.append(line[19])

print(labels)

print(labels)
features = np.array(features)
features = features.astype(np.float)
labels = np.array(labels)
labels = labels.astype(np.float)

gnb_acc = 0
svc_acc = 0
dt_acc = 0
linearSVC_acc = 0
clf_acc = 0

gnb_ttr, gnb_tnr = 0, 0
svc_ttr, svc_tnr = 0, 0
dt_ttr, dt_tnr = 0, 0

# Trains/tests the models 100 times and sums results to average at the end
for i in range(100):

  # Splits data into 80% training and 20% testing
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

  classifierNB = GaussianNB()
  classifierNB = classifierNB.fit(features_train, labels_train)
  gnb_predictions = classifierNB.predict(features_test)
  gnb_acc += accuracy_score(labels_test, gnb_predictions)
  gnb_ttr += measure(labels_test, gnb_predictions)[0]
  gnb_tnr += measure(labels_test, gnb_predictions)[1]

  classifierSVC = svm.SVC(kernel = 'linear', class_weight='balanced')
  classifierSVC = classifierSVC.fit(features_train, labels_train)
  svc_predictions = classifierSVC.predict(features_test)
  svc_acc += accuracy_score(labels_test, svc_predictions)
  svc_ttr += measure(labels_test, svc_predictions)[0]
  svc_tnr += measure(labels_test, svc_predictions)[1]

  classifierTree = tree.DecisionTreeClassifier(max_depth=9)
  classifierTree = classifierTree.fit(features_train, labels_train)
  dt_predictions = classifierTree.predict(features_test)
  dt_acc += accuracy_score(labels_test, dt_predictions)
  dt_ttr += measure(labels_test, dt_predictions)[0]
  dt_tnr += measure(labels_test, dt_predictions)[1]

  # Neural Net
  clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
  clf = clf.fit(features_train, labels_train)
  clf_predictions = clf.predict(features_test)
  clf_acc += accuracy_score(labels_test, clf_predictions)

gnb_acc = gnb_acc / 100
svc_acc = svc_acc / 100
dt_acc = dt_acc / 100
clf_acc = dt_acc / 100
print("Accuracy:", gnb_acc)
print("Accuracy:", svc_acc)
print("Accuracy:", dt_acc)
print("Accuracy:", clf_acc)
print()

gnb_ttr = gnb_ttr / 100
gnb_tnr = gnb_tnr / 100
svc_ttr = svc_ttr / 100
svc_tnr = svc_tnr / 100
dt_ttr = dt_ttr / 100
dt_tnr = dt_tnr / 100
print("GNB TTR:", gnb_ttr)
print("GNB TNR:", gnb_tnr)
print("SVC TTR:", svc_ttr)
print("SVC TNR:", svc_tnr)
print("DT TTR:", dt_ttr)
print("DT TNR:", dt_tnr)

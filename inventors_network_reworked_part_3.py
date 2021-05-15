# Authors: Group 7 - McKenzie Hawkins, Alexander Mazon, Haolin Hu

# Imports
import pandas as pd
import csv
from numpy import array
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from pandas import read_csv
#from seaborn import scatterplot
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
'''
with open('datasets/inventor-patent-tickers-dates-prices-centrality-to-numbers.csv','r') as file:
  for line in csv.reader(file):
      #print(len(line))
      if (line[0] != "Firstname"):
        features.append(line[0:16]+line[17:20])
        labels.append(line[16])
        '''
dataset = pd.read_csv('datasets/inventor-patent-tickers-dates-prices-centrality.csv')
#X = dataset.iloc[:, [1:10], [11:18], [20:23]].values


# Turns the dataframe to numbers
for col in dataset.keys():
  if ((col == 'Firstname') or (col == 'Lastname') or (col == 'City') or (col == 'State') or (col == 'Country')
  or (col == 'Zipcode') or (col == 'Patent') or (col == 'AppYear') or (col == 'GYear')  or (col == 'Assignee')
  or (col == 'Tickers') or (col == 'Month Before Application Date') or (col == 'Application Date')
  or (col == 'Month After Application Date')):
    dataset[col] = pd.Categorical(dataset[col], ordered=True).codes
    pd.to_numeric(dataset[col], downcast='float')

dropped_change = dataset.drop(columns="Change")
dropped_AppDate = dropped_change.drop(columns="AppDate")
dropped_price_month_after = dropped_AppDate.drop(columns="Price a Month After")


new_df = dropped_price_month_after.to_numpy()
sc = StandardScaler()
# Normalizes the dataframe

standard_scaler_df = sc.fit_transform(new_df)
new_df = pd.DataFrame(standard_scaler_df)
new_df["Change"]=dataset["Change"]

# Saves the dataframe with values converted to numbers to be used in ML
#new_df.to_csv('datasets/inventor-patent-tickers-dates-prices-centrality-to-numbers.csv', index=False)
print("Dataset with values converted to numbers")
print(new_df)
features = new_df.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values
labels = new_df.iloc[:, 20].values
features = np.asarray(features)
features = features.astype(np.float64)
labels = np.asarray(labels)
labels = labels.astype(np.float64)
# Prints out length of entries and some lables
'''
print(len(features))
print(features[0])
print(labels[0:25])

# Turns features and labels to floats to perform ML
features = np.asarray(features)
features = features.astype(np.float64)
labels = np.asarray(labels)
labels = labels.astype(np.float64)
'''
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
#analysis on tree depth
print("params")
classifierTree = tree.DecisionTreeClassifier()
params = classifierTree.get_params()
print(params)
#skeleton for parameter tuning
'''
parameters = {'criterion': ["gini","random"],
              'splitter': ["best", "random"],
              'max_depth': np.arange(1,20)}

grid_tree = GridSearchCV(estimator=classifierTree, param_grid=parameters, cv=5, scoring="accuracy")
grid_tree.fit(features_train, labels_train)
print("results from grid search")
print("The best estimator across ALL searched params:\n",grid_tree.best_estimator_)
print("The best score across ALL searched params:\n",grid_tree.best_score_)
print("The best parameters across ALL searched params:\n",grid_tree.best_params_)
'''
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
# K Nearest Neighbours model
'''
classifierKNN = KNeighborsClassifier()
params = classifierKNN.get_params()
print("params")
print(params)
kNN_acc= 0
para_range = np.arange(1,7)
train_scores, validation_scores = validation_curve(classifierKNN, X=features, y=labels, param_name="n_neighbors",
                                                   param_range=para_range, scoring="accuracy")
mean_train_score = np.mean(train_scores, axis = 1)
std_train_score = np.std(train_scores, axis = 1)
mean_test_score = np.mean(validation_scores, axis = 1)
std_test_score = np.std(validation_scores, axis = 1)

plt.plot(para_range, mean_train_score, label="Training Score", color='b')
plt.plot(para_range, mean_test_score, label="Cross Validation Score", color='g')



plt.ylabel("Accuracy")
plt.xlabel("number of neighbors")
plt.title("Validation scores for KNN")
plt.tight_layout()
plt.legend(loc = 'best')
plt.savefig('figures/Validation_scores_for_KNN.png')
plt.show()

print("finished validation analysis for KNN")
classifierKNN = classifierKNN.fit(features_train, labels_train)
kNN_predictions = classifierKNN.predict(features_test)
kNN_acc += accuracy_score(labels_test, kNN_predictions)
print("KNN accuracy is" + str(kNN_acc))
'''
# Initializes accuracy values
gnb_train_acc = 0
gnb_valid_acc = 0
svc_acc = 0
dt_acc_arr = []
dt_prec_arr = []
dt_rec_arr = []
clf_acc = 0


# Trains/tests the models 5 times and sums results to average at the end
'''
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
'''



  # Gaussian Naive Bayes
classifierNB = GaussianNB()
params = classifierNB.get_params()

print("nb params are" + str(params))
classifierNB = classifierNB.fit(features_train, labels_train)
gnb_predictions_valid = classifierNB.predict(features_test)
gnb_predictions_train = classifierNB.predict(features_train)
gnb_valid_acc += accuracy_score(labels_test, gnb_predictions_valid)
gnb_train_acc += accuracy_score(labels_train, gnb_predictions_train)
print("gnb has validation accuracy:" + str(gnb_valid_acc))
print("gnb has training accuracy:" + str(gnb_train_acc))

'''
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
'''


#gnb_acc = gnb_acc / 100
#svc_acc = svc_acc / 100

#clf_acc = clf_acc / 100
#print("GNB Accuracy:", gnb_acc)
#print("SVC Accuracy:", svc_acc)
#print("DT Accuracy:", dt_acc)
#print("DT Precision:", dt_prec)
#print("DT Recall:", dt_rec)
#print("NN Accuracy:", clf_acc)
print()
# load the dataset


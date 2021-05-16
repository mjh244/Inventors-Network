# Authors: Group 7 - McKenzie Hawkins, Alexander Mazon, Haolin Hu

# Imports
import pandas as pd
import csv
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from numpy import array
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from pandas import read_csv
#from seaborn import scatterplot
from matplotlib import pyplot as plt





new_df = pd.read_csv('datasets/inventor-patent-tickers-dates-prices-centrality-to-numbers.csv')


# Saves the dataframe with values converted to numbers to be used in ML
#new_df.to_csv('datasets/inventor-patent-tickers-dates-prices-centrality-to-numbers.csv', index=False)
print("Dataset with values converted to numbers")
print(new_df)

features = new_df.iloc[:, 0:19].values
labels = new_df.iloc[:, 19].values
centrality_features = new_df.iloc[:, 16:19].values


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
centrality_features_train, centrality_features_test, centrality_labels_train, centrality_labels_test = train_test_split(
    centrality_features, labels, test_size=0.2)
#analysis on tree depth

print("params")
classifierTree = tree.DecisionTreeClassifier()

optimized_classifierTree = tree.DecisionTreeClassifier(splitter='best', max_features='auto', max_depth=8, criterion='gini')
centrality_history = optimized_classifierTree.fit(centrality_features_train, centrality_labels_train)

y_pred_centrality = centrality_history.predict(centrality_features_test)
centrality_acc = accuracy_score(centrality_labels_test, y_pred_centrality)
print("centrality accuracy is " + str(centrality_acc))
centrality_rec = recall_score(centrality_labels_test, y_pred_centrality)
print("centrality recall is " + str(centrality_rec))
# centrality accuracy is 0.5415861314634183
#centrality recall is 0.9969713732540451
cm_centrality = confusion_matrix(centrality_labels_test, y_pred_centrality)
sn.set(font_scale=1.4) # for label size
sn.heatmap(cm_centrality, annot=True, annot_kws={"size": 16}) # font size
plt.show()
plt.savefig('figures/confusion_matrix_for_optimized_DTREE_using_only_centrality_measures')

classifier_history = optimized_classifierTree.fit(features_train, labels_train)
y_pred = classifier_history.predict(features_test)
dTree_acc = accuracy_score(labels_test, y_pred)
dTree_rec = recall_score(labels_test, y_pred)
print("accuracy of dtree is " + str(dTree_acc))
print("recall of dtree is " + str(dTree_rec))
#accuracy of dtree is 0.6429930627350504
#recall of dtree is 0.8167433353655157
cm = confusion_matrix(labels_test, y_pred)
sn.set(font_scale=1.4) # for label size
sn.heatmap(cm, annot=True, annot_kws={"size": 16}) # font size
plt.show()
plt.savefig('figures/confusion_matrix_for_optimized_DTREE')








params = classifierTree.get_params()
print(params)



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
print("finished validation analysis for tree")

# it seems that the optimal max depth is 7.5

criterion = ["gini", "entropy"]
splitter = ["best", "random"]
max_depth = [7,8]
max_features = ["auto", "sqrt", "log2"]

param_grid = dict(criterion=criterion, splitter=splitter, max_features=max_features, max_depth=max_depth)
grid = RandomizedSearchCV(estimator=classifierTree,
                    param_distributions=param_grid,
                    cv=3)
grid_result = grid.fit(features_train, labels_train)
print(f'Best Accuracy for decision tree {grid_result.best_score_:.4} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'mean={mean:.4}, std={stdev:.4} using {param}')
# Initializes accuracy values



dt_acc_arr = []
dt_valid_acc_arr = []
dt_prec_arr = []
dt_rec_arr = []




# Trains/tests the models 5 times and sums results to average at the end

for j in range(1,20):
  dt_acc = 0
  dt_valid_acc = 0
  dt_prec = 0
  dt_rec = 0
  for i in range(5):

    # Splits data into 80% training and 20% testing
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)


    # Decision Tree
    classifierTree = tree.DecisionTreeClassifier(max_depth=j)
    classifierTree = classifierTree.fit(features_train, labels_train)
    dt_predictions = classifierTree.predict(features_test)
    dt_valid_accs  = cross_val_score(classifierTree, features, labels, cv=5)
    dt_valid_acc += np.mean(dt_valid_accs)

    dt_acc += accuracy_score(labels_test, dt_predictions)
    dt_prec += precision_score(labels_test, dt_predictions)
    #percent of positives we got right
    dt_rec += recall_score(labels_test, dt_predictions)
  dt_acc = dt_acc / 5
  dt_valid_acc = dt_valid_acc / 5
  dt_prec = dt_prec / 5
  dt_rec = dt_rec / 5
  dt_acc_arr.append(dt_acc)
  dt_rec_arr.append(dt_rec)
  dt_prec_arr.append(dt_prec)
  dt_valid_acc_arr.append(dt_valid_acc)
depths=np.arange(1, 20)
plt.plot(depths, dt_acc_arr)
plt.plot(depths, dt_rec_arr)
plt.plot(depths, dt_prec_arr)
plt.plot(depths, dt_valid_acc_arr)
plt.xlabel('Tree depths')
plt.legend(["Accuracy scores", "Recall scores", "Precision scores", "Cross Validation Accuracy"])
plt.title("Classification metrics as a function of tree depth")
plt.savefig('figures/metrics_vs_tree_depth.png')
plt.show()








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

new_df = pd.read_csv('datasets/inventor-patent-tickers-dates-prices-centrality-to-numbers.csv')

features = new_df.iloc[:, 0:19].values
labels = new_df.iloc[:, 19].values
centrality_features = new_df.iloc[:, 16:19].values

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
centrality_features_train, centrality_features_test, centrality_labels_train, centrality_labels_test = train_test_split(
    centrality_features, labels, test_size=0.2)

param_dist = {'min_samples_split': np.arange(2,11),
              'min_samples_leaf': np.arange(2, 11)}
model = RandomForestClassifier(n_estimators = 20, max_depth = 7, max_features = 'sqrt', bootstrap = False, n_jobs= -1, random_state= 0)
#model = GradientBoostingClassifier(n_estimators=20, learning_rate=1, max_features=2, max_depth=2, random_state=0)
#model = SVC(kernel="linear")
#model = GaussianNB()

cv = GridSearchCV(model, param_grid = param_dist, scoring="accuracy", n_jobs= -1)
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
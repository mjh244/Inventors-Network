import pandas as pd
import csv
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
features = new_df.iloc[:, 0:19].values
labels = new_df.iloc[:, 19].values
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
# K Nearest Neighbours model

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
KNN_valid_acc = np.mean(cross_val_score(classifierKNN, features, labels, cv=5), axis=1)

print("KNN accuracy is" + str(kNN_acc))

 # Gaussian Naive Bayes analysis

classifierNB = GaussianNB()
params = classifierNB.get_params()
gnb_valid_accs  = cross_val_score(classifierNB, features, labels, cv=5)
print("gnb valid accs are" + str(gnb_valid_accs))
print("nb params are" + str(params))
classifierNB = classifierNB.fit(features_train, labels_train)
gnb_predictions = classifierNB.predict(features_test)


#gnb_valid_acc = np.mean(gnb_valid_accs, axis=1)
gnb_valid_acc = np.mean(gnb_valid_accs)
gnb_acc = accuracy_score(labels_test, gnb_predictions)
print("gnb has validation accuracy:" + str(gnb_valid_acc))
print("gnb has regular testing accuracy:" + str(gnb_acc))
train_sizes, train_scores, validation_scores = learning_curve(classifierNB, X=features,y=labels, scoring="accuracy")
mean_train_score = np.mean(train_scores, axis = 1)
std_train_score = np.std(train_scores, axis = 1)
mean_test_score = np.mean(validation_scores, axis = 1)
std_test_score = np.std(validation_scores, axis = 1)

plt.plot(train_sizes, mean_train_score, label="Training Score", color='b')
plt.plot(train_sizes, mean_test_score, label="Cross Validation Score", color='g')



plt.ylabel("Accuracy")
plt.xlabel("training size")
plt.title("Learning Curve for Naive Bayes")
plt.tight_layout()
plt.legend(loc = 'best')
plt.savefig('figures/Learning_curve_for_GNB.png')
plt.show()


#SVM analysis
classifierSV = SVC(random_state=0)
param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
    classifierSV, features, labels, param_name="gamma", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_scores_mean, label="Training Score", color='b')
plt.plot(param_range, test_scores_mean, label="Cross Validation Score", color='g')



plt.ylabel("Accuracy")
plt.xlabel("$\gamma$")
plt.title("Validation scores for SVM")
plt.tight_layout()
plt.legend(loc = 'best')
plt.savefig('figures/Validation_scores_for_SVM.png')
plt.show()
print("finished validation analysis for SVM")

# random forest analysis
classifierRF = RandomForestClassifier(n_estimators = 100, max_depth = 3, max_features = 4, bootstrap = False, n_jobs= -1, random_state= 0)
classifierRF.fit(features_train, labels_train)
rf_predictions = classifierRF.predict(features_test)
print(accuracy_score(labels_test, rf_predictions))
"""
n_estimators = [10, 50, 100]
criterion = ["Gini", "entropy"]
max_depth = np.arange(1,10)
min_samples_leaf = np.arange(1,10)
max_features = ["auto", "sqrt", "log2"]
bootstrap = [True, False]


param_grid = dict(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf
                  ,max_features=max_features, bootstrap=bootstrap)


grid = RandomizedSearchCV(estimator=classifierRF,
                    param_distributions=param_grid,
                    cv=3)
grid_result = grid.fit(features_train, labels_train)


# print results
print(f'Best Accuracy for random forest {grid_result.best_score_:.4} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'mean={mean:.4}, std={stdev:.4} using {param}')
"""

# gradient boosting analysis
classifierGB = GradientBoostingClassifier(random_state=0)
n_estimators = [20, 50, 100]
learning_rate = [.1,.25,.5,.71,1]
max_features = ["auto", "sqrt", "log2"]
max_depth = np.arange(1,6)
param_grid = dict(n_estimators=n_estimators, learning_rate=learning_rate, max_features=max_features, max_depth=max_depth)
grid = RandomizedSearchCV(estimator=classifierGB,
                    param_distributions=param_grid,
                    cv=3)
grid_result = grid.fit(features_train, labels_train)
print(f'Best Accuracy for gradient boosting {grid_result.best_score_:.4} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'mean={mean:.4}, std={stdev:.4} using {param}')


# for ANN best params are batch_size = 128, epochs = 60, kernal_initializer='glorot_uniform', internal nodes=6


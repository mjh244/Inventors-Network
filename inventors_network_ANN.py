import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# Importing the dataset
dataset = pd.read_csv('datasets/inventor-patent-tickers-dates-prices-centrality.csv')
#X = dataset.iloc[:, [1:10], [11:18], [20:23]].values



for col in dataset.keys():
  if ((col == 'Firstname') or (col == 'Lastname') or (col == 'City') or (col == 'State') or (col == 'Country')
  or (col == 'Zipcode') or (col == 'Patent') or (col == 'AppYear') or (col == 'GYear')  or (col == 'Assignee')
  or (col == 'Tickers') or (col == 'Month Before Application Date') or (col == 'Application Date')
  or (col == 'Month After Application Date')):
    dataset[col] = pd.Categorical(dataset[col], ordered=True).codes
    pd.to_numeric(dataset[col], downcast='float')
'''
new_df = pd.DataFrame(dataset["Price a Month Before"], columns = ["Price a Month Before"])
new_df["Price the Day of"] = dataset["Price the Day of"]
new_df["Change"] = dataset["Change"]
new_df["Degree"] = dataset["Degree"]
new_df["Betweenness"] = dataset["Betweenness"]
new_df["Eigenvector"] = dataset["Eigenvector"]

print(new_df)
'''
dropped_change = dataset.drop(columns="Change")
dropped_AppDate = dropped_change.drop(columns="AppDate")
dropped_price_month_after = dropped_AppDate.drop(columns="Price a Month After")
# Turns the dataframe to numbers
#df = df.apply(pd.to_numeric, downcast='float')
#df = pd.to_numeric(df, downcast='float')
new_df = dropped_price_month_after.to_numpy()
sc = StandardScaler()
# Normalizes the dataframe
#min_max_df = MinMaxScaler().fit_transform(new_df)
standard_scaler_df = sc.fit_transform(new_df)
new_df= pd.DataFrame(standard_scaler_df)
new_df["Change"]=dataset["Change"]
#new_df = pd.DataFrame(min_max_df)

#print(new_df)

#print(max(list(new_df[0])))
#print(max(list(new_df[1])))
#print(max(list(new_df[2])))
#print(max(list(new_df[3])))
# Saves the dataframe with values converted to numbers to be used in ML
#new_df.to_csv('datasets/inventor-patent-tickers-dates-prices-centrality-to-numbers.csv', index=False)
print("Dataset with values converted to numbers")
print(new_df)

#X = new_df.iloc[:, ["Firstname","Lastname","City","State","Country","Zipcode","Patent","AppYear","GYear","Assignee",
                    #"Tickers","Month Before Application Date","Application Date","Month After Application Date",
                    #"Price a Month Before","Price the Day of","Degree","Betweenness","Eigenvector"]].values
X = new_df.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values
#X = dataset.iloc[:, [13,14,15,16,17,20,21,22]].values
#X = new_df.iloc[:, [0,1,3,4,5]].values
#y = new_df.iloc[:, 2].values
#y = new_df.iloc[:, "Change"].values
y= new_df.iloc[:, 20].values
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

'''
labelencoder_X_0 = LabelEncoder()
X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0])
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
labelencoder_X_5 = LabelEncoder()
X[:, 5] = labelencoder_X_5.fit_transform(X[:, 5])
labelencoder_X_6 = LabelEncoder()
X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])
labelencoder_X_7 = LabelEncoder()
X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7])
labelencoder_X_8 = LabelEncoder()
X[:, 8] = labelencoder_X_8.fit_transform(X[:, 8])
labelencoder_X_9 = LabelEncoder()
X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9])
labelencoder_X_10 = LabelEncoder()
X[:, 10] = labelencoder_X_10.fit_transform(X[:, 10])
labelencoder_X_11 = LabelEncoder()
X[:, 11] = labelencoder_X_11.fit_transform(X[:, 11])
labelencoder_X_12 = LabelEncoder()
X[:, 12] = labelencoder_X_12.fit_transform(X[:, 12])
labelencoder_X_13 = LabelEncoder()
X[:, 13] = labelencoder_X_13.fit_transform(X[:, 13])
'''
#onehotencoder = OneHotEncoder(categorical_features = [1])
#onehotencoder = OneHotEncoder(categories='auto')
#X = onehotencoder.fit_transform(X).toarray()
#removing the dummy variable
#X = X[:, 1:]

print("start ANN")
print("X is" + str(X))
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
print("about to train, test, split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
print("about to normalize")
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
print("normalized")

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential #used to initialize the NN

from keras.layers import Dense  #used to build the hidden Layers
from keras.layers import Dropout

# Initialising the ANN
def create_classifier(init_mode='uniform', num_nodes=3):
    classifier = keras.Sequential()
    print("classifier initialized")
    # Adding the input layer and the first hidden layer with dropout
    classifier.add(Dense(units = num_nodes, kernel_initializer = init_mode, activation = 'relu', input_dim =19))
    print("input layer done")
    classifier.add(Dropout(0.1))

    # Adding the second hidden layer
    classifier.add(Dense(units = num_nodes, kernel_initializer = init_mode, activation = 'relu'))
    print("hidden layer done")
    classifier.add(Dropout(0.1))

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = init_mode, activation = 'sigmoid'))
    print("about to compile")
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
    return classifier
def create_classifier2(optimizer='adam', init='glorot_uniform'):
    classifier = keras.Sequential()
    print("classifier initialized")
    # Adding the input layer and the first hidden layer with dropout
    classifier.add(Dense(units=12, kernel_initializer=init, activation='relu', input_dim=19))
    print("input layer done")
    classifier.add(Dropout(0.1))

    # Adding the second hidden layer
    classifier.add(Dense(units=12, kernel_initializer=init, activation='relu'))
    print("hidden layer done")
    classifier.add(Dropout(0.1))

    # Adding the output layer
    classifier.add(Dense(units=1, kernel_initializer=init, activation='sigmoid'))
    print("about to compile")
    # Compiling the ANN
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
    return classifier
print("fitting")
seed = 7
np.random.seed(seed)
#batch_size = 128
#epochs = 10
# hyperparameter tuning comes from https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594
#model_CV = KerasClassifier(build_fn=create_classifier, epochs=epochs,
                           #batch_size=batch_size, verbose=1)
model_init_epochs_batch_CV = KerasClassifier(build_fn=create_classifier2, verbose=1)
init_mode = ['glorot_uniform', 'glorot_normal']
batches = [68, 128, 256, 512]
epochs = [10, 20, 30, 40, 50, 60]
# define the grid search parameters
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero',
             #'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

param_grid = dict(epochs=epochs, batch_size=batches, init=init_mode)
#grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, n_jobs=-1, cv=3)
#grid = GridSearchCV(estimator=model_init_epochs_batch_CV,
                    #param_grid=param_grid,
                    #cv=3)
#less accurate but faster
grid = RandomizedSearchCV(estimator=model_init_epochs_batch_CV,
                    param_distributions=param_grid,
                    cv=3)
grid_result = grid.fit(X_train, y_train)
# print results
'''
print(f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f' mean={mean:.4}, std={stdev:.4} using {param}')

'''
# print results
print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'mean={mean:.4}, std={stdev:.4} using {param}')
'''
para_range= np.arange(0, 12)
train_scores, validation_scores = validation_curve(model_CV, X=X, y=y, param_name="num_nodes",
                                                   param_range=para_range, scoring="accuracy")
mean_train_score = np.mean(train_scores, axis = 1)
std_train_score = np.std(train_scores, axis = 1)
mean_test_score = np.mean(validation_scores, axis = 1)
std_test_score = np.std(validation_scores, axis = 1)

plt.plot(para_range, mean_train_score, label="Training Score", color='b')
plt.plot(para_range, mean_test_score, label="Cross Validation Score", color='g')



plt.ylabel("Accuracy")
plt.xlabel("number of nodes in layer")
plt.title("Validation scores for ANN")
plt.tight_layout()
plt.legend(loc = 'best')
plt.savefig('figures/Validation_scores_for_ANN.png')
plt.show()

'''
#classifier_history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, validation_data=(X_test, y_test))
#y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
# plotting code comes from https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594
# Plot the loss function
#plt.ylabel(r'Loss')
#plt.xlabel(r'Epoch')
#plt.title("Loss Function for ANN")


#fig, ax = plt.subplots(1, 1, figsize=(10,6))
#plt.plot(np.sqrt(classifier_history.history['loss']), 'r', label='train')
#plt.plot(np.sqrt(classifier_history.history['val_loss']), 'b' ,label='val')
#ax.set_xlabel(r'Epoch', fontsize=20)
#ax.set_ylabel(r'Loss', fontsize=20)
#ax.legend()
#plt.tight_layout()
#plt.legend(loc = 'best')

#ax.tick_params(labelsize=20)
#plt.show()
#plt.savefig('figures/Loss_function_for_ANN.png')
#plt.show()

# Plot the accuracy
#fig, ax = plt.subplots(1, 1, figsize=(10,6))
#plt.plot(np.sqrt(classifier_history.history['acc']), 'r', label='train')
#plt.plot(np.sqrt(classifier_history.history['val_acc']), 'b' ,label='val')
#plt.xlabel(r'Epoch')
#plt.ylabel(r'Accuracy')
#plt.legend(loc = 'best')
#plt.title("Accuracy function for ANN")
#ax.set_xlabel(r'Epoch', fontsize=20)
#ax.set_ylabel(r'Accuracy', fontsize=20)
#ax.legend()

#plt.savefig('figures/Accuracy_function_for_ANN.png')
#plt.show()
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

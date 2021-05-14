import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset
dataset = pd.read_csv('datasets/inventor-patent-tickers-dates-prices-centrality.csv')
#X = dataset.iloc[:, [1:10], [11:18], [20:23]].values


for col in dataset.keys():
  if ((col == 'Firstname') or (col == 'Lastname') or (col == 'City') or (col == 'State') or (col == 'Country')
  or (col == 'Zipcode') or (col == 'Patent') or (col == 'AppYear') or (col == 'GYear') or (col == 'AppDate')
  or (col == 'Assignee') or (col == 'Tickers') or (col == 'Month Before Application Date') or (col == 'Application Date')
  or (col == 'Month After Application Date')):
    dataset[col] = pd.Categorical(dataset[col], ordered=True).codes
    pd.to_numeric(dataset[col], downcast='float')

#X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,20,21,22]].values
X = dataset.iloc[:, [13,14,15,16,17,20,21,22]].values
y = dataset.iloc[:, 19].values
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
sc = StandardScaler()
print("about to normalize")
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("normalized")

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential #used to initialize the NN

from keras.layers import Dense  #used to build the hidden Layers
from keras.layers import Dropout

# Initialising the ANN
classifier = keras.Sequential()
print("classifier initialized")
# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
print("input layer done")
classifier.add(Dropout(0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
print("hidden layer done")
classifier.add(Dropout(0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
print("about to compile")
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
print("fitting")
classifier_history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, validation_data=(X_test, y_test))
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# plotting code comes from https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594
# Plot the loss function
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(classifier_history.history['loss']), 'r', label='train')
ax.plot(np.sqrt(classifier_history.history['val_loss']), 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)
#plt.show()

# Plot the accuracy
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(classifier_history.history['acc']), 'r', label='train')
ax.plot(np.sqrt(classifier_history.history['val_acc']), 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)
plt.show()
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
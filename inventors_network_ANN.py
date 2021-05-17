import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn


# Importing the dataset

new_df = pd.read_csv('datasets/inventor-patent-tickers-dates-prices-centrality-to-numbers.csv')
print("Dataset with values converted to numbers")
print(new_df)


X = new_df.iloc[:, 0:19].values

y= new_df.iloc[:, 19].values


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
seed = 7
np.random.seed(seed)
# analysis on ANN


model_init_epochs_batch_CV = KerasClassifier(build_fn=create_classifier2, verbose=1)

init_mode = ['glorot_uniform', 'glorot_normal']
batches = [68, 128, 256, 512]
epochs = [10, 20, 30, 40, 50, 60]


param_grid = dict(epochs=epochs, batch_size=batches, init=init_mode)


grid = RandomizedSearchCV(estimator=model_init_epochs_batch_CV,
                    param_distributions=param_grid,
                    cv=3)
grid_result = grid.fit(X_train, y_train)

# print results

print(f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f' mean={mean:.4}, std={stdev:.4} using {param}')


# print results
print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'mean={mean:.4}, std={stdev:.4} using {param}')

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


def create_classifier3(nodes=3):
    classifier = keras.Sequential()
    print("classifier initialized")
    # Adding the input layer and the first hidden layer with dropout
    classifier.add(Dense(units=nodes, kernel_initializer='glorot_uniform', activation='relu', input_dim=19))
    print("input layer done")
    classifier.add(Dropout(0.1))

    # Adding the second hidden layer
    classifier.add(Dense(units=nodes, kernel_initializer='glorot_uniform', activation='relu'))
    print("hidden layer done")
    classifier.add(Dropout(0.1))

    # Adding the output layer
    classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    print("about to compile")
    # Compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return classifier
node_range = np.arange(1,13)
model3 = KerasClassifier(build_fn=create_classifier3, verbose=1)
train_scores, validation_scores = validation_curve(model3, X=X, y=y, param_name="nodes",
                                                   param_range=node_range, scoring="accuracy")
mean_train_score = np.mean(train_scores, axis = 1)
std_train_score = np.std(train_scores, axis = 1)
mean_test_score = np.mean(validation_scores, axis = 1)
std_test_score = np.std(validation_scores, axis = 1)

plt.plot(node_range, mean_train_score, label="Training Score", color='b')
plt.plot(node_range, mean_test_score, label="Cross Validation Score", color='g')



plt.ylabel("Accuracy")
plt.xlabel("number of nodes in layer")
plt.title("Validation scores for ANN")
plt.tight_layout()
plt.legend(loc = 'best')
plt.savefig('figures/Validation_curve_for_ANN.png')
plt.show()

# plotting of graphs for best params determined by RandomizedSearchcv

classifier = keras.Sequential()
print("classifier initialized")
# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(units=6, kernel_initializer='glorot_uniform', activation='relu', input_dim=19))
print("input layer done")
classifier.add(Dropout(0.1))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='glorot_uniform', activation='relu'))
print("hidden layer done")
classifier.add(Dropout(0.1))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
print("about to compile")
# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
classifier_history = classifier.fit(X_train, y_train, batch_size = 128, epochs = 60, validation_data=(X_test, y_test))
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Plot the loss function
plt.ylabel(r'Loss')
plt.xlabel(r'Epoch')
plt.title("Loss Function for Optimized ANN")

plt.plot(classifier_history.history['loss'], 'r', label='train')
plt.plot(classifier_history.history['val_loss'], 'b' ,label='val')

plt.tight_layout()
plt.legend(loc = 'best')
plt.savefig('figures/Loss_function_for_optimized_ANN.png')
plt.show()



# Plot the accuracy
plt.ylabel(r'Accuracy')
plt.xlabel(r'Epoch')
plt.title("Accuracy Function for Optimized ANN")

plt.plot(classifier_history.history['acc'], 'r', label='train')
plt.plot(classifier_history.history['val_acc'], 'b' ,label='val')

plt.legend(loc = 'best')
plt.savefig('figures/Accuracy_function_for_optimized_ANN.png')
plt.show()


# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
sn.set(font_scale=1.4) # for label size
sn.heatmap(cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()
plt.savefig('figures/confusion_matrix_for_optimized_ANN')

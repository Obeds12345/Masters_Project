import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold, KFold
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPool2D,MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

print(tf.__version__)
#### Your Code Goes Here #### 
print("OK")

dataframe = pd.read_csv("heart.csv")

target = dataframe["target"].values
X = dataframe.drop(['target'], axis=1).values
Y = dataframe[['target']].values
Y = Y.ravel()


# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = 100
no_epochs = 25
optimizer = Adam()
verbosity = 1
num_folds = 10


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((Y_train, Y_test), axis=0)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

in_shape = X_train.shape[1:]

X_reshape = X.reshape(X.shape[0], X.shape[1], 1)
input_shape = X_reshape.shape

acc_per_fold = []
loss_per_fold = []

kfold = KFold(n_splits=5,  random_state=0, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    # Define the model architecture
    # model = Sequential()
    # model.add(Conv1D(32, kernel_size=2, activation='relu', input_shape=input_shape))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(64, kernel_size=2, activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(1, activation='softmax'))

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    # model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])


    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    # history = model.fit(X, Y, batch_size=batch_size, epochs=no_epochs, verbose=verbosity)

    print(model.summary())
    model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
    model.fit(inputs[train], targets[train], epochs = 256, batch_size=128, verbose = 1 )


    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
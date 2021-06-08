import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model, Sequential

import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPool2D,MaxPooling2D, Flatten, Dropout, BatchNormalization


print(tf.__version__)

#### Your Code Goes Here #### 
print("OK")

dataframe = pd.read_csv("heart.csv")

target = dataframe["target"].values
X = dataframe.drop(['target'], axis=1).values
Y = dataframe[['target']].values
Y = Y.ravel()


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


in_shape = X_train.shape[1:]



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

print(model.summary())

model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])

# model.fit(X_train, Y_train, epochs=10, batch_size=128, verbose=0)
model.fit(X_train, Y_train, epochs = 256, batch_size=128, verbose = 1 )

# evaluate the model
loss, acc = model.evaluate(X_test, Y_test)
print('Accuracy: %.3f' % acc)
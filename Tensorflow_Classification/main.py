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


print(tf.__version__)

#### Your Code Goes Here #### 
print("OK")

## Step 1: Load Data from CSV File ####
dataframe = pd.read_csv("heart.csv")

target = dataframe["target"].values
X = dataframe.drop(['target'], axis=1).values
Y = dataframe[['target']].values
Y = Y.ravel()

print(X.shape)
print(Y.shape)

mean = X.mean(axis = 0)

X -= mean
std = X.std(axis = 0)
X /= std

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = Sequential()
model.add(Dense(8, input_dim = len(X_train[0,:]), activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))


# print(model.summary())

model.compile(loss = 'binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x = X_train, y = Y_train, epochs = 265, verbose = 1 )

prediction = model.predict(X_test)
print(prediction[:10])
print(Y_test[:10])

my_accuracy = accuracy_score(Y_test, prediction.round())
print(my_accuracy)


# input_layer = Input(shape=(X.shape[1],))
# dense_layer_1 = Dense(15, activation='relu')(input_layer)
# dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
# output = Dense(Y.shape[1], activation='softmax')(dense_layer_2)

# model = Model(inputs=input_layer, outputs=output)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# print(model.summary())
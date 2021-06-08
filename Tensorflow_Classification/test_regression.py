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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# print('',X.shape[1])
input_layer = Input(shape=(X.shape[1],))
dense_layer_1 = Dense(100, activation='relu')(input_layer)
dense_layer_2 = Dense(50, activation='relu')(dense_layer_1)
dense_layer_3 = Dense(25, activation='relu')(dense_layer_2)
output = Dense(1)(dense_layer_3)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

print(model.summary())

history = model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)


print("Test Score:", score[0])
print("Test Accuracy:", score[1])

# predictions = model.predict(X_test)

# print("Correct Classify Instances:", metrics.accuracy_score(y_test,predictions,normalize=True))
# print("Incorrect Classify Instances:", predictions.size - metrics.accuracy_score(y_test,predictions,normalize=True))
# print("Accuracy Score: ", model.score(X_test,y_test))
# print("Error: ", 1 - model.score(X_test,y_test))




from sklearn.metrics import mean_squared_error
from math import sqrt

# pred_train = model.predict(X_train)
# print(np.sqrt(mean_squared_error(y_train,pred_train)))

# pred = model.predict(X_test)
# print(np.sqrt(mean_squared_error(y_test,pred)))
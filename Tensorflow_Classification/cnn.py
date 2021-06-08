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

from tensorflow.keras.layers import Conv2D, Conv1D, MaxPool2D, Flatten, Dropout, BatchNormalization


print(tf.__version__)

#### Your Code Goes Here #### 
print("OK")

dataframe = pd.read_csv("heart.csv")

target = dataframe["target"].values
X = dataframe.drop(['target'], axis=1).values
Y = dataframe[['target']].values
Y = Y.ravel()


print(X.shape) 
print(Y.shape) 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print('X_train',X_train.shape) 
print('X_test',X_test.shape) 

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print('X_train',X_train.shape) 
print('X_test',X_test.shape) 

# print('1',X_train.shape[0]) 
# print('2',X_train.shape[1]) 
# print(type(X_train.shape))

in_shape = X_train.shape[1:]
print('X_train.shape[0:]', X_train.shape[0:])
print('X_train.shape[1:]', X_train.shape[1:])


# model = Sequential()
visible = in_shape
conv1 = Conv2D(16, kernel_size=(3,3), activation='relu', strides=(1, 1))(visible)
conv2 = Conv2D(16, kernel_size=(3,3), activation='relu', strides=(1, 1))(conv1)
bat1 = BatchNormalization()(conv2)
zero1 = ZeroPadding2D(padding=(1, 1))(bat1)

conv3 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.05))(zero1)
conv4 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.05))(conv3)
bat2 = BatchNormalization()(conv4)

conv5 = Conv2D(64, kernel_size=(3,3), activation='relu',strides=(1, 1), padding='valid')(bat2)
conv6 = Conv2D(64, kernel_size=(3,3), activation='relu',strides=(1, 1), padding='valid')(conv5)
bat3 = BatchNormalization()(conv6)
pool1 = MaxPooling2D(pool_size=(2, 2))(bat3)
zero2 = ZeroPadding2D(padding=(1, 1))(pool1)

conv7 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.01))(zero2)
conv8 = Conv2D(128, kernel_size=(2,2), activation='relu', strides=(1, 1), padding='valid')(conv7)
bat4 = BatchNormalization()(conv8)

conv9 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.02))(bat4)
conv10 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.02))(conv9)
bat5 = BatchNormalization()(conv10)

conv11 = Conv2D(64, kernel_size=(3,3), activation='relu', strides=(1, 1))(bat5)
conv12 = Conv2D(64, kernel_size=(3,3), activation='relu', strides=(1, 1))(conv11)
bat6 = BatchNormalization()(conv12)
pool2 = MaxPooling2D(pool_size=(2, 2))(bat6)

conv13 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.02))(pool2)
conv14 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.02))(conv13)
bat7 = BatchNormalization()(conv14)

conv15 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.05))(bat7)
conv16 = Conv2D(128, kernel_size=(2,2), activation='relu', strides=(1, 1), padding='valid')(conv15)
bat8 = BatchNormalization()(conv16)

flat = Flatten()(bat8)
hidden1 = Dense(32, activation='relu')(flat)
drop1 = Dropout(0.3)(hidden1)

hidden2 = Dense(32, activation='relu')(drop1)
drop2 = Dropout(0.2)(hidden2)

output = Dense(5, activation='sigmoid')(drop2)
model = Model(inputs=visible, outputs=output)

# model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=in_shape))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(1, activation="sigmoid"))





print(model.summary())

model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])

# model.fit(X_train, Y_train, epochs=10, batch_size=128, verbose=0)
model.fit(X_train, Y_train, epochs = 256, batch_size=128, verbose = 1 )

# evaluate the model
loss, acc = model.evaluate(X_test, Y_test)
print('Accuracy: %.3f' % acc)
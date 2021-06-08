import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#### Your Code Goes Here #### 
print("OK")

## Step 1: Load Data from CSV File ####
dataframe = pd.read_csv("heart.csv")

## Step 2: Plot the Data ####
target = dataframe["target"].values

# X = dataframe[['age','slope']].values
X = dataframe.drop(['target'], axis=1).values
Y = dataframe[['target']].values
Y = Y.ravel()
# print(Y)


model = svm.SVC(kernel='linear', C = 1.0)
# model train
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, Y, cv=5)
print('scores :', scores)
print('cross_val_score Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
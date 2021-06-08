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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


#### Your Code Goes Here #### 
print("OK")

## Step 1: Load Data from CSV File ####
dataframe = pd.read_csv("heart.csv")

## Step 2: Plot the Data ####
target = dataframe["target"].values

X = dataframe.drop(['target'], axis=1).values
Y = dataframe[['target']].values
Y = Y.ravel()
# print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# scalling data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

print("Correct Classify Instances:", metrics.accuracy_score(Y_test,predictions,normalize=False))
print("Incorrect Classify Instances:", predictions.size - metrics.accuracy_score(Y_test,predictions,normalize=False))
print("Accuracy Score: ", model.score(X_test,Y_test))
print("Error: ", 1 - model.score(X_test,Y_test))


# model train
from sklearn.model_selection import cross_val_score
X = scaler.transform(X)
# Y = scaler.transform(Y)
cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(model, X, Y, cv=cv, scoring='accuracy', n_jobs=-1)
print('scores :', scores, 'scores :')
print('cross_val_score Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(Y_test,predictions))
# print(classification_report(Y_test,predictions))

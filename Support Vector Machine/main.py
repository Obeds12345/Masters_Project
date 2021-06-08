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
ages = dataframe["age"].values
sex = dataframe["sex"].values
cp = dataframe["cp"].values
trestbps = dataframe["trestbps"].values
slope = dataframe["slope"].values
thal = dataframe["thal"].values
ca = dataframe["ca"].values
target = dataframe["target"].values

colors = []
for item in target:
    if item == 0:
        colors.append("green")
    else:
        colors.append("red")

# X = dataframe[['age','slope']].values
X = dataframe.drop(['target'], axis=1).values
Y = dataframe[['target']].values
Y = Y.ravel()
# print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = svm.SVC(kernel='linear', C = 1.0)

# model train
model.fit(X_train,Y_train)
scores = cross_val_score(model, X, y, cv=5)

predicted_values = model.predict(X_test)


## Step 5: Estimate Error ####
print("Correct Classify Instances:", metrics.accuracy_score(Y_test,predicted_values,normalize=False))
print("Incorrect Classify Instances:", predicted_values.size - metrics.accuracy_score(Y_test,predicted_values,normalize=False))
print("Training Score: ", model.score(X_train,Y_train))
print("Accuracy Score: ", model.score(X_test,Y_test))
print("Error: ", 1 - model.score(X_test,Y_test))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(Y_test, predicted_values))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(Y_test, predicted_values))

# for item in zip(Y_test, predicted_values):
#     print("Auctual was: ", item[0], "Predicted is: ", item[1])


w = model.coef_[0]
print('w', w)

a = -w[0] / w[1]

xx = np.linspace(0,120)
yy = a * xx - model.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X_train[:, 0], X_train[:, 1], c = Y_train)
plt.legend()
plt.show()
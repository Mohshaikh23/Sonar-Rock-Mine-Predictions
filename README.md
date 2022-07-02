# Sonar-Rock-Mine-Predictions
Sonar data for Rock V/S Mine Prediction using Logistic Regression.

# SONAR ROCK MINE PREDICTION PROJECT

### Importing required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Importing data

df = pd.read_csv("Sonar Data.csv", header = None)

df.head()

### performing EDA

df.shape

df.isnull().sum()

df.info()

df.describe()

#Checking for the balance in the dataset
df[60].value_counts()

#Checkiing for outliers misbalancing data
df.groupby(60).mean()

## Feature Selection 

#Splitting up the data into attributes and target
X = df.drop(columns= 60, axis=1)
y= df[60]

X.shape

y.shape

### Feature Splitting

from sklearn.model_selection import train_test_split

#to perform modelling we need to split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

## Model Building

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr

#to check the model working we need to fit the data into the model we have created 
lr.fit(X_train, y_train)

#For Chekcing the working of the model we built we need to import accuracy score from sklearn.metrics
from sklearn.metrics import accuracy_score

## Prediction

#predicting the target for training dataset
X_train_prediction = lr.predict(X_train)

training_data_accuracy = accuracy_score(X_train_prediction, y_train)

print("The Accuracy score for the Training data is :" , round(training_data_accuracy * 100,2), "%")

#predicting the accuracy for test dataset
X_test_prediction = lr.predict(X_test)

test_data_accuracy = accuracy_score(X_test_prediction, y_test)

print("The Accuracy score for the Test data is :" , round(test_data_accuracy * 100,2), "%")

### To check the model is running correct or not

input_data = (0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)

input_data_as_numpy_array= np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction= lr.predict(input_data_reshaped)
print(prediction)

if (prediction == "R"):
    print("It is a Rock")
else:
    print("It is a Mine")

## Thus We can check out the Sonar Waves Prediction using this Model.

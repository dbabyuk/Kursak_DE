import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.stats import mode


# read initial raw data set
data_init = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv')
# Rename first column
data_init.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
# Converting "index" column into str type to exclude further stat operations for it
data_init['index'] = data_init['index'].astype(str)

# Extracting number of rows and columns
_, cols = data_init.shape

# Set output option with all columns displayed
pd.set_option('display.max_columns', cols)

# Select X features for input
X_features = ['wage', 'education', 'distance']
# Select target set Y as 'ethnicity'
Y_feature = ['ethnicity']
# Transform input data into Numpy format
X = data_init[X_features].to_numpy()
Y = data_init[Y_feature].to_numpy().ravel()


# Splitting data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)


estimators = [('DecisionTree', DecisionTreeClassifier(max_depth=11)),
              ('kNN', KNeighborsClassifier(n_neighbors=21)),
              ('RandomForest', RandomForestClassifier()),
              ('svm', SVC(gamma=1.0, C=1.0, probability=True))]


def fit(estimators, X, y):
    for model, estimator in estimators:
        estimator.fit(X, y)
    return estimators

estimators = fit(estimators, X_train, Y_train)


def predict_individual(X, estimators, proba=False):
    n_estimators = len(estimators)
    n_samples = X.shape[0]
    y = np.zeros((n_samples, n_estimators)).astype(str)
    for i, (model, estimator) in enumerate(estimators):
        if proba:
            y[:, i] = estimator.predict_proba(X)[:, 1]
        else:
            y[:, i] = estimator.predict(X)
    return y


y_individ = predict_individual(X_test, estimators, proba=False)
for i, (model, estimator) in enumerate(estimators):
    print(f'Accuracy for individual {model} = {accuracy_score(Y_test, y_individ[:, i])}')


def combine_using_majority_vote(X, estimators):
    y_individual = predict_individual(X, estimators, proba=False)
    y_final = mode(y_individual, axis=1)
    return y_final[0].reshape(-1, )


ypred = combine_using_majority_vote(X_test, estimators)
acc_combined = accuracy_score(Y_test, ypred)
print()
print('Accuracy for combined estimators = ', acc_combined)
# The majority vote gives higher accuracy than each single predictor.
# Unfortunatelly I did not have time to make weighted calculations etc

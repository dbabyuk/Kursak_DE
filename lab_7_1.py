import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


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

# Single prediction for DecisionTree
tree = DecisionTreeClassifier(random_state=100)
tree.fit(X_train, Y_train)
y_pred_single = tree.predict(X_test)
acc_tree_single = accuracy_score(Y_test, y_pred_single)
print('Single Accuracy Tree = ', acc_tree_single)

# Parallel ensemble for Bag Classifier
bag_ens = BaggingClassifier(base_estimator=tree, n_estimators=600,
                            max_samples=500, oob_score=True, random_state=100)
bag_ens.fit(X_train, Y_train)
ypred_bag = bag_ens.predict(X_test)

acc_bag = accuracy_score(Y_test, ypred_bag)
print('Bagging Accuracy Tree = ', acc_bag)
# Bagging ensemble improves the accuracy by 6% for set if optimized parameters

# Single prediction for kNN
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
ypred_knn = knn.predict(X_test)
acc_knn_single = accuracy_score(Y_test, ypred_knn)
print()
print('Single Accuracy kNN = ', acc_knn_single)


# Parallel ensemble for ADABoosting
adaboost_ens = AdaBoostClassifier(base_estimator=tree,
                                  n_estimators=15, learning_rate=0.75, random_state=100)
adaboost_ens.fit(X_train, Y_train)
ypred_ada = adaboost_ens.predict(X_test)
acc_ada = accuracy_score(Y_test, ypred_ada)
print('ADABoosting Accuracy = ', acc_ada)
# ADABoosting improves a little the accuracy. I did not have time for the remaining items of this lab

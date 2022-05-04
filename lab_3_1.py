import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.model_selection import GridSearchCV

# -- Step 1
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

# --Step 2
# Select X features for input
X_features = ['wage', 'education', 'distance']
# Select target set Y as 'ethnicity'
Y_feature = ['ethnicity']
# Transform input data into Numpy format
X = data_init[X_features].to_numpy()
Y = data_init[Y_feature].to_numpy()

# --Step 3
# Splitting data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# --Step 4
# Train data with default parameters
dt_default = tree.DecisionTreeClassifier(criterion='gini', random_state=100)
dt_default.fit(X_train, Y_train)
# Predict Y values for test set
Y_predicted = dt_default.predict(X_test)

# --Step 5
# Estimation of accuracy ratio and confusion matrix for default parameter set
acc_default = accuracy_score(Y_test, Y_predicted)
print('Accuracy of default Decision Tree = ', acc_default)
print()
conf_matrix = confusion_matrix(Y_test, Y_predicted)
print('Confusion Matrix')
print(conf_matrix)
print()

# --Step 6
# Preparing parameters for Decision Tree optimization
max_depth = range(7, 15)
min_samples_leaf = range(7, 13)
min_samples_split = range(2, 6)

param_grid = dict(min_samples_leaf=min_samples_leaf,
                  max_depth=max_depth, min_samples_split=min_samples_split)
grid_search = GridSearchCV(dt_default, param_grid, verbose=1)
# Searching best fit for parameters
grid_result = grid_search.fit(X_train, Y_train)
best_params, best_score = (grid_result.best_params_, grid_result.best_score_)
# Output best parameter set
print('Optimized parameters: ')
print(best_params)
print('Estimated accuracy = ', best_score)

# --Step 7
# Training model with optimal parameters
dt_optimized = tree.DecisionTreeClassifier(criterion='gini', max_depth=best_params['max_depth'],
                                           min_samples_leaf=best_params['min_samples_leaf'],
                                           min_samples_split=best_params['min_samples_split'], random_state=100)
dt_optimized.fit(X_train, Y_train)
Y_predicted_opt = dt_optimized.predict(X_test)
# Accuracy evaluation of optimized model
acc_optimized = accuracy_score(Y_test, Y_predicted_opt)
print()
print('Accuracy of optimized Decision Tree = ', acc_optimized)

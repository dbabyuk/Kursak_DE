import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
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
Y = data_init[Y_feature].to_numpy().ravel()

# --Step 3
# Splitting data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# --Step 4
# Train data with default parameters
svm_default = svm.SVC()
svm_default.fit(X_train, Y_train)
# Predict Y values for test set
Y_predicted = svm_default.predict(X_test)

# --Step 5
# Estimation of accuracy ratio and confusion matrix for default parameter set
acc_default = accuracy_score(Y_test, Y_predicted)
print('Accuracy of default SVM = ', acc_default)
print()
conf_matrix = confusion_matrix(Y_test, Y_predicted)
print('Confusion Matrix')
print(conf_matrix)
print()

# Searching best fit for parameters
Cs = [0.5, 1, 2]
gammas = [100, 500, 1000]
param_grid = {'C': Cs, 'gamma': gammas}
svm_grid = GridSearchCV(svm_default, param_grid, verbose=1)
grid_result = svm_grid.fit(X_train, Y_train)
best_params, best_score = (grid_result.best_params_, grid_result.best_score_)
# Output best parameter set
print('Optimized parameters: ')
print(best_params)
print('Estimated accuracy = ', best_score)

# --Step 7
# Training model with optimal parameters
Y_predicted_opt = svm_grid.predict(X_test)
# Accuracy evaluation of optimized model
acc_optimized = accuracy_score(Y_test, Y_predicted_opt)
print()
print('Accuracy of optimized svm = ', acc_optimized)
conf_matrix = confusion_matrix(Y_test, Y_predicted_opt)
print('Confusion Matrix')
print(conf_matrix)
print()

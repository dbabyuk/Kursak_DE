import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

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

# --Step 1
# Group by "ethnicity" and show "score" stat
print()
print('Score stat for "ethnicity"')
print(data_init.groupby(['ethnicity'])['score'].describe())
# Group by "gender" and show "score" stat
print()
print('Score stat for "gender"')
print(data_init.groupby(['gender'])['score'].describe())
# Sort by "score"
print()
print('Sorted by "score"')
print(data_init.sort_values(by='score', ascending=False))

# -- Step 2
# Crosstab for some features
print()
print('Crosstab for ethnicity - income normalized by ethnicity')
print(pd.crosstab(data_init['ethnicity'], data_init['income'], normalize='index'))
print()

# -- Step 3
# Normal distribution validation
data_init.hist(['score'])
plt.show()
stats.probplot(data_init['score'], dist=stats.norm, plot=plt)
plt.show()

_, pvalue = stats.shapiro(data_init['score'])
print('p-value = ', pvalue)


box_cox_data = stats.boxcox(data_init['score'], lmbda=0.94)
_, pvalue = stats.shapiro(box_cox_data)
print('Box-Cox transformed')
print('p-value = ', pvalue)

stats.probplot(box_cox_data, dist=stats.norm, plot=plt)
plt.show()


log_data = np.log10(data_init['score'])
_, pvalue = stats.shapiro(log_data)
print('Log transformed')
print('p-value = ', pvalue)

stats.probplot(log_data, dist=stats.norm, plot=plt)
plt.show()

# --Step 4
# Get correlation matrix
corr_matrix = data_init.corr()
# Get rid of diagonal elements
corr_matrix.replace({1.0: 0}, inplace=True)

print()
print('Correlation Matrix:')
print(corr_matrix)

# Get pairs  with the largest correlations
pairs = corr_matrix.idxmax()
print()
print('The following features are the most correlated:')
print(pairs)

# Visualize the correlation matrix
plt.matshow(corr_matrix)
plt.show()

# --Step 5
# Binning "score" into three categories: bad, good, excellent
data_init['score'] = pd.cut(data_init.score, 3, labels=['bad', 'good', 'excellent'])
print()
print('score is now category')
print(data_init['score'].head())

# Converting categorical "ethnicity" feature into numerical [0, 1, 2]
data_init['ethnicity'].replace({'afam': 0, 'hispanic': 1, 'other': 2}, inplace=True)
print()
print('ethnicity is now numerical')
print(data_init['ethnicity'].head())

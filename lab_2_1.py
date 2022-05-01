import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize

# --Step 1
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
# Output descriptive statistics for numerical features
print('Descriptive statistics: INITIAL')
print(data_init.describe())
print()

# --Step 3
# Create alternative data set
data_main = data_init.copy()
data_main.boxplot()
plt.show()

number_of_nulls = data_main.isnull().sum().sum()
number_of_nans = data_main.isna().sum().sum()

print('Number of nulls = ', number_of_nulls)
print('Number of NaNs = ', number_of_nans)


# --Step 5 (No nulls and Nones)
# Replace some values of feature 'score' with None
number_of_samples = 30
# Select random sample for values to be replaced
values_to_none = data_main['score'].sample(number_of_samples).to_list()
# Replacement to None
data_main['score'] = data_main['score'].replace(values_to_none, [None] * number_of_samples)

# --Step 4 (Step 5 preceded Step 4 because Nones were artificially created)
# Replace None with mean value
data_main['score'].fillna(data_main['score'].mean(), inplace=True)
print()
print('Descriptive statistics for "score" after treatment missing data')
print(data_main['score'].describe())


# --Step 6
# Outliers treatment
# Winsorize all columns
for col in data_main.columns:
    if col != 'index':
        winsorize(data_main[col], limits=(0.05, 0.08), inplace=True)
data_main.hist()
plt.show()

# --Step 7
# Output descriptive stats after Outliers treatment
print()
print('Descriptive statistics: after OUTLIERS treatment')
print(data_main.describe())

# --Step 8
# Data normalization
data_normal = (data_main - data_main.mean(numeric_only=True))/data_main.std(numeric_only=True)
print()
print('Sum of normalized data:')
print(data_normal.sum())
print()
print('Descriptive statistics: NORMALIZED')
print(data_normal.describe())

data_normal.boxplot()
plt.show()

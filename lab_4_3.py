import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# read initial raw data set
data_init = pd.read_csv('Pizza.csv')

data_init.drop(columns=['id'], inplace=True)
unique_names = sorted(list(set(data_init.brand)))
brand_mapper = dict(zip(unique_names, range(len(unique_names))))
data_init['brand'].replace(brand_mapper, inplace=True)

# Extracting number of rows and columns
_, cols = data_init.shape

# Set output option with all columns displayed
pd.set_option('display.max_columns', cols)

# Data definition
X = data_init.iloc[:, 1:7].values
Y = data_init.iloc[:, 0].values


# Splitting data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)


# PCA: Finding optimal number of principal components by estimating cumulative explained variance
scaler = StandardScaler()
pca = PCA()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
pca.fit(X_train, Y_train)
explained_variance = pca.explained_variance_
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('components_number')
plt.ylabel('Cumulative explained variance')
plt.show()

# The plot reveals that n_components = 2
# Use of PCA with n_components=2 with subsequent Logistic regression and prediction
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
classifier = LogisticRegression(random_state=100)
classifier.fit(X_train, Y_train)
Y_predicted = classifier.predict(X_test)

# Computing confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_predicted)
print('PCA Confusion Matrix')
print(conf_matrix)
print()

# Plotting final data

plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test[:], marker='.')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PCA with 2 components')
plt.show()

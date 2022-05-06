import pandas as pd
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np

number_of_clusters = 3

X, y_true = make_blobs(n_samples=500, centers=number_of_clusters, cluster_std=3, random_state=45)
X[:, 0] = np.rint((X[:, 0] + 15) *10000)
X[:, 1] = X[:, 1] + 20

plt.scatter(X[:, 0], X[:, 1])
plt.show()


df_init = pd.DataFrame(X, columns=['annual_income', 'home_index'])
df_init['expected'] = y_true


csv_data = df_init.to_csv()
with open('clustering_data.csv', 'w') as file:
    file.write(csv_data)
    file.close()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from sklearn_extra.cluster import KMedoids
from sklearn import preprocessing
from sklearn import metrics


data_init = pd.read_csv("BrentOilPrices.csv", index_col='Date')
data_init.index = pd.to_datetime(data_init.index)

init_year = 1988
end_year = 2020
start_date = pd.to_datetime(f'{init_year}-01-01')
end_date = pd.to_datetime(f'{end_year}-01-01')

new_data = data_init[(data_init.index >= start_date) & (data_init.index < end_date)]

data_resampled = new_data['Price'].resample('W').mean().to_frame()
plt.plot(data_resampled)
plt.show()

cycle, trend = sm.tsa.filters.hpfilter(data_resampled['Price'])
fig, ax = plt.subplots(3, 1)
ax[0].plot(data_resampled['Price'])
ax[0].set_title('Original')
ax[1].plot(trend)
ax[1].set_title('Trend')
ax[2].plot(cycle)
ax[2].set_title('Cycle')
plt.show()
#
res = STL(new_data['Price']).fit()
chart = res.plot()
plt.show()
#
res = seasonal_decompose(new_data['Price'], model='additive',  period=30)
x = res.plot()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, Holt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Data read and time series (TS) creation
data_init = pd.read_csv("global_temperature.csv", index_col='dt')
data_init.index = pd.to_datetime(data_init.index)
main_feature = 'LandAverageTemperature'

# Output of raw time series data
plt.plot(data_init[main_feature])
plt.title('Raw Data')
plt.ylabel(main_feature)
plt.show()
# Visualization of resampled by year mean
data_resampled = data_init[main_feature].resample('Y').mean()
plt.plot(data_resampled)
plt.ylabel(main_feature)
plt.title('Resampled yearly mean')
plt.show()

# Verification for NaN in the TS
nans = data_init[main_feature].isna().values.sum()
print()
print('The number of NaN = ', nans)
print()
# There's no need to fill empty values in the TS because nans = 0

# The data is very dense. Therefore only its slice in range (init_year, end_year) will be considered
init_year = 1970
end_year = 2000
fill_delta = 10  # Number of years for future forecast

start_date = pd.to_datetime(f'{init_year}-01-01')
end_date = pd.to_datetime(f'{end_year}-01-01')
future_date = pd.to_datetime(f'{end_year + fill_delta}-01-01')
data_main = data_init[(data_init.index >= start_date) & (data_init.index < end_date)][main_feature].to_frame()
data_future = data_init[(data_init.index >= end_date) & (data_init.index < future_date)][main_feature].to_frame()

# Decomposition of the TS into trend, seasonal and residual for additive model
data_decomposed = seasonal_decompose(data_main[main_feature], model='additive')
data_decomposed.plot()
plt.show()

# Creation of noise-free TS
data_cleaned = data_decomposed.trend + data_decomposed.seasonal

# Plot of noise
plt.plot(data_decomposed.resid, color='k')
plt.title('Extracted Noise')
plt.show()

# Plot of cleaned TS vs original
fig = plt.figure()
axes = fig.add_subplot(111)
line_1, = axes.plot(data_cleaned, color='r', linestyle='-.')
line_1.set_label('Cleaned')
line_2, = axes.plot(data_main[main_feature], color='b', linestyle=':')
line_2.set_label('Original')
axes.legend(loc=1)
plt.ylabel(main_feature)
plt.title('Noise-free TS vs original')
plt.show()

# Smoothing with Holt and ExponentialSmoothing methods
alpha = 0.2
fitted_holt = Holt(data_main).fit(smoothing_level=alpha)
fitted_exp = ExponentialSmoothing(data_main, seasonal='add').fit(smoothing_level=alpha)
smoothed_exp = fitted_exp.fittedvalues
smoothed_holt = fitted_holt.fittedvalues
predicted_holt = fitted_holt.forecast(12 * fill_delta)
predicted_exp = fitted_exp.forecast(12 * fill_delta)

# Plot of smoothing results
fig = plt.figure()
axes = fig.add_subplot(111)
original = axes.plot(data_main[main_feature].iloc[-96:], color='k', label='original TS')
exact_future = axes.plot(data_future, marker='*', color='r', linestyle='-.', label='exact TS')
exponential = axes.plot(predicted_exp, marker='o', linestyle='None',
                        markerfacecolor='None', label='Exponential Smoothing')
holt = axes.plot(predicted_holt, linestyle='-.', color='g',
                 markerfacecolor='None', label='Holt Smoothing')
axes.legend(loc=1)
plt.title(f'Forecasts for smoothing: alpha={alpha}')
plt.ylabel(main_feature)
plt.show()

# As seen from the graph, Holt smoothing produces linear forecast while Exponential Smoothing
# reflects seasonal behavior of the TS

# Forecast reliability for Exponential smoothing
mean_abs_err = mean_absolute_error(data_future, predicted_exp)
mean_s_err = mean_squared_error(data_future, predicted_exp)
print()
print('Exponential Smoothing prediction:')
print('Mean Square Error = ', mean_s_err)
print('Mean Absolute Error = ', mean_abs_err)

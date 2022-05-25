import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import statsmodels as sm


# Data read and time series (TS) creation
data_init = pd.read_csv("global_temperature.csv", index_col='dt')
data_init.index = pd.to_datetime(data_init.index)
main_feature = 'LandAverageTemperature'

# The data is very dense. Therefore only its slice in range (init_year, end_year) will be considered
init_year = 1970
end_year = 2000
fill_delta = 10  # Number of years for future forecast


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


start_date = pd.to_datetime(f'{init_year}-01-01')
end_date = pd.to_datetime(f'{end_year}-01-01')
future_date = pd.to_datetime(f'{end_year + fill_delta}-01-01')
data_main = data_init[(data_init.index >= start_date) & (data_init.index < end_date)][main_feature].to_frame()

# The best autocorrelation is realized at shift = 12. It is intuitive because the TS is periodic with 12 months.
# Therefore parameter d in SARISMA will be set to 12
shift = 12
diff = data_main - data_main.shift(shift)
diff = diff.dropna()
tsplot(diff[main_feature], lags=50)
plt.show()

# Analysis of Autocorellation graph reveals that parameter p should be in range 12-14
# Parameter q is up to 24. So the candidate models will be with parameter set (12-14, 1, 24)

model_ = sm.tsa.statespace.sarimax.SARIMAX(data_main[main_feature], order=(1, 12, 12)).fit()
model_.plot_diagnostics()
plt.show()





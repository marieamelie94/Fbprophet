import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
df = pd.read_csv('orders.csv')
df.head()

prophet = Prophet(yearly_seasonality = True, weekly_seasonality=True, seasonality_prior_scale = 10)
# to add monthly seasonality, check with different fourier orders
# prophet.add_seasonality(name='monthly', period=30.5, fourier_order= 20)

prophet.fit(df)

future = prophet.make_future_dataframe(periods=180, include_history = True)
future.tail()

forecast = prophet.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

forecast_orders_csv = forecast.to_csv('orders_forecast_sps10.csv')

# cross validation with actual y and ywhat
# df_cv = cross_validation(prophet, horizon = '180 days')
# df_cv.head()
# df_cv.to_csv('cv_orders_forecast_sps10.csv')


# from fbprophet.diagnostics import performance_metrics
# df_p = performance_metrics(df_cv)
# df_p.head()

# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
# fig1 = prophet.plot(forecast)


#-----------------------------------------forecast monthly
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
import numpy as np
df = pd.read_csv('monthly_orders.csv')
df.head()

#to log transform the ‘y’ variable to a try to convert non-stationary data to stationary
df['y_orig'] = df['y'] # to save a copy of the original data
# log-transform y
df['y'] = np.log(df['y'])

prophet = Prophet(yearly_seasonality = True, weekly_seasonality=False)
#.add_seasonality(name='monthly', period=30.5, fourier_order= 20)
prophet.fit(df)

future = prophet.make_future_dataframe(periods=6, freq = 'm', include_history = True)
future.tail()

forecast = prophet.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#get back to non log data
forecast_data_orig = forecast # make sure we save the original forecast data
forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])


forecast.to_csv('monthly_log_orders_forecast.csv')


# monthly returns
df = pd.read_csv('monthly_returns.csv')
df['y_orig'] = df['y'] # to save a copy of the original data
# log-transform y
df['y'] = np.log(df['y'])
df.head()

prophet = Prophet(yearly_seasonality = True, weekly_seasonality=False)
prophet.fit(df)

future = prophet.make_future_dataframe(periods=6, freq = 'm', include_history = True)
future.tail()

forecast = prophet.predict(future)
forecast.to_csv('monthly_returns_log_forecast.csv')


# df_cv = cross_validation(prophet, horizon = 1)
# df_cv.head()
# df_cv.to_csv('cv_orders_forecast_sps10.csv')


# from fbprophet.diagnostics import performance_metrics
# df_p = performance_metrics(df_cv)
# df_p.head()

# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
# fig1 = prophet.plot(forecast)
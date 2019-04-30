
import pandas as pd
from fbprophet import Prophet
import numpy as np
from sqlalchemy import create_engine
from forecast_tables import create_tables, Forecast, session

engine = create_engine('postgresql://msandrock:PWD&@HOST/montredo')
query = """
        select date_trunc('month', order_date::date)::date as ds, count(unique_order_id) as y
        from dwh_il.fct_orders o
        where order_month >= '2016-01' 
        and (lower(order_status) like '%%new%%')
        group by 1
        order by 1;
        """
df = pd.read_sql(query, engine)

df['y_orig'] = df['y'] # to save a copy of the original data
df['y'] = np.log(df['y'])
df.head()

prophet = Prophet(yearly_seasonality = True, weekly_seasonality=False)
#.add_seasonality(name='monthly', period=30.5, fourier_order= 20)
prophet.fit(df)

future = prophet.make_future_dataframe(periods=6, freq = 'm', include_history = True)
future.tail()

forecast = prophet.predict(future)

#get back to non log data
forecast_data_orig = forecast # make sure we save the original forecast data
forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])
forecast_data_orig[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#load to the db table
forecast_data_orig[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_sql('prophet_forecast_successful_orders',engine, 'forecasting', if_exists='replace')



forecast.to_csv('orders_forecast_test.csv')

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


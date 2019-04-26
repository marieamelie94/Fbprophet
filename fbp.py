import pandas as pd
from fbprophet import Prophet

df = pd.read_csv('orders.csv')
df.head()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

c= forecast.to_csv



from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
fig1 = m.plot(forecast)

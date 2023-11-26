print("Time Series Analysis with ARIMA model")
'''
Time Series Analysis: It is the continuous data only numerical but data will be varies as per the times or year.
        but these data is having the connection with date, month or year or hourly, minutes, daily, weekly, monthly
        quarterly, annually.  

1. Visualize the time series: Trends, Seasonality, Noise (residual), cyclical
2. Make the series stationary or data stationary like adding all the factors to it.
3. Finding the optimal parameters - ACF & PACF plot (IT WILL GIVE PARAMETRES TO US)
4. Building using model: ARIMA
5. Make prediction

ARIMA HAS THREE PARAMETERS.

Trends - Numbers are changing means average increasing or decreasing on daily bases, monthly, quaterly, annually
        increasing or decreasing trend.
Seasonality - As per the season, some businesses increases or decreases and need to see their patterns like above.
Noise - Like there is strike on some day. This is one of the case out of many.
        Due to this factors, increasing trend or decreasing trend in some businesses.
Cyclical - have no fixed time. Cant predict anything about business. This is not happening eveytime.

Searsonal we can predict
but in Cyclical we can not predict.

'''
print("1. Visualize the time series: Trends, Seasonality, Noise (residual)")
import pandas as pd
import matplotlib.pyplot as plt

link = "D:\\TOP MENTOR\\Dataset-master\\Dataset-master\\AirPassengers.csv"
data = pd.read_csv(link)
print(data)
'''
import plotly.express as px
fig = px.line(data,x="Month",y="#Passengers")
fig.show()
'''
plt.figure(figsize=(20, 7))  # This figure just an example.
plt.plot(data["Month"], data["#Passengers"])
plt.show()

# convert Month into datetime
from datetime import datetime

data["Month"] = pd.to_datetime(data["Month"])
data.set_index("Month", inplace=True)
timeseries = data["#Passengers"]

'''
Decomposition of the data:
1. Additive decomposition
    y = t + s + r   # Trend, Seasonality, and Resiual(Noise)
2. Multiplicative decomposition:
    y = t * s * r

Additive model preferred when the seasonal variation is almost constant over time
Multiplicative time series will have increasing seasonality.

In our case, the seasonal graph seems to be increasing so we assume we need to perform
Multiplicative decomposition.

To perform multiplicative decomposition, we apply log
'''
import numpy as np

ts_log = np.log(timeseries)
# call the below library for decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

decom = seasonal_decompose(ts_log)
decom.plot()
plt.show()
# Now data getting seperately
trend = decom.trend
seasonal = decom.seasonal
residual = decom.resid

print("2. Make the series stationary")
from statsmodels.tsa.stattools import adfuller

result = adfuller(timeseries)
print("ADF Stats: ", result[0])
print("ADF p-value: ", result[1])
print("Entire Result:\n", result)

'''
(0.8153688792060482, 0.991880243437641, 
since p-value is > 0.05, we fail to reject null hypopthesis and we conclude
time series is not stationary
'''
result = adfuller(ts_log)
print("ADF Stats: ", result[0])
print("ADF p-value: ", result[1])
print("Entire Result:\n", result)
'''
Even after applying log, the time series is still not stationary.

Now we will apply differencing method: it removes the impact of
previous time period.
Y(t) - Y(t-1)
'''
diff_data = timeseries.diff().dropna()
result = adfuller(diff_data)
print("ADF Stats: ", result[0])
print("ADF p-value: ", result[1])
print("Entire Result:\n", result)
# Now the data is almost stationary, and we can use this to predict

plt.plot(diff_data)
plt.show()

print("3. Finding the optimal parameters - ACF & PACF plot")
''' 
TSA = ARIMA = Auto Regressive (AR) Integrated (I) Moving Average (MA)
AR = applying regression at the current timestamp (time period) 
    from previous time period. factor = p
I = Integrated is used where to make the ts stationary using differencing method.
    factor = d
MA = Moving average tells you how many past timestamps you 
    need to take to predict. If more than 1, then we take the average.
    factor = q

To find a efficient model, we need to get the right values for p,d,q
Autocorrelation (ACF) and Partial ACF (PACF) will help us to get get these values.

p (AR) = look for the period where there is sharp decline in PACF
q (MA) = look at the exponential decrease in ACF plot
d = we need to perform trial and error, start with 1
'''
# ACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(diff_data)  # differencing data
plt.show()  # q=1

plot_pacf(diff_data)
plt.show()  # p=1 or 2
p, i, q = 1, 1, 2

print("4. Building using model: ARIMA")
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(diff_data, order=(p, i, q))
result_ARIMA = model.fit()
# summary of ARIMA model:
print(result_ARIMA.summary())

print("5. Make prediction")
plt.plot(diff_data)
plt.plot(result_ARIMA.fittedvalues, color="red")
plt.show()

# prediction for future time period
data['Predicted'] = result_ARIMA.predict(start=120, end=140)
data[['#Passengers', 'Predicted']].plot()
plt.show()

#
model = ARIMA(data['#Passengers'], order=(p, i, q))
result_ARIMA = model.fit()
data['Predicted'] = result_ARIMA.predict(start=120, end=140)
data[['#Passengers', 'Predicted']].plot()
plt.show()

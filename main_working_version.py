import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objs as go
from plotly.offline import plot


apple = pd.read_csv('C:/Users/Windows 10/Desktop/Apple stocks.csv')
print(apple.head())
print(apple.info())

apple['Date'] = pd.to_datetime(apple['Date'])

print(f"Dataframe contains stock prices between {apple.Date.min()} {apple.Date.max()}")
total_days = (apple.Date.max() - apple.Date.min()).days
print(f"Total days = {total_days} days")

print(apple.describe())

apple[['Open', 'High', 'Low', 'Close', 'Adj Close']].plot(kind='box')


layout = go.Layout(
    title='Stock Prices of Apple',
    xaxis = dict(
        title='Date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18
        )
    ),
    yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courier New, monospace',
            size=18
        )
    )
)

apple_trend  = [{'x':apple['Date'], 'y':apple['Close']}]
apple_plot = go.Figure(data=apple_trend, layout=layout)

plot(apple_plot)


# Regression model:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

X = np.array(apple.index).reshape(-1, 1)
Y = apple['Close']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

scaler = StandardScaler().fit(X_train)

lm = LinearRegression()
lm.fit(X_train, Y_train)

trace0 = go.Scatter(x = X_train.T[0], y = Y_train, mode= 'markers', name = 'Actual')
trace1 = go.Scatter(x = X_train.T[0], y = lm.predict(X_train).T, mode = 'lines', name = 'Predicted')

apple_lm_model = [trace0, trace1]
layout.xaxis.title.text = 'Day'
apple_lm_plot = go.Figure(data=apple_lm_model, layout=layout)

plot(apple_lm_plot)






plt.show()





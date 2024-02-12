import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

WINDOW = 20
SHIFT = 1
RANGE = 15


def historical_data(ticker):
    """
    Accesses Yahoo! Finance's entire historical data for the specified ticker.
    """
    stock_data = yf.download(ticker)
    return stock_data


def perform_linear_regression(stock_data):
    """
    Computes the linear regression model to predict the relationship between two variables and evaluates its performance.
    """
    # We create a new column where each row represents
    stock_data['Average_previous_day'] = stock_data['Close'].rolling(window=WINDOW).mean().shift(SHIFT).bfill()
    X = stock_data['Average_previous_day'].to_frame()
    y = stock_data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=54)

    # We get intercept and coefficient of explanatory variable to use in formula for forecasting future prices.
    lm = LinearRegression(fit_intercept=True)
    stock_prediction_model = lm.fit(X_train, y_train)
    coef = stock_prediction_model.coef_
    intercept = stock_prediction_model.intercept_

    # We use X_test to get r-squred to see how well our model performs on untrained data
    test_predictions = stock_prediction_model.predict(X_test)
    r_squared = r2_score(y_test, test_predictions)

    # But we use all available data to create a new data frame that we will use for plotting.
    predictions = stock_prediction_model.predict(X)
    results_df = pd.DataFrame({'Actual': y, 'Predicted': predictions})

    return intercept, coef, results_df, r_squared


def lm_forecasting(linear_model):
    """
    Employs linear regression modeling to forecast future stock prices.
    """
    # We retrieve the current date and set our forecasting period.
    current_day = datetime.today()
    future_weeks = [current_day + timedelta(days=i) for i in range(RANGE)]
    future_weeks_str = [date.strftime('%m-%d') for date in future_weeks]

    # We retrieve the needed values from our regression model and define formula.
    current_price = linear_model[2]['Actual'].iloc[-1]
    intercept, coefficient = linear_model[:2]
    def formula():
        forecasted_price = intercept + coefficient * current_price
        return forecasted_price

    # We create a new data frame for our forecasted prices and corresponding dates.
    forecasted_stocks = [current_price := formula() for _ in range(RANGE)]
    lm_forecasted_week_df = pd.DataFrame({'Date': future_weeks_str, 'Forecasted Close Price': forecasted_stocks})

    return lm_forecasted_week_df


def calculate_daily_returns(stock_data):
    """
    Calculates daily returns from the stock's close prices.
    """
    stock_data['Daily_Return'] = stock_data['Close'].pct_change().dropna()
    return stock_data


def calculate_volatility(stock_data):
    """
    Calculates the annualized volatility based on daily returns.
    """
    return np.std(stock_data['Daily_Return']) * np.sqrt(252)  # Annualizing volatility


def geometric_brownian_motion(stock_data):
    """
    Employs Geometric Brownian Motion formula to forecast future stock prices.
    """
    # We retrieve the current date and set our forecasting period.
    current_day = datetime.today()
    future_week = [current_day + timedelta(days=i) for i in range(RANGE)]
    future_weeks_str = [date.strftime('%m-%d') for date in future_week]

    # Today's price of the stock
    S0 = stock_data['Close'].iloc[-1]
    # The drift rate:
    mu = stock_data["Daily_Return"].mean()
    # The volatility:
    sigma = calculate_volatility(stock_data)
    # Time increment
    dt = 1/365
    # Wiener process increment:
    W = np.random.normal(loc=0, scale=np.sqrt(dt))

    def formula():
        S1 = S0 * np.exp((mu - 0.5*sigma**2) * dt + sigma * W)
        return S1

    # We create a new data frame for our forecasted prices and corresponding dates.
    forecasted_S = [S0 := formula() for _ in range(RANGE)]
    forecasted_gbm_df = pd.DataFrame({'Date': future_weeks_str, 'Forecasted Close Price (GBM)': forecasted_S})
    return forecasted_gbm_df


# PLOTS:
def time_series_plot(stock_data, predicted):
    """
    Visually represents both actual and predicted data points plotted over time.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data.index, stock_data["Close"])
    plt.plot(stock_data.index, predicted)

    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Time Series Plot: actual & predicted stock prices (linear regression)')
    # plt.legend()
    plt.grid(True)

    plt.show()


def plot_actual_vs_predicted(actual, predicted):
    """
    Visualizes the overall trend of correlation between predicted and actual values.
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(actual, predicted, color="blue")

    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title("Scatter Plot: actual vs. predicted stock prices (linear regression)")
    plt.grid(True)

    plt.show()

def plot_act_vs_pred_pct_change(stock_data, actual, predicted):
    """
    Visually represents the percentage change between actual and predicted data points plotted over time.
    """
    plt.figure(figsize=(10, 5))
    percent_change = ((predicted - actual) / actual) * 100
    plt.plot(stock_data.index, percent_change)

    plt.xlabel('Date')
    plt.ylabel('Percentage change')
    plt.title("Time Series Plot: Percentage change between actual and predicted stock prices (linear regression)")
    plt.grid(True)

    plt.show()

def plot_lm_forecast(lm_forecasted_week_df):
    """
    Visualizes the results of linear model forecasting plotted over the specified date.
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(lm_forecasted_week_df['Date'], lm_forecasted_week_df["Forecasted Close Price"], color="purple")

    # We define the y-axis limits for a better visualization of our graph.
    margin = 0.3
    max_value = lm_forecasted_week_df['Forecasted Close Price'].max()
    upper_limit = max_value + margin * max_value
    plt.ylim(0, upper_limit)

    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title("Stock Prices Forecast: Linear Regression method")
    plt.grid(True)

    plt.show()


def plot_daily_returns(stock_data):
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data.index, stock_data['Daily_Return'])

    plt.xlabel('Date')
    plt.ylabel('Daily Returns')
    plt.title("Time Series Plot: Daily Returns")
    plt.grid(True)

    plt.show()


def plot_gbm(gbm_forecast):
    """
    Visualizes the results of Geometric Brownian Motion model plotted over the specified date.
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(gbm_forecast['Date'], gbm_forecast["Forecasted Close Price (GBM)"], color="green")

    # We define the y-axis limits for a better visualization of our graph.
    margin = 0.3
    max_value = gbm_forecast['Forecasted Close Price (GBM)'].max()
    upper_limit = max_value + margin * max_value
    plt.ylim(0, upper_limit)

    plt.xlabel('Date')
    plt.ylabel('Close Price (GBM)')
    plt.title("Stock Prices Forecast: Geometric Brownian Motion method")
    plt.grid(True)

    plt.show()

def main():
    """
    Main function to run the enhanced project workflow.
    """
    # You may enter a code of any interested publicly traded company.
    ticker = input("Please, enter a stock ticker symbol: ")

    # We retrieve the historical data for a corresponding ticker symbol.
    stock_data = historical_data(ticker)
    print(stock_data)

    # We run a regression model.
    linear_model = perform_linear_regression(stock_data)
    r_squared = round(linear_model[3], 3)
    print(f"Our r-squared is {r_squared}. In other words, our the model explains {r_squared*100}% of fluctations.")
    print(linear_model[2])

    # We also retrieve the following values to use in different functions.
    actual = linear_model[2]['Actual']
    predicted = linear_model[2]['Predicted']

    # We visualize the main correlations and trends associated with the regression model.
    time_series_plot(stock_data, predicted)
    plot_actual_vs_predicted(actual, predicted)
    plot_act_vs_pred_pct_change(stock_data, actual, predicted)

    # Using our regression model, we make a forecast for future stock prices and visualize the results.
    lm_forecast = lm_forecasting(linear_model)
    print(lm_forecast)
    plot_lm_forecast(lm_forecast)

    # We retrieve daily returns and visualize the results.
    stock_data = calculate_daily_returns(stock_data)
    plot_daily_returns(stock_data)

    # We make another forecast, using Geometric Brownian Motion model.
    gbm_forecast = geometric_brownian_motion(stock_data)
    print(gbm_forecast)
    plot_gbm(gbm_forecast)


if __name__ == "__main__":
    main()




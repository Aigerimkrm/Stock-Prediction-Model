# Stock-Prediction-Model

This Python script is designed for stock price prediction and analysis, utilizing various statistical and machine learning techniques. It leverages libraries such as NumPy, pandas, yfinance, matplotlib, and scikit-learn to access historical stock data, perform linear regression, calculate volatility, and forecast future stock prices using both linear regression and Geometric Brownian Motion (GBM).


## Features

- **Historical Stock Data Retrieval**: Accesses Yahoo! Finance's historical data for specified stock tickers.
- **Linear Regression Model**: Predicts stock prices using a simple linear regression model and evaluates its performance with the R-squared metric.
- **Stock Price Forecasting**: Employs linear regression and Geometric Brownian Motion to forecast future stock prices.
- **Volatility Calculation**: Calculates the annualized volatility of stock prices based on daily returns.
- **Data Visualization**: Plots historical stock prices, linear regression predictions, percentage changes, forecasted prices, and daily returns for comprehensive analysis.

## Dependencies

- Python 3
- NumPy
- pandas
- yfinance
- matplotlib
- scikit-learn

Necessary dependencies could be installed using pip:

```bash
pip install numpy pandas yfinance matplotlib sklearn
```

## Usage
- **1. Run the script in your Python environment**: Please make sure you have necessary dependencies
- **2. Enter your stock TICKER**: When entered you will obtain analysis of the time series of daily returns.

## Functions Overview

- **historical_data(ticker)**: Retrieves historical stock data for the specified ticker.
- **perform_linear_regression(stock_data)**: Computes and evaluates a linear regression model for predicting stock prices.
- **lm_forecasting(linear_model)**: Forecasts future stock prices based on the linear regression model.
- **calculate_daily_returns(stock_data)**: Calculates the daily returns from the stock's close prices.
- **calculate_volatility(stock_data)**: Calculates the annualized volatility of the stock.
- **geometric_brownian_motion(stock_data)**: Forecasts future stock prices using the Geometric Brownian Motion formula.
- **Visualization functions**: Various functions to plot the actual vs. predicted prices, daily returns, and forecasts.

## Authors
Aigerim Zhumagulova & Dominik Plaček

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

Necessery dependencies could be installed using pip:

```bash
pip install numpy pandas yfinance matplotlib sklearn

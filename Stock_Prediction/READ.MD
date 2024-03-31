Stock Price Prediction using NeuralProphet

This repository contains Python code for predicting stock prices and trends using the NeuralProphet library. The code is written in Python 3 and can be executed in a Jupyter Notebook environment.

Dependencies
numpy: Fundamental package for scientific computing with Python.
pandas: Data manipulation and analysis library.
matplotlib: Plotting library for creating static, interactive, and animated visualizations in Python.
yfinance: Python library to access historical market data from Yahoo Finance.
NeuralProphet: Neural network-based time series forecasting library built on top of PyTorch.
Usage
Input: Users input the stock symbol, start date, and end date when prompted.
Data Retrieval: The program fetches historical stock data from Yahoo Finance using the provided inputs.
Data Preparation and Visualization: The fetched data is preprocessed and visualized to understand the historical trends in the stock price.
Model Training: The NeuralProphet model is trained using the historical stock data.
Prediction: The model makes predictions for future stock prices and trends.
Visualization: Predictions and actual stock prices are visualized to compare the model's performance.
Component Analysis: The model's components, such as trend and seasonality, are visualized to understand their impact on predictions.
How to Run
Clone the repository to your local machine.
Install the required dependencies mentioned in the requirements.txt file.
Open the Jupyter Notebook containing the code.
Follow the instructions within the Notebook to input the stock symbol and date range.
Run each cell in the Notebook sequentially to execute the code step by step.
Note
Ensure that you have a stable internet connection to fetch the stock data from Yahoo Finance.
The accuracy of the predictions may vary based on the selected stock and the historical data available.
Adjust the parameters and hyperparameters of the NeuralProphet model as needed to improve prediction performance.
Feel free to contribute to the repository by enhancing the code or adding new features.

# StockPredictions
A time series forecasting project that uses a Long Short-Term Memory (LSTM) neural network to predict stock prices. This script is built with Python, TensorFlow/Keras, and uses the `yfinance` library to fetch historical stock data.

## Features

-   **Time Series Forecasting:** Implements an LSTM model, a type of Recurrent Neural Network (RNN) well-suited for sequence prediction.
-   **Live Data Fetching:** Downloads historical stock data for any valid ticker symbol from Yahoo Finance.
-   **Data Preprocessing:** Scales the data using `MinMaxScaler` for optimal model performance.
-   **Dynamic Training/Testing Split:** Splits the data into training and testing sets to evaluate the model on unseen data.
-   **Result Visualization:** Plots the actual stock prices against the model's predictions for a clear visual comparison of its performance.

## Technology Stack

-   **Python**
-   **TensorFlow / Keras:** For building and training the LSTM neural network.
-   **yfinance:** To download historical market data from Yahoo Finance.
-   **scikit-learn:** For data normalization.
-   **NumPy:** For numerical operations.
-   **Matplotlib:** For plotting the results.
-   **Pandas:** For data manipulation.

## Setup and Usage

It's highly recommended to use a virtual environment for this project.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Anjali-Ajesh/stock-price-predictor.git](https://github.com/Anjali-Ajesh/stock-price-predictor.git)
    cd stock-price-predictor
    ```

2.  **Install Dependencies:**
    ```bash
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install the required libraries
    pip install tensorflow yfinance scikit-learn numpy matplotlib pandas
    ```

3.  **Run the Predictor:**
    Execute the Python script from your terminal. You can change the stock ticker inside the script to predict for a different company.
    ```bash
    python stock_predictor.py
    ```
    The script will fetch the data, train the model (this may take a few minutes), and then display a plot with the prediction results.

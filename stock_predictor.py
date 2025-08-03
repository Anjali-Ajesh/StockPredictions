# stock_predictor.py

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import math

def create_and_run_model(ticker='AAPL', start_date='2012-01-01', end_date='2024-01-01'):
    """
    Fetches stock data, builds an LSTM model, trains it, and predicts stock prices.
    """
    # --- 1. Fetch and Prepare Data ---
    print(f"Fetching stock data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    dataset = data.values # Convert the dataframe to a numpy array
    training_data_len = math.ceil(len(dataset) * .8) # 80% of the data for training

    # --- 2. Scale the Data ---
    # Scaling is important for neural networks
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # --- 3. Create the Training Dataset ---
    train_data = scaled_data[0:training_data_len, :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    # Use the past 60 days of data to predict the 61st day's price
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data to be 3D for the LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # --- 4. Build the LSTM Model ---
    print("Building the LSTM model...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    # --- 5. Compile and Train the Model ---
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Training the model... (This may take a moment)")
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # --- 6. Create the Testing Dataset ---
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :] # Actual values for testing

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        
    # Convert the data to a numpy array and reshape
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # --- 7. Get the Model's Predicted Price Values ---
    print("Making predictions...")
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions) # Un-scale the values

    # --- 8. Visualize the Data ---
    print("Plotting the results...")
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(16,8))
    plt.title(f'Model - {ticker} Stock Price Prediction')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val (Actual)', 'Predictions'], loc='lower right')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # You can change the ticker to any valid stock symbol
    create_and_run_model(ticker='GOOGL')

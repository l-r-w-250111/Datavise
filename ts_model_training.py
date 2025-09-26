import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet

def create_sequences(data, sequence_length):
    """
    Creates sequences from time series data.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def build_lstm_model(sequence_length, lstm_units):
    """
    Builds and compiles a simple LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=lstm_units, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_predict_lstm(df: pd.DataFrame, date_col: str, value_col: str, sequence_length: int, lstm_units: int, epochs: int, batch_size: int):
    """
    Preprocesses data, trains an LSTM model, and returns predictions.
    """
    try:
        # 1. Prepare and scale data
        data = df.filter([value_col])
        dataset = data.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # 2. Split data (80% train, 20% test)
        training_data_len = int(np.ceil(len(dataset) * .8))
        train_data = scaled_data[0:int(training_data_len), :]

        # 3. Create training sequences
        X_train, y_train = create_sequences(train_data, sequence_length)
        if X_train.shape[0] == 0:
            return None, None, None, "Error: Not enough training data to create sequences with the given length. Try a smaller sequence length."
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # 4. Build and train model
        model = build_lstm_model(sequence_length, lstm_units)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

        # 5. Create test sequences
        test_data = scaled_data[training_data_len - sequence_length:, :]
        X_test, y_test = create_sequences(test_data, sequence_length)
        if X_test.shape[0] == 0:
            return None, None, None, "Error: Not enough test data to create sequences. The dataset might be too small."
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # 6. Make predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # 7. Prepare results for plotting
        train_df = data[:training_data_len]
        valid_df = data[training_data_len:]
        valid_df['Predictions'] = predictions

        return valid_df, model, scaler, None

    except Exception as e:
        return None, None, None, f"An error occurred during LSTM model training: {e}"

def train_and_predict_prophet(df: pd.DataFrame, date_col: str, value_col: str, periods_to_forecast: int, regressors: list = None, future_regressor_values: dict = None):
    """
    Trains a Prophet model, optionally with external regressors, and returns a forecast.
    """
    try:
        # 1. Prepare data for Prophet
        columns_to_keep = [date_col, value_col]
        if regressors:
            columns_to_keep.extend(regressors)

        prophet_df = df[columns_to_keep].rename(columns={
            date_col: 'ds',
            value_col: 'y'
        })

        # Clean data: remove rows with NaN values that cause Prophet to fail
        prophet_df.dropna(subset=['ds', 'y'], inplace=True)

        if len(prophet_df) < 2:
            return None, None, "Error: Not enough data points to train Prophet model after removing missing values."

        # 2. Instantiate and train the model
        model = Prophet()

        # Add regressors if they are provided
        if regressors:
            for regressor in regressors:
                model.add_regressor(regressor)

        model.fit(prophet_df)

        # 3. Create a future dataframe and make predictions
        future = model.make_future_dataframe(periods=periods_to_forecast)

        # Add regressor values to the future dataframe
        if regressors and future_regressor_values:
            # Add historical regressor values
            for regressor in regressors:
                future = future.join(prophet_df[['ds', regressor]].set_index('ds'), on='ds')

            # Fill future regressor values with user input
            future_dates = future['ds'] > prophet_df['ds'].max()
            for regressor, value in future_regressor_values.items():
                future.loc[future_dates, regressor] = value

        forecast = model.predict(future)

        # 4. Return the original data, the forecast, and the model
        return model, forecast, None

    except Exception as e:
        return None, None, f"An error occurred during Prophet model training: {e}"

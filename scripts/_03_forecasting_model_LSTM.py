# _03_forecasting_model_LSTM.py

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesForecastingLSTM:
    """
    A class for time series forecasting of stock prices using LSTM models.
    """

    def __init__(self, stock_name, processed_dir, plot_dir, model_dir):
        """
        Initializes the TimeSeriesForecastingLSTM class.

        Args:
            stock_name (str): The name of the stock.
            processed_dir (str): Directory for processed data.
            plot_dir (str): Directory for saving plots.
            model_dir (str): Directory for saving the trained model.
        """
        self.stock_name = stock_name
        self.processed_dir = processed_dir
        self.plot_dir = plot_dir
        self.model_dir = model_dir
        self.lstm_results = {}
        self.df = None

        # Create output directories if they do not exist
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        print("🧪 Running full forecasting pipeline...\n")
        self.load_data()

    def safe_relpath(self, path, start=None):
        """
        Safely returns a relative path, falling back to absolute path if necessary.

        Args:
            path (str): The path to make relative.
            start (str, optional): The starting directory for the relative path.
            Defaults to None.

        Returns:
            str: The relative or absolute path.
        """
        try:
            return os.path.relpath(path, start)
        except ValueError:
            # Fallback to absolute path if on different drives
            return path

    def load_data(self):
        """
        Loads and preprocesses the stock data from a CSV file.

        Returns:
            pandas.DataFrame: The loaded and processed DataFrame.
        """
        try:
            self.df = pd.read_csv(self.processed_dir)
            if self.df.empty:
                print(f"⚠️ Warning: No data returned for {self.stock_name}")
            else:
                self.df = self.df.iloc[2:]
                self.df = self.df.rename(columns={"Price": "Date"})
                self.df["Date"] = pd.to_datetime(self.df["Date"])
                # Correct data types
                dict_col = {
                    "Date": "datetime",
                    "Close": "float",
                    "High": "float",
                    "Low": "float",
                    "Open": "float",
                    "Volume": "int",
                    "Trend": "float",
                    "Volatility": "float",
                    "Return": "float",
                }

                for col, dtype in dict_col.items():
                    if col in self.df.columns:
                        if dtype == "datetime":
                            self.df[col] = pd.to_datetime(self.df[col])
                        else:
                            self.df[col] = self.df[col].astype(dtype)

                print("Datatypes changed.")
                print("🔹DataFrame Head:")
                display(self.df.head())
                print(f"\n🔹 Shape: {self.df.shape}")
                print(f"\n🔹 Columns: {list(self.df.columns)}")
                print("\n🔹 DataFrame Info:")
                self.df.info()
                print("\n")

        except Exception as e:
            print(f"⚠️ Stock price missing. Load data first for {self.stock_name}: {e}")
        return self.df

    def create_dataset(self, dataset, look_back):
        """
        Creates input and output datasets for the LSTM model.

        Args:
            dataset (numpy.ndarray): The input time series data.
            look_back (int): The number of previous time steps to use as input features.

        Returns:
            tuple: A tuple containing the input (X) and output (Y) datasets.
        """
        X, Y = [], []  # previous lookback time step, value to be predicted
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i : (i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    def train_lstm(self):
        """
        Trains the LSTM model for time series forecasting.

        Returns:
            dict: A dictionary containing the LSTM prediction results.
        """
        # LSTM Forecasting

        # Iterate through each stock in df
        stock_data = self.df["Return"].dropna()

        # Split data into training and testing sets
        # Before MinMax fiting to avoid data bleeding
        size = int(len(stock_data) * 0.8)  # 80/20 data split
        train_data, test_data = stock_data[:size], stock_data[size:]

        # Create and fit a MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(
            train_data.values.reshape(-1, 1)
        )  # transform and fit
        scaled_data_test = scaler.transform(
            test_data.values.reshape(-1, 1)
        )  # transform only (avoids data bleeding)

        # Function call
        look_back = 180  # 180 days look back
        X_train, Y_train = self.create_dataset(scaled_data, look_back)
        X_test, Y_test = self.create_dataset(scaled_data_test, look_back)

        # Reshape input to be [samples, look_back, feature]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Create the LSTM model and compile the model
        model = Sequential()
        model.add(Input(shape=(look_back, X_train.shape[2])))
        model.add(LSTM(units=64, return_sequences=True))  # First Layer
        model.add(LSTM(units=16))  # Second layer
        model.add(Dropout(0.2))  # To avoid overfitting
        model.add(Dense(1))  # Add 1 unit dense layer
        model.compile(loss="mean_squared_error", optimizer="adam")

        # Train and assign the model
        model.fit(X_train, Y_train, epochs=100, batch_size=64, shuffle=False, verbose=0)
        self.model = model

        # Make predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(
            predictions
        )  # Invert predictions to original scale
        Y_test = scaler.inverse_transform(
            Y_test.reshape(-1, 1)
        )  # Invert Y_test to original scale

        train_predictions = model.predict(X_train)
        train_predictions = scaler.inverse_transform(
            train_predictions
        )  # Invert predictions to original scale
        Y_train = scaler.inverse_transform(
            Y_train.reshape(-1, 1)
        )  # Invert Y_test to original scale

        # Evaluate the model
        print(f"LSTM forecast model evaluation for {self.stock_name} stock")

        # Calculate Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(Y_test, predictions))
        print(f"Test RMSE : {rmse:.5f}")  # rounded to five decimal place

        # Calculate Mean Absolute Error
        mae = mean_absolute_error(Y_test, predictions)
        print(f"Test MAE for : {mae:.5f}")

        # Calcualte R square
        r2 = r2_score(Y_test, predictions)
        print(f"Test R-squared for : {r2:.2f}")  # rounded to two decimal place

        Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()

        # Store predictions and actual values for plotting and model for finetuning
        self.lstm_results = {
            "predictions": predictions.flatten(),
            "actual": Y_test,
            "X_train": X_train,
            "X_test": X_test,
        }

        return self.lstm_results

    def save_lstm_model(self):
        """
        Saves the trained LSTM model to a file.
        """
        # Save model
        model_path = os.path.join(self.model_dir, f"{self.stock_name}_lstm_model.pkl")
        joblib.dump(self.model, model_path)
        print(
            f"\n💾 {self.stock_name} LSTM model saved to \
                {self.safe_relpath(model_path)}"
        )

    def plot_lstm(self):
        """
        Generates and saves plots of the actual and predicted stock returns
        and closing prices.
        """
        # Get data from the above cell
        stock_data = self.df["Return"].dropna()

        test = self.lstm_results["actual"]
        predictions = self.lstm_results["predictions"]
        X_train = self.lstm_results["X_train"]
        X_test = self.lstm_results["X_test"]

        test_dates = stock_data.index[len(X_train) : len(X_train) + len(X_test)]

        # Return prediction plot
        plt.figure(figsize=(12, 4))
        plt.plot(test_dates, test, label="Actual Return", color="Green")
        plt.plot(
            test_dates,
            predictions,
            label="Predicted Return",
            color="Red",
            linestyle="--",
        )

        plt.title(f"{self.stock_name} - Log Return Stock Price Prediction with LSTM")
        plt.tick_params(axis="x", rotation=0)
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir,
                f"{self.stock_name}_stock_price_return_prediction_lstm.png",
            )
            plt.savefig(plot_path)
            print(f"\n💾 Plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

        # Closing price prediction plot
        # Get the last known closing price before test period
        last_train_index = len(self.lstm_results["X_train"]) - 1
        last_close_price = self.df["Close"].iloc[last_train_index]

        # Reconstruct predicted closing prices
        predicted_close_price = last_close_price * np.exp(np.cumsum(predictions))

        # Reconstruct actual closing prices from log returns
        actual_close_price = last_close_price * np.exp(np.cumsum(test))

        plt.figure(figsize=(12, 4))
        plt.plot(test_dates, actual_close_price, label="Actual Price", color="Green")
        plt.plot(
            test_dates,
            predicted_close_price,
            label="Predicted Price",
            color="Red",
            linestyle="--",
        )

        plt.title(
            f"{self.stock_name} - Closing Stock Price Prediction with LSTM \
            (Reconstructed from Log Returns)"
        )
        plt.tick_params(axis="x", rotation=0)
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir,
                f"{self.stock_name}_closing_stock_price_prediction_lstm.png",
            )
            plt.savefig(plot_path)
            print(f"\n💾 Plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    def forecastor_model(self):
        """
        Runs the complete forecasting pipeline, including training,
        saving the model, and plotting.
        """
        self.train_lstm()
        self.save_lstm_model()
        self.plot_lstm()

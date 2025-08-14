# _05_forecast_trends

import os
import random

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ForecastTrend:
    """
    A class for time series forecasting of stock prices using LSTM models.

    Attributes:
        asset_name (str): The name of the stock.
        processed_path (str): Path to the processed data CSV file.
        processed_dir (str): Directory for processed data.
        plot_dir (str): Directory for saving plots.
        model_path (str): Path to the fine-tuned LSTM model file.
        look_back (int): Number of previous time steps to use as input for forecasting.
        steps (int): Number of future time steps to forecast.
        lstm_results (dict): Dictionary to store LSTM forecasting results.
        df (pd.DataFrame): DataFrame containing the loaded and processed data.
        scalers (dict): Dictionary of scalers for different features.
        model (keras.Model): The loaded LSTM model.
        predictions (list): List of scaled predictions.
        forecast_dates (pd.DatetimeIndex): Datetime index for the forecast period.
        predicted_close (np.ndarray): Inverse transformed predicted closing prices.
        prediction_df (pd.DataFrame): DataFrame containing the forecast results.
    """

    def __init__(
        self,
        asset_name,
        processed_path,
        processed_dir,
        plot_dir,
        model_path,
        look_back=30,
        steps=126,
    ):
        """
        Initialises the ForecastTrend class.

        Args:
            asset_name (str): The name of the stock.
            processed_path (str): Path to the processed data CSV file.
            processed_dir (str): Directory for processed data.
            plot_dir (str): Directory for saving plots.
            model_path (str): Path to the fine-tuned LSTM model file.
            look_back (int, optional): Number of previous time steps to use as input.
                                        Defaults to 30.
            steps (int, optional): Number of future time steps to forecast.
                                    Defaults to 126.
        """
        self.asset_name = asset_name
        self.processed_path = processed_path
        self.plot_dir = plot_dir
        self.processed_dir = processed_dir
        self.model_path = model_path
        self.look_back = look_back
        self.steps = steps  # Number of steps to forecast 126/6mo or 256/12mo
        self.lstm_results = {}
        self.df = None
        self.scalers = {
            "Log Close": MinMaxScaler(),
            "Daily Return": StandardScaler(),
            "Volatility": StandardScaler(),
            "Trend": MinMaxScaler(),
        }
        self.model = None
        self.predictions = None
        self.forecast_dates = None
        self.predicted_close = None
        self.prediction_df = None

        # Create output directories if they do not exist
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        print("🧪 Running full forecasting pipeline...\n")
        self.set_seed(42)
        self.load_data()

    def set_seed(self, seed=42):
        """
        Sets random seed for reproducibility across NumPy, random, and TensorFlow.

        Args:
            seed (int, optional): The seed value. Defaults to 42.
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

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
            self.df = pd.read_csv(self.processed_path)
            if self.df.empty:
                print(f"⚠️ Warning: No data returned for {self.asset_name}")
            else:
                self.df = self.df.iloc[2:]
                self.df = self.df.rename(columns={"Price": "Date"})
                self.df["Date"] = pd.to_datetime(self.df["Date"])
                self.df = self.df.sort_values(by="Date").reset_index(drop=True)
                self.df = self.df.drop(columns=["Unnamed: 0"], errors="ignore")
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
                    "Daily Return": "float",
                    "Log Close": "float",
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
            print(f"⚠️ Stock price missing. Load data first for {self.asset_name}: {e}")
        return self.df

    def load_finetuned_model(self):
        """
        Loads the fine-tuned LSTM model from disk.

        Returns:
            keras.Model: The loaded Keras model.
        """

        model = keras.models.load_model(self.model_path)
        print(f"\n📥 Loaded LSTM model from {self.safe_relpath(self.model_path)}")
        self.model = model
        return self.model

    def scale_data(self):
        """
        Scales the endogenous and exogenous data using appropriate scalers.

        Returns:
            tuple: A tuple containing the scalers and the scaled data arrays
                    (scalers, scaled_data, scaled_return_exo,
                    scaled_vol_exo, scaled_trend_exo).
        """

        asset_data = self.df["Log Close"].iloc[-self.look_back :]
        return_exo_data = self.df["Daily Return"].iloc[-self.steps :]
        vol_exo_data = self.df["Volatility"].iloc[-self.steps :]
        trend_exo_data = self.df["Trend"].iloc[-self.steps :]

        # Separately fit asset datas to avoid data bleeding
        # Create and fit a MinMaxScaler and StandardScaler for endo and exo features

        # Close (endo)
        self.scaled_data = self.scalers["Log Close"].fit_transform(
            asset_data.values.reshape(-1, 1)
        )

        # Daily Return (exo)
        self.scaled_return_exo = self.scalers["Daily Return"].fit_transform(
            return_exo_data.values.reshape(-1, 1)
        )

        # Volatility (exo)
        self.scaled_vol_exo = self.scalers["Volatility"].fit_transform(
            vol_exo_data.values.reshape(-1, 1)
        )

        # Trends (exo)
        self.scaled_trend_exo = self.scalers["Trend"].fit_transform(
            trend_exo_data.values.reshape(-1, 1)
        )

        print("⚖️ Data scaling completed.")

        return (
            self.scalers,
            self.scaled_data,
            self.scaled_return_exo,
            self.scaled_vol_exo,
            self.scaled_trend_exo,
        )

    def recursive_forecast(self):
        """
        Performs recursive forecasting using the loaded LSTM model.

        Returns:
            list: A list of scaled predictions.
        """
        initial_sequence = self.scaled_data[-self.look_back :]  # shape: (look_back, 1)

        predictions = []
        current_seq = initial_sequence.copy()

        for i in range(self.steps):
            # Prepare endogenous input
            input_X = current_seq[-self.look_back :].reshape(self.look_back, 1)

            # Prepare exogenous inputs (repeat across time steps)
            input_return = np.repeat(
                self.scaled_return_exo[i].reshape(1, 1), self.look_back, axis=0
            )
            input_vol = np.repeat(
                self.scaled_vol_exo[i].reshape(1, 1), self.look_back, axis=0
            )
            input_trend = np.repeat(
                self.scaled_trend_exo[i].reshape(1, 1), self.look_back, axis=0
            )

            # Stack into multivariate input: shape (1, look_back, 4)
            multivariate_input = np.concatenate(
                [input_X, input_return, input_vol, input_trend], axis=1
            )
            multivariate_input = multivariate_input.reshape(1, self.look_back, 4)

            # Predict next value
            # pred = self.model.predict(multivariate_input)
            pred = self.model.predict(multivariate_input, verbose=0)
            predictions.append(pred[0][0])

            # Update sequence with new prediction
            current_seq = np.append(current_seq, pred[0][0]).reshape(-1, 1)

        self.predictions = predictions
        return self.predictions

    def create_forecast_df(self):
        """
        Creates a DataFrame containing the forecast results and saves it to a CSV file.

        Returns:
            tuple: A tuple containing the forecast DataFrame, forecast dates, and
                    predicted close prices
                    (prediction_df, forecast_dates, predicted_close).
        """
        # Create a DataFrame for predictions
        # Create forecast dates
        last_date = self.df["Date"].iloc[-1]
        self.forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=self.steps, freq="B"
        )

        # Inverse transform predictions to original scale
        true_preds = (
            self.scalers["Log Close"]
            .inverse_transform(np.array(self.predictions).reshape(-1, 1))
            .flatten()
        )
        self.predicted_close = np.exp(true_preds)

        prediction_df = pd.DataFrame(
            {
                "Date": self.forecast_dates,
                "Predicted Close": self.predicted_close.flatten(),
            }
        )

        prediction_df["Predicted Volatility"] = (
            prediction_df["Predicted Close"].rolling(window=30).std()
        )
        prediction_df["Predicted Return"] = prediction_df[
            "Predicted Close"
        ].pct_change()
        prediction_df["Predicted Trend"] = (
            prediction_df["Predicted Close"].rolling(window=30).mean()
        )

        # Print the DataFrame head
        display(prediction_df.head())

        output_path = os.path.join(
            self.processed_dir, f"{self.asset_name}_prediction_df.csv"
        )
        prediction_df.to_csv(output_path, index=True)
        print(f"\n💾 Forecast DataFrame saved to {self.safe_relpath(output_path)}")

        self.prediction_df = prediction_df
        return self.prediction_df, self.forecast_dates, self.predicted_close

    def plot_forecast(self):
        """
        Plots the recursive forecast against historical 'Log Close' values,
        including historical and forecasted trend, volatility, and daily return.
        """
        if self.predictions is None:
            print(f"⚠️ No predictions found for {self.asset_name}.")
            return

        # Get historical 'Log Close' for plotting
        historical_log_close = self.df["Log Close"].values

        historical_close = np.exp(historical_log_close)
        predicted_close = self.prediction_df["Predicted Close"]
        predicted_vol = self.prediction_df["Predicted Volatility"]
        predicted_return = self.prediction_df["Predicted Return"]
        predicted_trend = self.prediction_df["Predicted Trend"]

        # Plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        ax1.plot(self.df["Date"], historical_close, label="Historical Close")
        ax1.plot(self.df["Date"], self.df["Trend"], label="Historical Trend")
        ax1.plot(
            self.forecast_dates,
            predicted_close,
            label="Forecasted Close",
            color="Green",
            linestyle="--",
        )
        ax1.plot(
            self.forecast_dates,
            predicted_trend,
            label="Forecasted Trend",
            color="Yellow",
            linestyle="--",
        )
        ax1.set_title(
            f"{self.asset_name} - Recursive Closing Price, Trend, \
                Return and Volatility Forecast ({self.steps} steps)"
        )
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Close Price")
        ax1.legend()
        ax1.grid(True)

        # Plot Volatility
        ax2.plot(
            self.df["Date"],
            self.df["Volatility"],
            label="Volatility (Rolling Std)",
            color="Red",
        )
        ax2.plot(
            self.forecast_dates,
            predicted_vol,
            label="Forecasted Volatility",
            color="Green",
            linestyle="--",
        )
        ax2.set_ylabel("Volatility")
        ax2.set_xlabel("Date")
        ax2.legend(loc="upper left")
        ax2.grid(True)

        # Plot Return
        ax3.plot(
            self.df["Date"],
            self.df["Daily Return"],
            label="Daily Return",
            color="Purple",
        )
        ax3.plot(
            self.forecast_dates,
            predicted_return,
            label="Forecasted Return",
            color="Green",
            linestyle="--",
        )
        ax3.set_ylabel("Daily Return")
        ax3.set_xlabel("Date")
        ax3.legend(loc="upper left")
        ax3.grid(True)

        # Adjust layout and show plot
        plt.tight_layout()
        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir, f"{self.asset_name}_close_trend_vol_return_forecast.png"
            )
            plt.savefig(plot_path)
            print(f"\n💾 Forecast plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    def trend_forecaster(self):
        """
        Runs the full forecasting pipeline:
        prepares inputs, scales data, loads model, forecasts, plots, and saves.
        """
        self.scale_data()
        self.load_finetuned_model()
        self.recursive_forecast()
        self.create_forecast_df()
        self.plot_forecast()

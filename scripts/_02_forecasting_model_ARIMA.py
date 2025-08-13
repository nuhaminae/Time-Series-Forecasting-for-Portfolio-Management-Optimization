# _02_forecasting_model_ARIMA.py

import os
from math import sqrt

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA


class TimeSeriesForecastingARIMA:
    """
    A class for time series forecasting of stock prices using ARIMA models.
    """

    def __init__(self, stock_name, processed_dir, plot_dir, model_dir):
        """
        Initialises the TimeSeriesForecasting class.

        Args:
            stock_name (str): The name of the stock.
            processed_dir (str): The directory containing the processed data.
            plot_dir (str): The directory to save the plots.
            model_dir (str): The directory to save the trained models.
        """
        self.stock_name = stock_name
        self.processed_dir = processed_dir
        self.plot_dir = plot_dir
        self.model_dir = model_dir
        self.arima_results = {}
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
        Returns a relative path if possible, otherwise returns the absolute path.

        Args:
            path (str): The path to make relative.
            start (str, optional): The directory to make the path relative to.
                                    Defaults to None.

        Returns:
            str: The relative or absolute path.
        """
        try:
            return os.path.relpath(path, start)
        except ValueError:
            # Fallback to absolute path if on different drives
            return path

    # ----- Extract historical financial data using YFinance -----#
    def load_data(self):
        """
        Loads the processed stock data from a CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        try:
            self.df = pd.read_csv(self.processed_dir)
            if self.df.empty:
                print(f"⚠️ Warning: No data returned for {self.stock_name}")
            else:
                self.df = self.df.iloc[2:]
                self.df = self.df.rename(columns={"Price": "Date"})
                self.df["Date"] = pd.to_datetime(self.df["Date"])
                self.df.set_index("Date", inplace=True)

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

        except Exception as e:
            print(f"⚠️ Stock price missing. Load data first for {self.stock_name}: {e}")
        return self.df

    # ----- Define ARIMA model ----- #
    def fit_arima(self, series):
        """
        Fits an ARIMA model to the given time series data using auto_arima.

        Args:
            series (pd.Series): The time series data to fit the model on.

        Returns:
            pmdarima.arima.arima.ARIMA: The fitted ARIMA model.
        """
        model = auto_arima(
            series,
            max_p=2,
            max_d=2,
            max_q=2,
            trace=False,  # To increase execution speed
            error_action="ignore",
            suppress_warnings=True,
            random_state=42,
        )
        return model.fit(series)

    # ----- Train ARIMA model ----- #
    def train_arima(self):
        """
        Trains ARIMA model using walk-forward validation and evaluates its performance.

        Returns:
            dict: A dictionary containing training data, testing data, and predictions.
        """
        # data
        data = self.df["Daily Return"].dropna()

        # Split data into train and test sets
        size = int(len(data) * 0.8)
        train, test = data[:size], data[size:]

        history = [x for x in train]
        predictions = []

        # Fit the ARIMA model
        fitted_model = self.fit_arima(train)
        print(fitted_model.summary())
        print("\n")

        # Walk-forward validation
        for t in range(len(test)):
            model = ARIMA(history, order=fitted_model.order)
            model_fit = model.fit(
                method_kwargs={"warn_convergence": False}, low_memory=True
            )
            output = model_fit.forecast()

            # Generate a prediction
            yhat = output[0]
            predictions.append(yhat)

            # Add the actual observation to the history
            obs = test.iloc[t]
            history.append(obs)

        # Evaluate the model
        print("ARIMA rolling forecast model evaluation ...")

        rmse = sqrt(mean_squared_error(test, predictions))
        print(f"RMSE: {rmse:.5f}")

        mae = mean_absolute_error(test, predictions)
        print(f"MAE: {mae:.5f}")

        r2 = r2_score(test, predictions)
        print(f"R-squared: {r2:.5f}")

        # Store results
        self.arima_results = {"train": train, "test": test, "predictions": predictions}

        # Save model
        model_path = os.path.join(self.model_dir, f"{self.stock_name}_arima_model.pkl")
        auto_model_path = os.path.join(
            self.model_dir, f"{self.stock_name}_arima_object.pkl"
        )
        joblib.dump(fitted_model, auto_model_path)
        joblib.dump(model_fit, model_path)
        print(
            f"\n💾 {self.stock_name} ARIMA model saved to \
                {self.safe_relpath(model_path)}"
        )
        print(
            f"\n💾 {self.stock_name} ARIMA object saved to \
                {self.safe_relpath(auto_model_path)}"
        )

        return self.arima_results

    # ----- Plot ----- #
    def plot_arima(self):
        """
        Plots the actual and predicted stock returns and closing prices.
        """
        # Get data from the above cell
        train = self.arima_results["train"]
        test = self.arima_results["test"]
        predictions = self.arima_results["predictions"]

        # Daily Return prediction plot
        plt.figure(figsize=(12, 4))
        plt.plot(train.index, train.values, label="Training Data", color="Blue")
        plt.plot(test.index, test.values, label="Actual Daily Return", color="Green")
        plt.plot(
            test.index,
            predictions,
            label="Predicted Daily Return",
            color="Red",
            linestyle="--",
        )

        plt.title(f"{self.stock_name} - Stock Price Daily Return Prediction with ARIMA")
        plt.tick_params(axis="x", rotation=0)
        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir,
                f"{self.stock_name}_stock_price_prediction_arima.png",
            )
            plt.savefig(plot_path)
            print(f"\n💾 Plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    # ----- Run full pipeline ----- #
    def forecastor_model(self):
        """
        Runs the full ARIMA forecasting pipeline, including training and plotting.
        """
        self.train_arima()
        self.plot_arima()

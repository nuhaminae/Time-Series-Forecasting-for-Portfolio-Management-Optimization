# _04_finetune_LSTM.py

import json
import os
import random

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Model
from keras_tuner.tuners import RandomSearch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FineTuningLSTM:
    """
    A class for time series forecasting of asset prices using LSTM models
    with hyperparameter tuning using Keras Tuner.
    """

    def __init__(
        self,
        asset_name,
        processed_path,
        processed_dir,
        plot_dir,
        model_dir,
        look_back=30,
        project_name="lstm_tuning",
        max_trials=3,
        executions_per_trial=3,
        epochs=50,
        batch_size=64,
    ):
        """
        Initialises the FineTuningLSTM class with configuration parameters.

        Args:
            asset_name (str): The name of the asset being forecasted.
            processed_path (str): Path to the processed data file (CSV).
            processed_dir (str): Directory to save processed data artifacts.
            plot_dir (str): Directory for saving plots.
            model_dir (str): Directory for saving the trained model.
            look_back (int, optional): The number of previous time steps to use as input
                                    features for the LSTM. Defaults to 30.
            project_name (str, optional): The name of the Keras Tuner project.
                                        Defaults to "lstm_tuning".
            max_trials (int, optional): The maximum number of hyperparameter
                                        combinations to try during tuning.
                                        Defaults to 3.
            executions_per_trial (int, optional): The number of times to train each
                                                model variation during tuning.
                                                Defaults to 3.
            epochs (int, optional): The number of training epochs for the best model.
                                    Defaults to 50.
            batch_size (int, optional): The batch size for training the model.
                                        Defaults to 64.
        """
        self.asset_name = asset_name
        self.processed_dir = processed_dir
        self.processed_path = processed_path
        self.plot_dir = plot_dir
        self.model_dir = model_dir
        self.look_back = look_back
        self.project_name = project_name
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_results = {}
        self.df = None
        self.scalers = {
            "Log Close": MinMaxScaler(),
            "Daily Return": StandardScaler(),
            "Volatility": StandardScaler(),
            "Trend": MinMaxScaler(),
        }

        self.train_data = None
        self.test_data = None
        self.train_return_exo_data = None
        self.test_return_exo_data = None
        self.train_vol_exo_data = None
        self.test_vol_exo_data = None
        self.train_trend_exo_data = None
        self.test_trend_exo_data = None
        self.scaler = None
        self.scaled_data = None
        self.scaled_return_exo = None
        self.scaled_vol_exo = None
        self.scaled_trend_exo = None
        self.scaled_test_data = None
        self.scaled_test_return_exo = None
        self.scaled_test_vol_exo = None
        self.scaled_test_trend_exo = None

        # Create output directories if they do not exist
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # print("🧪 Running full finetuning pipeline...\n")
        print("🧪 Running full finetuning pipeline...\n", flush=True)

        self.set_seed(42)
        self.load_data()

    def set_seed(self, seed=42):
        """
        Sets random seed for reproducibility across NumPy, random, and TensorFlow.

        Args:
            seed (int, optional): The seed value to use. Defaults to 42.
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
        Loads and preprocesses the asset data from a CSV file specified by
        `self.processed_path`. The function renames columns, converts data
        types, sorts by date, and adds a 'Log Close' column.

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
                print("\n")

        except Exception as e:
            print(f"⚠️ asset price missing. Load data first for {self.asset_name}: {e}")
        return self.df

    def split_data(self):
        """
        Splits the loaded data into training and testing sets for both the endogenous
        ('Log Close') and exogenous features ('Daily Return', 'Volatility', 'Trend').
        The split is 80% for training and 20% for testing. The enriched DataFrame
        with the 'Log Close' column is also saved to a CSV file.

        Returns:
            tuple: A tuple containing the full DataFrame, train/test splits for
                   'Log Close', 'Daily Return', 'Volatility', and 'Trend'.
        """
        self.df["Log Close"] = np.log(self.df["Close"])

        # Data
        asset_data = self.df["Log Close"]
        return_exo_data = self.df["Daily Return"]
        vol_exo_data = self.df["Volatility"]
        trend_exo_data = self.df["Trend"]

        # Split data first before scaling to avoid data bleeding
        size = int(len(asset_data) * 0.8)  # 80/20 split
        self.train_data, self.test_data = asset_data[:size], asset_data[size:]
        self.train_return_exo_data, self.test_return_exo_data = (
            return_exo_data[:size],
            return_exo_data[size:],
        )
        self.train_vol_exo_data, self.test_vol_exo_data = (
            vol_exo_data[:size],
            vol_exo_data[size:],
        )
        self.train_trend_exo_data, self.test_trend_exo_data = (
            trend_exo_data[:size],
            trend_exo_data[size:],
        )

        print("💎 New column 'Log Close' is computed.")

        output_path = os.path.join(
            self.processed_dir, f"{self.asset_name}_log_enriched.csv"
        )
        self.df.to_csv(output_path, index=True)
        print(f"\n💾 Enriched data saved to {self.safe_relpath(output_path)}")

        display(self.df.head())
        print("🪓 Data splitting completed.")
        return (
            self.df,
            self.train_data,
            self.test_data,
            self.train_return_exo_data,
            self.test_return_exo_data,
            self.train_vol_exo_data,
            self.test_vol_exo_data,
            self.train_trend_exo_data,
            self.test_trend_exo_data,
        )

    def scale_data(self):
        """
        Scales the training and testing data for the endogenous ('Log Close')
        and exogenous features ('Daily Return', 'Volatility', 'Trend') using
        appropriate scalers (MinMaxScaler for 'Log Close' and 'Trend',
        StandardScaler for 'Daily Return' and 'Volatility'). Scalers are fitted
        only on the training data to prevent data leakage.

        Returns:
            tuple: A tuple containing the dictionary of scalers and the scaled
                   train/test data for 'Log Close', 'Daily Return', 'Volatility',
                   and 'Trend'.
        """
        # Separately fit asset datas to avoid data bleeding
        # Create and fit a MinMaxScaler and StandardScaler for endo and exo features

        # Close (endo)
        self.scaled_data = self.scalers["Log Close"].fit_transform(
            self.train_data.values.reshape(-1, 1)
        )
        self.scaled_test_data = self.scalers["Log Close"].transform(
            self.test_data.values.reshape(-1, 1)
        )

        # Daily Return (exo)
        self.scaled_return_exo = self.scalers["Daily Return"].fit_transform(
            self.train_return_exo_data.values.reshape(-1, 1)
        )
        self.scaled_test_return_exo = self.scalers["Daily Return"].transform(
            self.test_return_exo_data.values.reshape(-1, 1)
        )

        # Volatility (exo)
        self.scaled_vol_exo = self.scalers["Volatility"].fit_transform(
            self.train_vol_exo_data.values.reshape(-1, 1)
        )
        self.scaled_test_vol_exo = self.scalers["Volatility"].transform(
            self.test_vol_exo_data.values.reshape(-1, 1)
        )

        # Trends (exo)
        self.scaled_trend_exo = self.scalers["Trend"].fit_transform(
            self.train_trend_exo_data.values.reshape(-1, 1)
        )
        self.scaled_test_trend_exo = self.scalers["Trend"].transform(
            self.test_trend_exo_data.values.reshape(-1, 1)
        )

        print("⚖️ Data scaling completed.")
        return (
            self.scalers,
            self.scaled_data,
            self.scaled_test_data,
            self.scaled_return_exo,
            self.scaled_test_return_exo,
            self.scaled_vol_exo,
            self.scaled_test_vol_exo,
            self.scaled_trend_exo,
            self.scaled_test_trend_exo,
        )

    def create_dataset(
        self,
        dataset,
        return_exo_dataset,
        vol_exo_dataset,
        trend_exo_dataset,
        look_back=1,
    ):
        """
        Creates input (X) and output (Y) datasets for the LSTM model by
        generating sequences with a specified `look_back` period.

        Args:
            dataset (numpy.ndarray): The endogenous time series data.
            return_exo_dataset (numpy.ndarray): The 'Daily Return' exogenous data.
            vol_exo_dataset (numpy.ndarray): The 'Volatility' exogenous data.
            trend_exo_dataset (numpy.ndarray): The 'Trend' exogenous data.
            look_back (int, optional): The number of previous time steps to use as
                                       input features. Defaults to 1.

        Returns:
            tuple: A tuple containing the input (X), output (Y), and corresponding
                   exogenous features (return_exo, vol_exo, trend_exo) datasets.
        """
        # Function that creates dataset with lookback period

        X, Y, return_exo, vol_exo, trend_exo = [], [], [], [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i : (i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
            return_exo.append(return_exo_dataset[i + look_back, 0])
            vol_exo.append(vol_exo_dataset[i + look_back, 0])
            trend_exo.append(trend_exo_dataset[i + look_back, 0])
        return (
            np.array(X),
            np.array(Y),
            np.array(return_exo),
            np.array(vol_exo),
            np.array(trend_exo),
        )

    def build_model(self, hp):
        """
        Builds a Keras LSTM model for hyperparameter tuning. This function is
        used by Keras Tuner to search for the best model architecture and
        hyperparameters.

        Args:
            hp (HyperParameters): Keras Tuner HyperParameters object.

        Returns:
            keras.Model: The compiled Keras LSTM model.
        """
        input_layer = Input(shape=(self.look_back, self.train_data.shape[2]))
        # First LSTM layer
        x = LSTM(
            units=hp.Int("units_1", min_value=32, max_value=128, step=8),
            return_sequences=True,
        )(input_layer)
        # Second LSTM layer
        x = LSTM(units=hp.Int("units_2", min_value=32, max_value=128, step=8))(x)
        # Dropout layer
        x = Dropout(rate=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1))(x)
        # Output layer
        output = Dense(1)(x)

        # Build model
        model = Model(inputs=input_layer, outputs=output)

        # Compile model
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        model.compile(
            loss="mean_squared_error",
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        )

        return model

    def prepare_multivariate_input(self, X, return_exo, vol_exo, trend_exo, look_back):
        """
        Aligns and concatenates endogenous ('Log Close') and exogenous features
        ('Daily Return', 'Volatility', 'Trend') to create the multivariate input
        for the LSTM model.

        Args:
            X (np.ndarray): Endogenous input of shape (samples, look_back, 1)
            return_exo (np.ndarray): Exogen. feature 'Daily Return' of shape (samples,)
            vol_exo (np.ndarray): Exogenous feature 'Volatility' of shape (samples,)
            trend_exo (np.ndarray): Exogenous feature 'Trend' of shape (samples,)
            look_back (int): Number of time steps in the look-back window

        Returns:
            np.ndarray: Multivariate input of shape (samples, look_back, 4)
        """
        # Reshape exogenous features to (samples, 1, 1)
        return_exo = return_exo.reshape(-1, 1, 1)
        vol_exo = vol_exo.reshape(-1, 1, 1)
        trend_exo = trend_exo.reshape(-1, 1, 1)

        # Repeat across time steps to match look_back
        return_exo = np.repeat(return_exo, look_back, axis=1)
        vol_exo = np.repeat(vol_exo, look_back, axis=1)
        trend_exo = np.repeat(trend_exo, look_back, axis=1)

        # Concatenate along feature axis
        multivariate_input = np.concatenate((X, return_exo, vol_exo, trend_exo), axis=2)
        return multivariate_input

    def save_lstm_model(self):
        """
        Saves the fine-tuned LSTM model to a file in the directory specified by
        `self.model_dir`. The model is saved in the Keras format.

        Returns:
            str: The path where the model was saved.
        """
        # Save model
        model_path = os.path.join(
            self.model_dir, f"{self.asset_name}_lstm_model_tuned.keras"
        )
        self.model.save(model_path)
        print(f"\n💾 Fine-tuned LSTM model saved to {self.safe_relpath(model_path)}")
        return model_path

    def fine_tune(self):
        """
        Performs hyperparameter tuning for the LSTM model using Keras Tuner.
        It creates training and testing datasets with look-back periods, prepares
        the multivariate input, initialises and runs the RandomSearch tuner,
        gets the best hyperparameters and model, trains the best model, makes
        predictions, and stores the results. The best hyperparameters are also
        saved to a JSON file.

        Returns:
            dict: A dictionary containing the predictions, actual values,
                  training input (X_train), and testing input (X_test).
        """

        # Function call
        X_train, Y_train, return_exo_train, vol_exo_train, trend_exo_train = (
            self.create_dataset(
                self.scaled_data,
                self.scaled_return_exo,
                self.scaled_vol_exo,
                self.scaled_trend_exo,
                self.look_back,
            )
        )

        X_test, Y_test, return_exo_test, vol_exo_test, trend_exo_test = (
            self.create_dataset(
                self.scaled_test_data,
                self.scaled_test_return_exo,
                self.scaled_test_vol_exo,
                self.scaled_test_trend_exo,
                self.look_back,
            )
        )

        # Function call
        X_train = self.prepare_multivariate_input(
            X_train.reshape(-1, self.look_back, 1),
            return_exo_train,
            vol_exo_train,
            trend_exo_train,
            self.look_back,
        )
        X_test = self.prepare_multivariate_input(
            X_test.reshape(-1, self.look_back, 1),
            return_exo_test,
            vol_exo_test,
            trend_exo_test,
            self.look_back,
        )

        self.train_data = X_train
        self.test_data = X_test

        # Initialise the tuner
        tuner = RandomSearch(
            self.build_model,
            # Objective to optimise
            objective="val_loss",
            # Number of different hyperparameter combinations to try
            max_trials=self.max_trials,
            # Number of times to train each model variation
            executions_per_trial=self.executions_per_trial,
            project_name=self.project_name,
        )
        # Define early stopping before performing search
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # Perform search
        tuner.search(
            X_train,
            Y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, Y_test),
            callbacks=[early_stopping],
        )

        # Get the best model and best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = self.build_model(best_hps)

        # Save best hyperparameters
        hp_path = os.path.join(
            self.processed_dir, f"{self.asset_name}_best_hyperparameters.json"
        )
        with open(hp_path, "w") as f:
            json.dump(best_hps.values, f, indent=4)
        print(f"\n💾 Best hyperparameters saved to {self.safe_relpath(hp_path)}")

        # Print best model summary and best hyperparameters
        print("\n🥇 Best Model Summary and Hyperparameters")
        print(best_model.summary())
        print(best_hps.values)

        # Train the model
        best_model.fit(
            X_train,
            Y_train,
            validation_data=(X_test, Y_test),
            callbacks=[early_stopping],
        )
        self.model = best_model

        # Make predictions
        predictions = best_model.predict(X_test)
        predictions = self.scalers["Log Close"].inverse_transform(predictions)

        Y_test = self.scalers["Log Close"].inverse_transform(Y_test.reshape(-1, 1))

        train_predictions = best_model.predict(X_train)
        train_predictions = self.scalers["Log Close"].inverse_transform(
            train_predictions
        )
        Y_train = self.scalers["Log Close"].inverse_transform(Y_train.reshape(-1, 1))

        # Store predictions and actual values for plotting
        self.lstm_results = {
            "predictions": predictions,
            "actual": Y_test,
            "X_train": X_train,
            "X_test": X_test,
        }

        self.save_lstm_model()
        self.score_model()

        return self.lstm_results

    def score_model(self):
        """
        Evaluates the fine-tuned LSTM model using common regression metrics:
        Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).
        The metrics are printed to the console.
        """
        # Evaluate the model

        actual = self.lstm_results["actual"]
        predictions = self.lstm_results["predictions"]

        metrics = {
            "RMSE": np.sqrt(mean_squared_error(actual, predictions)),
            "MAE": mean_absolute_error(actual, predictions),
            "R²": r2_score(actual, predictions),
        }
        print(f"\n{self.asset_name} evaluation metrics:\n {metrics}")

    def plot_lstm(self):
        """
        Generates and saves a plot comparing the actual and predicted asset prices
        for the test set. The plot shows the 'Actual Price Trend' and 'Predicted
        Price Trend' over time. The plot is saved as a PNG file in the directory
        specified by `self.plot_dir`.
        """
        if self.lstm_results is None:
            print(f"⚠️ No LSTM results found for {self.asset_name}.")
            return

        test = self.lstm_results["actual"]
        predictions = self.lstm_results["predictions"]

        # Inverse transform the predictions and actual values
        true_prices = np.exp(test)
        predicted_prices = np.exp(predictions)

        test_dates = self.df["Date"].iloc[-len(test) :]

        # Trend prediction plot
        plt.figure(figsize=(12, 4))
        plt.plot(test_dates, true_prices, label="Actual Price Trend", color="Green")
        plt.plot(
            test_dates,
            predicted_prices,
            label="Predicted Price Trend",
            color="Red",
            # linestyle="--",
        )

        plt.title(
            f"{self.asset_name} - asset Price Trend Prediction with Fine-Tuned LSTM"
        )
        plt.tick_params(axis="x", rotation=0)
        plt.xlabel("Date")
        plt.ylabel("Price Trend")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir,
                f"{self.asset_name}_asset_price_prediction_finetuned_lstm.png",
            )
            plt.savefig(plot_path)
            print(f"\n💾 Plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    def finetune_model(self):
        """
        Runs the complete finetuning pipeline, including data splitting, scaling,
        hyperparameter tuning, training the best model, saving the model,
        scoring the model, and plotting the predictions.
        """
        self.split_data()
        self.scale_data()
        self.fine_tune()
        self.plot_lstm()

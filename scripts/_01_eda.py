# _01_eda.py
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yfinance as yf
from IPython.display import display
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore", message="Could not infer format")


class EDA:
    """
    Performs Exploratory Data Analysis (EDA) on stock price data.

    Args:
        stock_name (str): The ticker symbol of the stock (e.g., "AAPL").
        raw_dir (str): Directory to save raw data.
        processed_dir (str): Directory to save processed data.
        plot_dir (str): Directory to save plots.
        rolling_window (int, optional): Window size for rolling calculations.
                                        Defaults to 180.
    """

    def __init__(self, stock_name, raw_dir, processed_dir, plot_dir, rolling_window=1):
        self.stock_name = stock_name
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.plot_dir = plot_dir
        self.ticker = yf.Ticker(self.stock_name)
        self.rolling_window = rolling_window
        self.scaled_close = None
        self.df = None

        # Create output directories if they do not exist
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        print("🧪 Running full EDA pipeline...\n")
        self.load_data()

    def safe_relpath(self, path, start=None):
        """
        Return a relative path, handling cases where paths are on different drives.

        Args:
            path (str): The path to make relative.
            start (str, optional): The starting directory.
                                    Defaults to current working directory.

        Returns:
            str: The relative path if possible, otherwise the original path.
        """
        try:
            return os.path.relpath(path, start)
        except ValueError:
            # Fallback to absolute path if on different drives
            return path

    # ----- Extract historical financial data using YFinance -----#
    def load_data(self):
        """
        Loads historical stock data from Yahoo Finance for the specified date range.
        """
        try:
            self.df = yf.download(
                self.stock_name,
                start="2015-07-01",
                end="2025-07-31",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if self.df.empty:
                print(f"⚠️ Warning: No data returned for {self.stock_name}")
            else:
                # Save raw stock price data
                output_path = os.path.join(
                    self.raw_dir, f"{self.stock_name}_stock_price.csv"
                )
                self.df.to_csv(output_path, index=True)
                print(
                    f"💾 {self.stock_name} raw data saved to \
                        {self.safe_relpath(output_path)}.\n"
                )
        except Exception as e:
            print(f"⚠️ Download failed for {self.stock_name}: {e}")
        return self.df

    # ----- Data cleaning and understanding ----- #
    def data_cleaning(self):
        """
        Performs data cleaning on the loaded stock data.

        Includes handling missing values by dropping columns with high missing
        percentages, imputing numeric columns with the median, and dropping
        object type columns with missing values. Displays head, shape, columns,
        info, and description of the DataFrame before and after cleaning.
        """
        if self.df is None:
            print("⚠️ Stock price missing. Load data first.")
            return

        print("🔹DataFrame Head:")
        display(self.df.head())
        print(f"\n🔹 Shape: {self.df.shape}")
        print(f"\n🔹 Columns: {list(self.df.columns)}")
        print("\n🔹 DataFrame Info:")
        self.df.info()

        # Report missing values
        missing_per_col = self.df.isna().mean().sort_values(ascending=False)

        if missing_per_col.sum() == 0:
            print("\n✅ No missing values found.")
        else:
            print("\n⚠️ Missing values detected. Beginning cleanup...")

            # Drop columns with more than 30% missing
            high_mis_cols = missing_per_col[missing_per_col > 0.3].index.tolist()
            if high_mis_cols:
                self.df.drop(columns=high_mis_cols, inplace=True)
                print(f"\n🚮 Dropped columns with >30% missing: {list(high_mis_cols)}")

            # Impute numeric columns with missing values
            numeric_cols = self.df.select_dtypes(include="number")
            numeric_miss = numeric_cols.columns[numeric_cols.isna().any()].tolist()
            if numeric_miss:
                self.df[numeric_miss] = numeric_cols[numeric_miss].fillna(
                    numeric_cols[numeric_miss].median()
                )
                print(f"\n✂️ Missing values imputed using medians: {numeric_miss}")

            # Drop object columns with missing values
            obj_cols = self.df.select_dtypes(include="object")
            obj_missing = obj_cols.columns[obj_cols.isna().any()].tolist()
            if obj_missing:
                self.df.drop(columns=obj_missing, inplace=True)
                print(
                    f"\n🚮 Dropped obj. type columns with missing values: {obj_missing}"
                )

        print("\nDescribe:")
        display(self.df.describe())
        return self.df

    # ----- Perform a statistical test ---- #
    def run_stationarity_tests(self, series_name):
        """
        Runs Augmented Dickey-Fuller (ADF) and KPSS tests for stationarity
        on the specified time series.

        Args:
            series_name (str): The name of the column (time series) to test.
        """
        if self.df is None:
            print("⚠️ Stock price missing. Load data first.")
            return
        # Drop NaN values for stationarity tests
        series = self.df[series_name].dropna()
        adf_result = adfuller(series)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_result = kpss(series, regression="c")

        print(f"{series_name} — ADF p-value: {adf_result[1]:.4f}")
        print(f"{series_name} — KPSS p-value: {kpss_result[1]:.4f}")

        if adf_result[1] < 0.05 and kpss_result[1] > 0.05:
            print("✅ Series appears stationary.\n")
        else:
            print("⚠️ Series may be non-stationary. Consider differencing.\n")

    # ---- Calculate Trend, Volatility, and Return---- #
    def compute_stats(self):
        """
        Computes rolling mean, standard deviation and return of the 'Close' column.
        The rolling window size is determined by `self.rolling_window`.
        Adds 'Trend', 'Volatility', 'Daily Return' columns to the DataFrame.
        """

        self.df["Trend"] = self.df["Close"].rolling(window=self.rolling_window).mean()
        self.df["Volatility"] = (
            self.df["Close"].rolling(window=self.rolling_window).std()
        )
        self.df["Daily Return"] = self.df["Close"].pct_change()

        return self.df

    # ------ Plot Close, Rolling Mean and, Rolling Std----- #
    def plot_stats(self):
        """
        Plots the rolling mean and rolling standard deviation of the 'Close' column.
        Plots the daily returns of closing price prices over time.

        Saves the plot to the specified plot directory.
        """

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        # Plot Closing Price and Trend
        ax1.plot(self.df.index, self.df["Close"], label="Closing Price")
        ax1.plot(
            self.df.index,
            self.df["Trend"],
            label="Trend (Moving Average)",
            color="orange",
        )
        ax1.set_ylabel("Price")
        ax1.set_title(
            f"{self.stock_name} - Closing Price, Trend, Return and Volatility "
        )
        ax1.legend(loc="upper left")
        ax1.grid(True)

        # Plot Volatility
        ax2.plot(
            self.df.index,
            self.df["Volatility"],
            label="Volatility (Rolling Std)",
            color="Red",
        )
        ax2.set_ylabel("Volatility")
        ax2.set_xlabel("Date")
        ax2.legend(loc="upper left")
        ax2.grid(True)

        # Plot Return
        ax3.plot(
            self.df.index, self.df["Daily Return"], label="Daily Return", color="Purple"
        )
        ax3.set_ylabel("Daily Return")
        ax3.set_xlabel("Date")
        ax3.legend(loc="upper left")
        ax3.grid(True)

        # Adjust layout and show plot
        plt.tight_layout()
        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir, f"{self.stock_name}_close_trend_vol_return.png"
            )
            plt.savefig(plot_path)
            print(f"\n💾 Plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

        # Plot Outliers for all numeric columns
        plt.figure(figsize=(12, 4))
        sns.boxplot(data=self.df.select_dtypes(include="number"), palette="Set1")
        plt.title(f"{self.stock_name} - Box Plot")
        plt.xticks(rotation=0)
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        if self.plot_dir:
            plot_path = os.path.join(self.plot_dir, f"{self.stock_name}_boxplot.png")
            plt.savefig(plot_path)
            print(f"\n💾 plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    # ----- Detect Outliers ----- #
    def detect_outliers(self, threshold=2):
        """
        Detects outliers in the 'Return' column based on a z-score threshold.

        Args:
            threshold (int, optional): The z-score threshold for outlier detection.
            Defaults to 2.
        """
        z_scores = (self.df["Daily Return"] - self.df["Daily Return"].mean()) / self.df[
            "Daily Return"
        ].std()
        outliers = self.df[np.abs(z_scores) > threshold]
        print(f"🔍 Found {len(outliers)} outlier days with |z| > {threshold}")

        # Save processed data to CSV
        output_path = os.path.join(
            self.processed_dir, f"{self.stock_name}_outliers.csv"
        )
        outliers.to_csv(output_path, index=True)
        print(f"\n💾 Outlier DataFrame saved to {self.safe_relpath(output_path)}.")

        # Plot Daily Returns with outliers highlighted
        plt.figure(figsize=(12, 4))
        plt.scatter(self.df.index, self.df["Daily Return"], label="Normal", alpha=0.5)
        plt.scatter(
            outliers.index, outliers["Daily Return"], color="red", label="Outliers"
        )
        plt.title(f"{self.stock_name} - Daily Returns with Outliers Highlighted")
        plt.ylabel("Daily Return")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir, f"{self.stock_name}_daily_return_outlier.png"
            )
            plt.savefig(plot_path)
            print(f"\n💾 Plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    def compute_risk_metrics(self):
        """
        Computes and prints Value at Risk (VaR) at 95% confidence level and
        the Sharpe Ratio for the 'Return' column.
        """
        var_95 = np.percentile(self.df["Daily Return"].dropna(), 5)
        sharpe = self.df["Daily Return"].mean() / self.df["Daily Return"].std()
        print(f"📉 Value at Risk (95%): {var_95:.4f}")
        print(f"📈 Sharpe Ratio: {sharpe:.4f}")

        # Plot the distribution of daily returns with VaR line
        plt.figure(figsize=(10, 4))
        sns.histplot(self.df["Daily Return"], bins=100, kde=True, color="skyblue")
        plt.axvline(
            self.df["Daily Return"].mean(), color="black", linestyle="--", label="Mean"
        )
        plt.axvline(var_95, color="red", linestyle="--", label="VaR (95%)")
        plt.title(f"{self.stock_name} - Distribution of Daily Returns")
        plt.xlabel("Daily Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir, f"{self.stock_name}_daily_return_distribution.png"
            )
            plt.savefig(plot_path)
            print(f"\n💾 Plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    # ---- Save Processed DataFrame-----#
    def get_processed_data(self):
        """
        Prepares and saves the enriched DataFrame to a CSV file.

        Resets the index, formats the 'Date' column, saves the DataFrame
        in the processed data directory,
        and displays the head, shape, columns, info, and description
        of the processed DataFrame.

        Returns:
            None
        """
        # Reset the index so Date becomes a column again
        self.df = self.df.drop(self.df.index[0])

        print("\n🔹Processed DataFrame Head:")
        display(self.df.head())
        print(f"\n🔹Shape: {self.df.shape}")
        print(f"\n🔹Columns: {list(self.df.columns)}")

        # Save processed data to CSV
        output_path = os.path.join(
            self.processed_dir, f"{self.stock_name}_stock_price_enriched.csv"
        )
        self.df.to_csv(output_path, index=True)
        print(f"\n💾 Enriched DataFrame saved to {self.safe_relpath(output_path)}.")

    # ----- Run full pipeline ----- #
    def eda_processor(self):
        """
        Runs the full EDA pipeline: data cleaning, stationarity tests,
        statistical computations, plotting, outlier detection, risk metrics
        computation, and saving processed data.
        """
        self.data_cleaning()
        self.run_stationarity_tests("Close")
        self.compute_stats()
        self.plot_stats()
        self.run_stationarity_tests("Daily Return")
        self.detect_outliers()
        self.compute_risk_metrics()
        self.get_processed_data()

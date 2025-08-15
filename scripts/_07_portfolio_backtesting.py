# _07_portfolio_backtesting

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


class PortfolioBacktester:
    """
    A class to perform portfolio backtesting, comparing the performance of an
    optimised portfolio against a benchmark portfolio using historical daily returns.
    """

    def __init__(
        self,
        bnd_path,
        spy_path,
        tsla_path,
        optimal_weights,
        benchmark_weights,
        plot_dir,
        processed_dir,
        start_date="2024-08-01",
        end_date="2025-07-31",
    ):
        """
        Initialises the PortfolioBacktester with paths, weights, and backtest window.

        Args:
            bnd_path (str): Path to the CSV file containing BND daily returns.
            spy_path (str): Path to the CSV file containing SPY daily returns.
            tsla_path (str): Path to the CSV file containing TSLA daily returns.
            optimal_weights (dict): A dictionary where keys are asset tickers and
                                    values are the optimised weights.
            benchmark_weights (dict): A dictionary where keys are asset tickers and
                                    values are the benchmark weights.
            plot_dir (str): Directory to save generated plots.
            processed_dir (str): Directory to save processed data files.
            start_date (str, optional): Start date of the backtest window (YYYY-MM-DD).
                                        Defaults to "2024-08-01".
            end_date (str, optional): End date of the backtest window (YYYY-MM-DD).
                                    Defaults to "2025-07-31".
        """
        self.bnd_path = bnd_path
        self.spy_path = spy_path
        self.tsla_path = tsla_path
        self.plot_dir = plot_dir
        self.processed_dir = processed_dir
        self.optimal_weights = optimal_weights
        self.benchmark_weights = benchmark_weights
        self.start_date = start_date
        self.end_date = end_date

        # Create directory if it does not exist
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.daily_returns = None

        print("🧪 Running full backesting pipeline...")

        self.load_and_concatenate_data()

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

    def load_and_concatenate_data(self):
        """
        Loads daily return data from CSV files, cleans, sorts, and concatenates
        them into a single DataFrame for the specified backtest window.

        Returns:
            pd.DataFrame: DataFrame containing daily returns for all assets.
        """
        self.bnd_df = pd.read_csv(self.bnd_path)
        self.spy_df = pd.read_csv(self.spy_path)
        self.tsla_df = pd.read_csv(self.tsla_path)

        # Clean and index each DataFrame
        for df in [self.bnd_df, self.spy_df, self.tsla_df]:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

        # Slice to backtest window
        bnd_returns = self.bnd_df.loc[
            self.start_date : self.end_date, "Daily Return"
        ].rename("BND")
        spy_returns = self.spy_df.loc[
            self.start_date : self.end_date, "Daily Return"
        ].rename("SPY")
        tsla_returns = self.tsla_df.loc[
            self.start_date : self.end_date, "Daily Return"
        ].rename("TSLA")

        # Concatenate into one DataFrame
        self.daily_returns = pd.concat([bnd_returns, spy_returns, tsla_returns], axis=1)

        # Reset index to bring Date back as a column
        # self.daily_returns = self.daily_returns.reset_index()

        file_path = os.path.join(self.processed_dir, "combined_return_data.csv")
        self.daily_returns.to_csv(file_path)
        print(f"\n💾 Combined DataFrame saved to {self.safe_relpath(file_path)}")

        print("📊 DataFrame head:")
        display(self.daily_returns.head())
        return self.daily_returns

    def simulate_portfolio(self, weights):
        """
        Simulates the daily returns and cumulative value of a portfolio
        given a set of asset weights.

        Args:
            weights (dict): A dictionary where keys are asset tickers and
                            values are the weights for each asset.

        Returns:
            tuple: A tuple containing:
                - portfolio_returns (pd.Series): Daily returns of the portfolio.
                - portfolio_value (pd.Series): Cumulative value of the portfolio
                                               starting with $1.
        """
        weighted_returns = self.daily_returns.mul(pd.Series(weights), axis=1)
        portfolio_returns = weighted_returns.sum(axis=1)
        portfolio_value = (1 + portfolio_returns).cumprod()
        return portfolio_returns, portfolio_value

    def calculate_metrics(self, returns):
        """
        Calculates performance metrics for a given series of daily returns.

        Args:
            returns (pd.Series): A pandas Series of daily portfolio returns.

        Returns:
            dict: A dictionary containing calculated metrics:
                - "Annualised Volatility": Annualised standard deviation of returns.
                - "Sharpe Ratio": Annualised Sharpe ratio (assumes risk-free rate is 0).
                - "Total Return": The total cumulative return over the period.
        """
        mean_daily = returns.mean()
        std_daily = returns.std()
        sharpe = mean_daily / std_daily * np.sqrt(252)
        total_return = (1 + returns).prod() - 1
        return {
            "Annualised Volatility": std_daily * np.sqrt(252),
            "Sharpe Ratio": sharpe,
            "Total Return": total_return,
        }

    def plot_performance(self, opt_value, bench_value):
        """
        Plots the cumulative performance of the optimised and benchmark portfolios.

        Args:
            opt_value (pd.Series): Cumulative value of the optimised portfolio.
            bench_value (pd.Series): Cumulative value of the benchmark portfolio.
        """
        plt.figure(figsize=(12, 4))
        plt.plot(opt_value, label="Optimised Portfolio", linewidth=2)
        plt.plot(bench_value, label="Benchmark Portfolio", linestyle="--")
        plt.title("Portfolio Performance: Backtest")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        if self.plot_dir:
            plot_path = os.path.join(self.plot_dir, "portfolio_performance.png")
            plt.savefig(plot_path)
            print(f"\n💾 plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    def plot_asset_contributions(self):
        """
        Plots the cumulative returns of individual assets in the portfolio.
        """
        cumulative_returns = (1 + self.daily_returns).cumprod()
        plt.figure(figsize=(12, 4))
        for col in cumulative_returns.columns:
            plt.plot(cumulative_returns.index, cumulative_returns[col], label=col)
        plt.title("Cumulative Asset-Level Returns")
        plt.xlabel("Date")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        if self.plot_dir:
            plot_path = os.path.join(self.plot_dir, "asset_contributions.png")
            plt.savefig(plot_path)
            print(f"\n💾 plot saved to {self.safe_relpath(plot_path)}")
        plt.show()
        plt.close()

    def print_summary(self, metrics_dict):
        """
        Prints a summary of the performance metrics for the portfolios.

        Args:
            metrics_dict (dict): A dictionary where keys are portfolio names
                                 and values are dictionaries of their metrics.
        """
        print("\n📊 Performance Summary:")
        for name, metrics in metrics_dict.items():
            print(f"\n{name}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

    def run_backtest(self):
        """
        Runs the full backtesting process, including simulating portfolios,
        calculating metrics, plotting performance, and printing the summary.
        """
        opt_returns, opt_value = self.simulate_portfolio(self.optimal_weights)
        bench_returns, bench_value = self.simulate_portfolio(self.benchmark_weights)

        opt_metrics = self.calculate_metrics(opt_returns)
        bench_metrics = self.calculate_metrics(bench_returns)

        self.plot_performance(opt_value, bench_value)
        self.plot_asset_contributions()

        results = {
            "Optimised Portfolio": opt_metrics,
            "Benchmark Portfolio": bench_metrics,
        }
        self.print_summary(results)

# _06_portfolio_optimisation

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display
from pypfopt import EfficientFrontier, risk_models


class PortfolioOptimiser:
    """
    A class to perform portfolio optimisation using historical and forecasted data.

    This class loads historical stock data (BND, SPY, TSLA), combines their
    daily returns, calculates expected returns and the covariance matrix,
    simulates random portfolios, and then optimises the portfolio using
    the Efficient Frontier to find the maximum Sharpe ratio and minimum
    volatility portfolios. It also generates plots to visualise the
    efficient frontier and asset allocations.

    Attributes:
        bnd_path (str): Path to the BND historical data CSV file.
        spy_path (str): Path to the SPY historical data CSV file.
        tsla_path (str): Path to the TSLA historical or forecasted data CSV file.
        plot_dir (str): Directory to save generated plots.
        prediction_df (pd.DataFrame): DataFrame containing forecasted data.
        bnd_df (pd.DataFrame): DataFrame containing BND historical data.
        spy_df (pd.DataFrame): DataFrame containing SPY historical data.
        tsla_df (pd.DataFrame): DataFrame containing TSLA historical data.
        combined_returns (pd.DataFrame): DataFrame with combined daily returns.
        expected_returns (np.ndarray): Annualised expected returns for assets.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        random_portfolios (list): List of dictionaries for simulated portfolios.
        optimised_results (dict): Dictionary storing results for max Sharpe
                                and min volatility portfolios.
        cleaned_weights (dict): Cleaned weights for the max Sharpe portfolio.
        performance (tuple): Performance metrics for the max Sharpe portfolio.
    """

    def __init__(
        self,
        bnd_path,
        spy_path,
        tsla_path,
        plot_dir,
        col="Predicted Return",
    ):
        """
        Initialises the PortfolioOptimiser with data paths and plot directory.

        Args:
            bnd_path (str): Path to the BND historical data CSV file.
            spy_path (str): Path to the SPY historical data CSV file.
            tsla_path (str): Path to the TSLA historical or forecasted data CSV file.
            plot_dir (str): Directory to save generated plots.
            col (str): Column name to use for optimisation.
                            Defaults to "Predicted Return".
        """
        self.bnd_path = bnd_path
        self.spy_path = spy_path
        self.tsla_path = tsla_path
        self.col = col
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

        print("🧪 Running full optimisation pipeline...")
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
        Loads historical data for BND, SPY, and TSLA from their respective CSV files.
        Sorts data by date, resets index, and displays the head of each DataFrame.
        """
        self.bnd_df = pd.read_csv(self.bnd_path)
        self.spy_df = pd.read_csv(self.spy_path)
        self.tsla_df = pd.read_csv(self.tsla_path)
        for df in [self.bnd_df, self.spy_df, self.tsla_df]:
            df["Date"] = pd.to_datetime(df["Date"])
            df.sort_values(by="Date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
        print(
            f"\n📈 BND historical data loaded from {self.safe_relpath(self.bnd_path)}"
        )
        print("🔹 BND DataFrame Head:")
        display(self.bnd_df.head())

        print(
            f"\n📈 SPY historical data loaded from {self.safe_relpath(self.spy_path)}"
        )
        print("🔹 SPY DataFrame Head:")
        display(self.spy_df.head())

        print(
            f"\n📈 TSLA historical data loaded from {self.safe_relpath(self.tsla_path)}"
        )
        print("🔹 TSLA DataFrame Head:")
        display(self.tsla_df.head())

    def combine_returns(
        self,
    ):
        """
        Combines daily returns from loaded dataframes and computes
        annualised expected returns and the covariance matrix.
        """
        # Use raw daily returns
        self.combined_returns = pd.DataFrame(
            {
                # "TSLA": self.tsla_df["Daily Return"],
                "TSLA": self.tsla_df[self.col],
                "BND": self.bnd_df["Daily Return"],
                "SPY": self.spy_df["Daily Return"],
            }
        ).dropna()

        # Compute expected returns (annualised)
        self.expected_returns = np.array(
            [
                # self.tsla_df["Daily Return"].mean() * 252,
                self.tsla_df[self.col].mean() * 252,
                self.bnd_df["Daily Return"].mean() * 252,
                self.spy_df["Daily Return"].mean() * 252,
            ]
        )

        # Compute covariance matrix using raw daily returns
        self.cov_matrix = risk_models.sample_cov(
            self.combined_returns, returns_data=True
        ).values

        print("\n📊 Covariance Matrix Summary:")
        print("Mean:", np.mean(self.cov_matrix))
        print("Max:", np.max(self.cov_matrix))
        print("Min:", np.min(self.cov_matrix))

    def validate_covariance_scaling(self):
        """
        Validates the scaling of the covariance matrix calculated by PyPortfolioOpt
        against a manually calculated version.
        """
        manual_cov = self.combined_returns.cov().values * 252
        auto_cov = risk_models.sample_cov(
            self.combined_returns, returns_data=True
        ).values
        diff = np.abs(manual_cov - auto_cov).mean()
        print("\n📌 Manual Covariance Matrix:")
        print(manual_cov)
        print("\n📌 PyPortfolioOpt Covariance Matrix (returns_data=True):")
        print(auto_cov)
        print(
            f"\n🧪 Covariance Matrix Difference (manual vs PyPortfolioOpt): {diff:.6f}"
        )

    def simulate_random_portfolios(self, num_portfolios=10000):
        """
        Simulates a specified number of random portfolios to visualise the
        investment space and the efficient frontier.

        Args:
            num_portfolios (int, optional): The number of random portfolios
                                            to simulate. Defaults to 10000.
        """
        n_assets = 3
        weights = np.random.dirichlet(np.ones(n_assets), num_portfolios)
        returns = np.dot(weights, self.expected_returns)
        stddevs = np.sqrt(np.diag(weights @ self.cov_matrix @ weights.T))
        sharpe_ratios = returns / stddevs
        self.random_portfolios = [
            {"weights": w, "return": r, "stddev": s, "sharpe": sr}
            for w, r, s, sr in zip(weights, returns, stddevs, sharpe_ratios)
        ]

    def optimise_portfolio(self):
        """
        Optimises the portfolio to find the maximum Sharpe ratio and minimum
        volatility portfolios using the Efficient Frontier.
        Stores the weights and performance metrics for both optimal portfolios.
        """
        mu = pd.Series(self.expected_returns, index=["TSLA", "BND", "SPY"])
        S = risk_models.sample_cov(self.combined_returns, returns_data=True)
        ef = EfficientFrontier(mu, S)

        # Max Sharpe Ratio Portfolio
        ef.max_sharpe()
        self.cleaned_weights = ef.clean_weights()
        self.performance = ef.portfolio_performance()
        max_sharpe_weights = self.cleaned_weights
        max_sharpe_perf = self.performance

        # Re-instantiate for Min Volatility (to avoid internal state conflicts)
        ef_min = EfficientFrontier(mu, S)
        ef_min.min_volatility()
        min_vol_weights = ef_min.clean_weights()
        min_vol_perf = ef_min.portfolio_performance()

        # Store both results
        self.optimised_results = {
            "max_sharpe": {
                "weights": max_sharpe_weights,
                "performance": {
                    "return": max_sharpe_perf[0],
                    "stddev": max_sharpe_perf[1],
                    "sharpe": max_sharpe_perf[0] / max_sharpe_perf[1],
                },
            },
            "min_vol": {
                "weights": min_vol_weights,
                "performance": {
                    "return": min_vol_perf[0],
                    "stddev": min_vol_perf[1],
                    "sharpe": min_vol_perf[0] / min_vol_perf[1],
                },
            },
        }

        print("\n📈 Max Sharpe Portfolio:")
        for k, v in self.optimised_results["max_sharpe"]["weights"].items():
            print(f"{k}: {v:.2%}")
        print(
            "Return: {:.2%}, Volatility: {:.2%}, Sharpe: {:.2f}".format(
                *self.optimised_results["max_sharpe"]["performance"].values()
            )
        )

        print("\n📉 Min Volatility Portfolio:")
        for k, v in self.optimised_results["min_vol"]["weights"].items():
            print(f"{k}: {v:.2%}")
        print(
            "Return: {:.2%}, Volatility: {:.2%}, Sharpe: {:.2f}".format(
                *self.optimised_results["min_vol"]["performance"].values()
            )
        )

    def plot_enhanced_frontier(self):
        """
        Plots the efficient frontier, including random portfolios, the maximum
        Sharpe ratio portfolio, the minimum volatility portfolio, and individual assets.
        Saves the plot to the specified directory.
        """
        plt.figure(figsize=(12, 6))

        # Plot random portfolios
        returns = [p["return"] for p in self.random_portfolios]
        stddevs = [p["stddev"] for p in self.random_portfolios]
        sharpe_ratios = np.array(returns) / np.array(stddevs)
        plt.scatter(stddevs, returns, c=sharpe_ratios, cmap="viridis", s=10, alpha=0.3)
        plt.colorbar(label="Sharpe Ratio")

        # Plot Max Sharpe Portfolio
        max_perf = self.optimised_results["max_sharpe"]["performance"]
        plt.scatter(
            max_perf["stddev"],
            max_perf["return"],
            marker="*",
            color="red",
            s=500,
            label="Max Sharpe",
        )

        # Plot Min Volatility Portfolio
        min_perf = self.optimised_results["min_vol"]["performance"]
        plt.scatter(
            min_perf["stddev"],
            min_perf["return"],
            marker="*",
            color="blue",
            s=500,
            label="Min Volatility",
        )

        # Plot individual assets
        for i, asset in enumerate(["TSLA", "BND", "SPY"]):
            vol = np.sqrt(self.cov_matrix[i, i])
            ret = self.expected_returns[i]
            plt.scatter(vol, ret, color="black", s=200)
            plt.annotate(asset, (vol * 1.01, ret * 1.01))

        plt.title(f"Efficient Frontier with Optimal Portfolios {self.col}")
        plt.xlabel("Volatility")
        plt.ylabel("Return")
        plt.legend()
        plt.tight_layout()

        if self.plot_dir:
            plot_path = os.path.join(self.plot_dir, f"enhanced_frontier_{self.col}.png")
            plt.savefig(plot_path)
            print(
                f"\n💾 Enhanced frontier plot saved to {self.safe_relpath(plot_path)}"
            )

        plt.show()
        plt.close()

    def plot_allocation_pie(self):
        """
        Plots pie charts comparing the asset allocation of the maximum Sharpe
        ratio portfolio and the minimum volatility portfolio.
        Saves the plot to the specified directory.
        """

        def autopct_format(pct):
            return f"{pct:.1f}%" if pct > 0 else ""

        plt.figure(figsize=(12, 4))

        # Max Sharpe Allocation
        max_weights = self.optimised_results["max_sharpe"]["weights"]
        plt.subplot(1, 2, 1)
        colors = ["#2cc997", "#8E3F20", "#214494"]
        plt.pie(
            max_weights.values(),
            labels=max_weights.keys(),
            autopct=autopct_format,
            startangle=90,
            colors=colors,
            pctdistance=0.85,
            labeldistance=1.1,
        )
        plt.title("Max Sharpe Portfolio Allocation")
        plt.legend()

        # Min Volatility Allocation
        min_weights = self.optimised_results["min_vol"]["weights"]
        plt.subplot(1, 2, 2)
        colors = ["#18b784", "#c75122", "#164ecf"]
        plt.pie(
            min_weights.values(),
            labels=min_weights.keys(),
            autopct=autopct_format,
            startangle=90,
            colors=colors,
            pctdistance=0.85,
            labeldistance=1.1,
        )
        plt.legend()

        plt.title("Min Volatility Portfolio Allocation")
        plt.tight_layout()
        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir, f"allocation_comparison_{self.col}.png"
            )
            plt.savefig(plot_path)
            print(f"\n💾 Allocation pie chart saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    def summary(self):
        """
        Prints a summary of the portfolio optimisation results, including
        optimal weights and performance metrics for the maximum Sharpe ratio portfolio.
        """
        print("\n✅ Portfolio Optimisation Complete")
        print(f"🔹 Optimal Weights: {self.cleaned_weights}")
        print(f"🔹 Expected Annual Return: {self.performance[0]:.2%}")
        print(f"🔹 Annual Volatility: {self.performance[1]:.2%}")
        print(f"🔹 Sharpe Ratio: {self.performance[2]:.2f}")

    def run_pipeline(self):
        """
        Runs the complete portfolio optimisation pipeline.
        """
        print("🧪 Running full optimisation pipeline...")
        self.combine_returns()
        self.validate_covariance_scaling()
        self.simulate_random_portfolios()
        self.optimise_portfolio()
        self.plot_enhanced_frontier()
        self.plot_allocation_pie()
        self.summary()

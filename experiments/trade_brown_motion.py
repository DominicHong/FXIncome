import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import itertools
import multiprocessing as mp
from functools import partial


class BondPriceSimulator:
    """A class for simulating and analyzing bond price movements using Geometric Brownian Motion.

    This class provides methods for:
    - Generating bond price time series using either discrete or continuous GBM
    - Plotting price paths
    - Verifying log-normal distribution of returns
    """

    def __init__(
        self,
        S0: float = 100,
        mu: float = 0.02,
        sigma: float = 0.015,
        days: int = 365,
        days_of_year: int = 365,  # Use Calendar Day to generate price paths.
        num_paths: int = 10,
        mode: str = "continuous",  # Using str since Literal is not needed here
    ):
        """Initialize the simulator with given parameters.

        Args:
            S0: Initial price
            mu: Annualized expected return
            sigma: Annualized volatility
            days: Number of days of one simulation path
            days_of_year: Number of trading days in a year.
            num_paths: Number of simulation paths
            mode: "discrete" for discrete-time GBM or "continuous" for continuous-time GBM
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.days = days
        self.num_paths = num_paths
        self.mode = mode
        self.prices_df: pd.DataFrame | None = None
        self.days_of_year = days_of_year
        self._dt = (
            1 / self.days_of_year
        )  # The annualized sigma is estimated from trade days.
        self._sqrt_dt = np.sqrt(self._dt)

    def generate_prices(self) -> pd.DataFrame:
        """Generate bond price time series using either discrete or continuous GBM.

        Returns:
            pd.DataFrame: DataFrame containing the simulated price paths. Index is date and columns are price paths.
        """
        # Generate time series
        prices = np.zeros((self.num_paths, self.days))
        prices[:, 0] = self.S0

        if self.mode == "discrete":
            # Discrete-time GBM (Euler-Maruyama approximation)
            for t in range(1, self.days):
                prices[:, t] = (
                    prices[:, t - 1]
                    + self.mu * prices[:, t - 1] * self._dt
                    + self.sigma
                    * prices[:, t - 1]
                    * np.random.normal(0, 1, size=self.num_paths)
                    * self._sqrt_dt
                )
        else:  # continuous
            # Continuous-time GBM (exact solution)
            for t in range(1, self.days):
                prices[:, t] = prices[:, t - 1] * np.exp(
                    (self.mu - 0.5 * self.sigma**2) * self._dt
                    + self.sigma
                    * np.random.normal(0, 1, size=self.num_paths)
                    * self._sqrt_dt
                )

        # Create DataFrame with numerical index for long simulations
        try:
            # Try to create a business day date range
            if self.days <= 10000:  # Only use date range for reasonable periods
                dates = pd.date_range(start="today", periods=self.days, freq="D")
                self.prices_df = pd.DataFrame(
                    prices.T,
                    index=dates,
                    columns=[f"path_{i+1}" for i in range(self.num_paths)],
                )
            else:
                # For very long simulations, use numerical index
                self.prices_df = pd.DataFrame(
                    prices.T,
                    index=np.arange(self.days),
                    columns=[f"path_{i+1}" for i in range(self.num_paths)],
                )
        except (OverflowError, pd._libs.tslibs.np_datetime.OutOfBoundsTimedelta):
            # Fallback to numerical index if date range fails
            self.prices_df = pd.DataFrame(
                prices.T,
                index=np.arange(self.days),
                columns=[f"path_{i+1}" for i in range(self.num_paths)],
            )

        return self.prices_df

    def plot_price_paths(self) -> None:
        """Plot all simulated price paths."""
        if self.prices_df is None:
            raise ValueError("No price data available. Run generate_prices() first.")

        plt.figure(figsize=(12, 6))
        self.prices_df.plot(
            title=f"Bond Price Simulation - {self.mode.capitalize()} Geometric Brownian Motion"
        )
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.show()

    def verify_normal_distribution(self) -> pd.DataFrame:
        """Verify if:
          1. daily returns of each path follow a normal distribution in discrete mode.
          2. daily log returns each path follow a normal distribution in continuous mode.
        and compare theoretical vs actual values.

        Returns:
            pd.DataFrame: DataFrame containing test results for each path
        """
        if self.prices_df is None:
            raise ValueError("No price data available. Run generate_prices() first.")

        # Calculate daily returns
        returns = self.prices_df / self.prices_df.shift(1) - 1
        log_returns = np.log(self.prices_df / self.prices_df.shift(1))

        # Calculate theoretical values
        if self.mode == "discrete":
            theoretical_mean = self.mu * self._dt
            theoretical_var = self.sigma**2 * self._dt
        else:
            theoretical_mean = (self.mu - 0.5 * self.sigma**2) * self._dt
            theoretical_var = self.sigma**2 * self._dt
        # Create subplots
        n_paths = len(self.prices_df.columns)
        n_rows = (n_paths + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        axes = axes.ravel()

        test_results = []

        for i, column in enumerate(self.prices_df.columns):
            if self.mode == "discrete":
                path_returns = returns[column].dropna()
            else:
                path_returns = log_returns[column].dropna()
            empirical_mean = path_returns.mean()
            empirical_var = path_returns.var()

            # Plot histogram and kernel density
            sns.histplot(data=path_returns, stat="density", kde=True, ax=axes[i])

            # Fit normal distribution
            mu_fit, std_fit = stats.norm.fit(path_returns)
            x = np.linspace(path_returns.min(), path_returns.max(), 100)
            p = stats.norm.pdf(x, mu_fit, std_fit)
            axes[i].plot(x, p, "r-", lw=2, label="Normal Distribution Fit")

            # Perform Shapiro-Wilk test
            statistic, p_value = stats.shapiro(path_returns)
            test_results.append(
                {
                    "path": column,
                    "statistic": statistic,
                    "p_value": p_value,
                    "empirical_mean": empirical_mean,
                    "empirical_var": empirical_var,
                    "mean_diff": empirical_mean - theoretical_mean,
                    "var_diff": empirical_var - theoretical_var,
                }
            )

            axes[i].set_title(
                f"{column}\np-value: {p_value:.4f}\n"
                f"Mean: {empirical_mean:.6f} (theory: {theoretical_mean:.6f})\n"
                f"Var: {empirical_var:.6f} (theory: {theoretical_var:.6f})"
            )
            axes[i].set_xlabel("Daily Returns")
            axes[i].set_ylabel("Density")
            axes[i].legend()
            axes[i].grid(True)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

        # Create and display results DataFrame
        results_df = pd.DataFrame(test_results)
        self._print_analysis_results(results_df, theoretical_mean, theoretical_var)
        return results_df

    def _print_analysis_results(
        self, results_df: pd.DataFrame, theoretical_mean: float, theoretical_var: float
    ) -> None:
        """Print analysis results comparing theoretical and empirical values.

        Args:
            results_df: DataFrame containing test results
            theoretical_mean: Theoretical mean of daily returns
            theoretical_var: Theoretical variance of daily returns
        """
        print("\nTheoretical values:")
        print(f"Mean (theoretical) = {theoretical_mean*100:.4f}%")
        print(f"Variance (theoretical) = {theoretical_var*100:.6f}%")
        print("\nEmpirical results:")
        print(
            f"Mean (empirical avg) = {results_df['empirical_mean'].mean()*100:.4f}% "
            f"± {results_df['empirical_mean'].std()*100:.4f}%"
        )
        print(
            f"Variance (empirical avg) = {results_df['empirical_var'].mean()*100:.6f}% "
            f"± {results_df['empirical_var'].std()*100:.6f}%"
        )
        print("\nShapiro-Wilk normality test results:")
        print(
            results_df[
                ["path", "p_value", "empirical_mean", "empirical_var"]
            ].to_string(index=False)
        )
        print("\nOverall conclusion:")
        print(
            f"At 0.05 significance level, {sum(results_df['p_value'] > 0.05)} paths "
            "follow normal distribution"
        )
        print(
            f"{sum(results_df['p_value'] <= 0.05)} paths do not follow normal distribution"
        )


class TradeSimulator:
    """A class for simulating trading based on directional predictions with a given accuracy.

    The trader can be in these states:
    1. No position (0)
    2. Long 1 unit (+1)
    3. Short 1 unit (-1) [only if short_selling is allowed]

    Trading rules based on predictions:
    - Trading occurs every trading_interval days
    - At day T, the predictor predicts price direction at T + trading_interval
    - If predicted UP:
        - No position -> Buy 1 unit
        - Long -> Hold
        - Short -> Buy 1 unit (close position) [only if short_selling is allowed]
    - If predicted DOWN:
        - No position -> Sell 1 unit [only if short_selling is allowed, otherwise hold]
        - Long -> Sell 1 unit (close position)
        - Short -> Hold [only if short_selling is allowed]
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        initial_capital: float = 110.0,
        prediction_accuracy: float = 0.6,
        risk_free_rate: float = 0.02,
        days_of_year: int = 365,  # Use Calendar Day to generate price paths.
        slippage: float = 0.0004,  # 0.04% slippage by default
        short_selling: bool = True,  # Whether to allow short selling
        trading_interval: int = 5,  # Trading interval in days
    ):
        """Initialize the trade simulator.

        Args:
            prices_df: DataFrame with bond price paths
            initial_capital: Initial trading capital
            prediction_accuracy: Accuracy of the direction predictor (0.5-1.0)
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            days_of_year: Number of trading days in a year
            slippage: Trading slippage as a percentage (e.g., 0.0004 for 0.04%)
            short_selling: Whether to allow short selling
            trading_interval: Number of days between trades (default: 5)
        """
        if days_of_year <= 0:
            raise ValueError("days_of_year must be positive")
        if trading_interval <= 0:
            raise ValueError("trading_interval must be positive")
            
        self.prices_df = prices_df
        self.initial_capital = initial_capital
        self.prediction_accuracy = prediction_accuracy
        self.days_of_year = days_of_year
        self.daily_rf_rate = (1 + risk_free_rate) ** (1 / days_of_year) - 1
        self.slippage = slippage
        self.short_selling = short_selling
        self.trading_interval = trading_interval
        self.results: dict[str, pd.DataFrame] = {}

    def _calculate_trade_price(self, base_price: float, is_buy: bool) -> float:
        """Calculate the actual trade price including slippage.

        Args:
            base_price: The base price before slippage
            is_buy: True if buying, False if selling

        Returns:
            The actual trade price including slippage
        """
        slippage_factor = 1 + (self.slippage if is_buy else -self.slippage)
        return base_price * slippage_factor

    def _generate_predictions(
        self, prices: pd.Series, random_seed: int | None = None
    ) -> np.ndarray:
        """Generate predictions with exactly the target accuracy for future price movements.

        Args:
            prices: Series of price values
            random_seed: Optional random seed for reproducible predictions. If None, predictions
                        will be random each time.

        Returns:
            numpy array of boolean predictions with exactly the target accuracy
        """
        # Calculate future returns and actual price changes
        future_returns = prices.shift(-self.trading_interval) - prices
        actual_changes = future_returns > 0

        n_predictions = len(actual_changes)
        n_correct = round(n_predictions * self.prediction_accuracy)

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        predictions = np.zeros(n_predictions, dtype=bool)
        # Randomly select which predictions will be correct
        correct_indices = np.random.choice(n_predictions, size=n_correct, replace=False)

        # Set predictions to achieve exact accuracy
        for i in range(n_predictions):
            if i in correct_indices:
                predictions[i] = actual_changes.iloc[i]  # Correct prediction
            else:
                predictions[i] = not actual_changes.iloc[i]  # Incorrect prediction

        # Reset random seed if it was set
        if random_seed is not None:
            np.random.seed(None)

        return predictions

    def _execute_trade(self, curr_price: float, position: int, prediction: bool) -> tuple[int, float]:
        """Execute a trade based on the current position and prediction.

        Subclasses should override this method to implement their specific trading logic.
        
        Args:
            curr_price: Current price
            position: Current position (-1, 0, or 1)
            prediction: True for predicted up, False for predicted down
            
        Returns:
            Tuple of (new_position, capital_change)
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _process_trading_results(self, trades: list, column: str) -> tuple[pd.DataFrame, dict]:
        """Process trading results into DataFrame and calculate statistics.
        
        Args:
            trades: List of trade records
            column: Name of the price path
            
        Returns:
            Tuple of (trades DataFrame, statistics dictionary)
        """
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        trades_df["path"] = column
        trades_df["total_value"] = (
            trades_df["capital"] + trades_df["position"] * trades_df["price"]
        )
        trades_df["returns"] = trades_df["total_value"].pct_change()

        # Calculate statistics
        total_return = (trades_df["total_value"].iloc[-1] - self.initial_capital) / self.initial_capital
        daily_returns = trades_df["returns"].dropna()
        
        # Calculate Sharpe ratio with proper handling of edge cases
        if len(daily_returns) > 0:
            returns_std = daily_returns.std()
            if returns_std > 0:  # Avoid division by zero
                sharpe_ratio = (
                    np.sqrt(self.days_of_year)
                    * (daily_returns.mean() - self.daily_rf_rate)
                    / returns_std
                )
            else:
                sharpe_ratio = 0.0  # If all returns are identical, Sharpe ratio is 0
        else:
            sharpe_ratio = 0.0  # If no trades, Sharpe ratio is 0

        stats = {
            "path": column,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self._calculate_max_drawdown(trades_df["total_value"]),
            "final_capital": trades_df["capital"].iloc[-1],
            "daily_win_rate": (daily_returns > 0).mean() if len(daily_returns) > 0 else 0.0,
        }

        return trades_df, stats

    def simulate_trades(
        self, random_seed: int | None = None
    ) -> dict[str, pd.DataFrame]:
        """Simulate trades for all price paths and calculate statistics.

        Args:
            random_seed: Optional random seed for reproducible predictions. If None, predictions
                        will be random each time.
        Returns:
            Dict containing DataFrames with trading results and statistics
        """
        all_trades = []
        all_stats = []

        for column in self.prices_df.columns:
            # Get price series for this path
            prices = self.prices_df[column]

            # Generate predictions for price changes
            predictions = self._generate_predictions(prices, random_seed=random_seed)

            # Initialize trading variables
            position = 0  # Start with no position
            capital = self.initial_capital
            trades = []

            # Simulate trading every trading_interval days
            for t in range(
                0, len(prices) - self.trading_interval, self.trading_interval
            ):
                curr_price = prices.iloc[t]

                # Record state before trade
                trades.append(
                    {
                        "date": prices.index[t],
                        "price": curr_price,
                        "position": position,
                        "capital": capital,
                        "prediction": "Up" if predictions[t] else "Down",
                    }
                )

                # Execute trading logic
                new_position, capital_change = self._execute_trade(
                    curr_price, position, predictions[t]
                )
                position = new_position
                capital += capital_change

                # Record state for days between trades
                for inter_t in range(
                    t + 1, min(t + self.trading_interval, len(prices))
                ):
                    trades.append(
                        {
                            "date": prices.index[inter_t],
                            "price": prices.iloc[inter_t],
                            "position": position,
                            "capital": capital,
                            "prediction": "NONE",  # No prediction on non-trading days
                        }
                    )

            # Close final position at last price
            if position == 1:
                final_trade_price = self._calculate_trade_price(
                    prices.iloc[-1], is_buy=False
                )
                capital += final_trade_price
                position = 0
            elif position == -1:
                final_trade_price = self._calculate_trade_price(
                    prices.iloc[-1], is_buy=True
                )
                capital -= final_trade_price
                position = 0
            trades.append(
                {
                    "date": prices.index[-1],
                    "price": prices.iloc[-1],
                    "position": position,
                    "capital": capital,
                    "prediction": "NONE",
                }
            )

            # Process results
            trades_df, stats = self._process_trading_results(trades, column)
            all_trades.append(trades_df)
            all_stats.append(stats)
            

        # Combine results
        self.results["trades"] = pd.concat(all_trades, ignore_index=True)
        self.results["statistics"] = pd.DataFrame(all_stats)

        return self.results

    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate the maximum drawdown from peak for a series of values.

        The drawdown is calculated as the percentage decline from the running maximum.
        For negative values, we use absolute values to ensure correct calculation.

        Returns:
            float: The maximum drawdown as a negative percentage (e.g., -0.2 for 20% drawdown)
        """
        # Calculate running maximum
        peak = values.expanding(min_periods=1).max()

        # Calculate drawdown using absolute values to handle negative numbers correctly
        drawdown = (values - peak) / abs(peak)

        return drawdown.min()

    def plot_results(self) -> None:
        """Plot trading results including:
        1. Capital evolution for each path
        2. Distribution of returns
        3. Distribution of Sharpe ratios
        """
        if not self.results:
            raise ValueError("No results available. Run simulate_trades() first.")

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot capital evolution
        for path in self.prices_df.columns:
            path_trades = self.results["trades"][self.results["trades"]["path"] == path]
            axes[0, 0].plot(
                path_trades["date"], path_trades["total_value"], alpha=0.5, label=path
            )
        axes[0, 0].set_title("Capital Evolution")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Total Value")
        axes[0, 0].grid(True)

        # Plot return distribution
        stats = self.results["statistics"]
        sns.histplot(data=stats, x="total_return", kde=True, ax=axes[0, 1])
        axes[0, 1].set_title("Distribution of Total Returns")
        axes[0, 1].axvline(
            stats["total_return"].mean(),
            color="r",
            linestyle="--",
            label=f'Mean: {stats["total_return"].mean():.2%}',
        )
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        # Plot Sharpe ratio distribution
        sns.histplot(data=stats, x="sharpe_ratio", kde=True, ax=axes[1, 0])
        axes[1, 0].set_title("Distribution of Sharpe Ratios")
        axes[1, 0].axvline(
            stats["sharpe_ratio"].mean(),
            color="r",
            linestyle="--",
            label=f'Mean: {stats["sharpe_ratio"].mean():.2f}',
        )
        axes[1, 0].grid(True)
        axes[1, 0].legend()

        # Print summary statistics
        summary_text = (
            f"Summary Statistics:\n"
            f"Average Return: {stats['total_return'].mean():.2%}\n"
            f"Return Std: {stats['total_return'].std():.2%}\n"
            f"Average Sharpe: {stats['sharpe_ratio'].mean():.2f}\n"
            f"Average Max Drawdown: {stats['max_drawdown'].mean():.2%}\n"
            f"Win Rate: {(stats['total_return'] > 0).mean():.1%}"
        )
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12)
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.show()


class ConservativeTradeSimulator(TradeSimulator):
    """A conservative trading strategy based on directional predictions with a given accuracy.
    It minimizes position changes and holds positions longer than the aggressive strategy.

    Conservative Strategy:
    The trader can be in these states:
    1. No position (0)
    2. Long 1 unit (+1)
    3. Short 1 unit (-1) [only if short_selling is allowed]

    Trading rules based on predictions:
    - Trading occurs every trading_interval days
    - At day T, the predictor predicts price direction at T + trading_interval
    - If predicted UP:
        - No position -> Buy 1 unit
        - Long -> Hold
        - Short -> Buy 1 unit (close position) [only if short_selling is allowed]
    - If predicted DOWN:
        - No position -> Sell 1 unit [only if short_selling is allowed, otherwise hold]
        - Long -> Sell 1 unit (close position)
        - Short -> Hold [only if short_selling is allowed]
    """

    def _execute_trade(self, curr_price: float, position: int, prediction: bool) -> tuple[int, float]:
        """Execute a trade based on the current position and prediction.
        
        This is the conservative implementation that minimizes position changes.
        
        Args:
            curr_price: Current price
            position: Current position (-1, 0, or 1)
            prediction: True for predicted up, False for predicted down
            
        Returns:
            Tuple of (new_position, capital_change)
        """
        capital_change = 0.0
        
        if prediction:  # Predicted Up
            if position == 0:  # No position -> Buy
                new_position = 1
                trade_price = self._calculate_trade_price(curr_price, is_buy=True)
                capital_change = -trade_price
            elif position == -1 and self.short_selling:  # Short -> Close position
                new_position = 0
                trade_price = self._calculate_trade_price(curr_price, is_buy=True)
                capital_change = -trade_price  # Buy to close short
            else:
                new_position = position  # Hold
        else:  # Predicted Down
            if position == 0 and self.short_selling:  # No position -> Short (only if allowed)
                new_position = -1
                trade_price = self._calculate_trade_price(curr_price, is_buy=False)
                capital_change = trade_price
            elif position == 1:  # Long -> Close position
                new_position = 0
                trade_price = self._calculate_trade_price(curr_price, is_buy=False)
                capital_change = trade_price  # Sell to close long
            else:
                new_position = position  # Hold
                
        return new_position, capital_change


class AggressiveTradeSimulator(TradeSimulator):
    """An aggressive trading strategy based on directional predictions with a given accuracy.
    It has more swings between long and short than the conservative strategy.

    Aggressive Strategy:
    The trader can be in these states:
    1. No position (0)
    2. Long 1 unit (+1)
    3. Short 1 unit (-1) [only if short_selling is allowed]

    Trading rules based on predictions:
    - Trading occurs every trading_interval days
    - At day T, the predictor predicts price direction at T + trading_interval
    - If predicted UP:
        - Always adjust position to +1 (long 1 unit)
    - If predicted DOWN:
        - Always adjust position to -1 (short 1 unit) [if short_selling is allowed]
        - Otherwise adjust position to 0 (no position)
    """

    def _execute_trade(self, curr_price: float, position: int, prediction: bool) -> tuple[int, float]:
        """Execute a trade based on the current position and prediction.
        
        This is the aggressive implementation that always adjusts to target position.
        
        Args:
            curr_price: Current price
            position: Current position (-1, 0, or 1)
            prediction: True for predicted up, False for predicted down
            
        Returns:
            Tuple of (new_position, capital_change)
        """
        # Determine target position based on prediction
        target_position = 1 if prediction else (-1 if self.short_selling else 0)
        
        # Calculate position change and update capital
        position_change = target_position - position
        capital_change = 0.0
        
        if position_change > 0:  # Need to buy
            trade_price = self._calculate_trade_price(curr_price, is_buy=True)
            capital_change = -position_change * trade_price
        elif position_change < 0:  # Need to sell
            trade_price = self._calculate_trade_price(curr_price, is_buy=False)
            capital_change = abs(position_change) * trade_price
            
        return target_position, capital_change


class ParameterAnalyzer:
    """A class for analyzing how different parameters affect trading performance."""

    def __init__(
        self,
        base_params: dict,
        param_ranges: dict[str, list],
        num_paths: int = 50,
        random_seed: int | None = None,
    ):
        """Initialize the parameter analyzer.

        Args:
            base_params: Dictionary of base parameters for simulation
            param_ranges: Dictionary mapping parameter names to lists of values to test
            num_paths: Number of paths to simulate for each parameter combination
            random_seed: Optional random seed for reproducibility
        """
        self.base_params = base_params
        self.param_ranges = param_ranges
        self.num_paths = num_paths
        self.random_seed = random_seed
        self.results: dict[str, pd.DataFrame] = {}

    def _run_single_simulation(self, params: dict) -> dict:
        """Run a single simulation with given parameters.

        Args:
            params: Dictionary of parameters for this simulation

        Returns:
            Dictionary containing simulation results
        """
        # Separate price simulation parameters from trade parameters
        price_params = {
            "mu": params.get("mu", 0.02),
            "sigma": params.get("sigma", 0.015),
        }

        # Update base parameters with current combination
        trade_params = self.base_params.copy()
        trade_params.update(
            {k: v for k, v in params.items() if k not in ["mu", "sigma"]}
        )

        # Generate price paths
        price_sim = BondPriceSimulator(
            S0=100,
            mu=price_params["mu"],
            sigma=price_params["sigma"],
            days=trade_params["days_of_year"],
            days_of_year=trade_params["days_of_year"],
            num_paths=self.num_paths,
            mode="discrete",
        )
        prices_df = price_sim.generate_prices()

        trade_params["prices_df"] = prices_df

        # Run both strategies
        conservative_sim = ConservativeTradeSimulator(**trade_params)
        aggressive_sim = AggressiveTradeSimulator(**trade_params)

        conservative_results = conservative_sim.simulate_trades(
            random_seed=self.random_seed
        )
        aggressive_results = aggressive_sim.simulate_trades(
            random_seed=self.random_seed
        )

        return {
            "params": params,
            "conservative_stats": conservative_results["statistics"],
            "aggressive_stats": aggressive_results["statistics"],
        }

    def analyze_parameters(self) -> dict[str, pd.DataFrame]:
        """Run simulations with different parameter combinations and analyze results.

        Returns:
            Dictionary containing analysis results and parameter combinations
        """
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations()
        total_combinations = len(param_combinations)
        print(f"Running {total_combinations} simulations...")

        # Use multiprocessing to run simulations concurrently
        num_processes = mp.cpu_count()  # Use all available CPU cores
        with mp.Pool(processes=num_processes) as pool:
            # Create a progress bar using tqdm
            from tqdm import tqdm
            all_results = list(tqdm(
                pool.imap(self._run_single_simulation, param_combinations),
                total=total_combinations,
                desc="Simulating"
            ))

        # Process results
        self.results = self._process_results(all_results)
        return self.results

    def _generate_param_combinations(self) -> list[dict]:
        """Generate all possible combinations of parameters."""

        # Create lists of parameter names and their values
        param_names = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())

        # Generate all combinations
        combinations = list(itertools.product(*param_values))

        # Convert to list of dictionaries
        return [dict(zip(param_names, combo)) for combo in combinations]

    def _process_results(self, all_results: list) -> dict[str, pd.DataFrame]:
        """Process simulation results into analysis DataFrames."""
        # Create DataFrames for each strategy
        conservative_data = []
        aggressive_data = []

        for result in all_results:
            params = result["params"]
            conservative_stats = result["conservative_stats"]
            aggressive_stats = result["aggressive_stats"]

            # Add parameter values to statistics
            for param, value in params.items():
                conservative_stats[param] = value
                aggressive_stats[param] = value

            conservative_data.append(conservative_stats)
            aggressive_data.append(aggressive_stats)

        # Combine results
        conservative_df = pd.concat(conservative_data, ignore_index=True)
        aggressive_df = pd.concat(aggressive_data, ignore_index=True)

        return {"conservative": conservative_df, "aggressive": aggressive_df}

    def plot_parameter_sensitivity(self) -> None:
        """Create plots showing parameter sensitivity analysis."""
        if not self.results:
            raise ValueError("No results available. Run analyze_parameters() first.")

        # Create subplots for each parameter
        n_params = len(self.param_ranges)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.ravel()

        for i, (param, values) in enumerate(self.param_ranges.items()):
            # Calculate mean returns for each parameter value
            conservative_means = []
            aggressive_means = []

            for value in values:
                conservative_mean = self.results["conservative"][
                    self.results["conservative"][param] == value
                ]["total_return"].mean()
                aggressive_mean = self.results["aggressive"][
                    self.results["aggressive"][param] == value
                ]["total_return"].mean()

                conservative_means.append(conservative_mean)
                aggressive_means.append(aggressive_mean)

            # Plot results
            axes[i].plot(values, conservative_means, "b-o", label="Conservative")
            axes[i].plot(values, aggressive_means, "r-o", label="Aggressive")
            axes[i].set_title(f"Sensitivity to {param}")
            axes[i].set_xlabel(param)
            axes[i].set_ylabel("Average Return")
            axes[i].grid(True)
            axes[i].legend()

        # Hide unused subplots if any
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    def print_summary_table(self) -> None:
        """Print a summary table of results for each parameter combination."""
        if not self.results:
            raise ValueError("No results available. Run analyze_parameters() first.")

        # Create summary DataFrame
        summary_data = []

        for param, values in self.param_ranges.items():
            for value in values:
                conservative_stats = self.results["conservative"][
                    self.results["conservative"][param] == value
                ]
                aggressive_stats = self.results["aggressive"][
                    self.results["aggressive"][param] == value
                ]

                summary_data.append(
                    {
                        "Parameter": param,
                        "Value": value,
                        "Conservative_Return": conservative_stats[
                            "total_return"
                        ].mean(),
                        "Conservative_Sharpe": conservative_stats[
                            "sharpe_ratio"
                        ].mean(),
                        "Aggressive_Return": aggressive_stats["total_return"].mean(),
                        "Aggressive_Sharpe": aggressive_stats["sharpe_ratio"].mean(),
                    }
                )

        summary_df = pd.DataFrame(summary_data)
        print("\nParameter Sensitivity Summary:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    # Initialize and run bond price simulation
    discrete_simulator = BondPriceSimulator(
        S0=100,  # Initial price
        mu=0.02,  # Annual expected return
        sigma=0.015,  # Annual volatility
        days=365,  # Trading days
        days_of_year=365,  # Use Calendar Day to generate price paths.
        num_paths=100,  # Number of simulation paths
        mode="discrete",  # Use discrete GBM as specified
    )

    # Generate price paths
    prices_df = discrete_simulator.generate_prices()

    # Common parameters for both simulators
    sim_params = {
        "prices_df": prices_df,
        "initial_capital": 110.0,
        "prediction_accuracy": 0.8,
        "risk_free_rate": 0.02,
        "days_of_year": discrete_simulator.days_of_year,
        "slippage": 0.0003,
        "short_selling": True,  # Enable short selling
        "trading_interval": 5,  # Trading interval defaults to 5 day
    }

    # Run simulations with short selling enabled
    print("\nResults with Short Selling Enabled:")

    conservative_trade_sim = ConservativeTradeSimulator(**sim_params)
    aggressive_trade_sim = AggressiveTradeSimulator(**sim_params)

    results1 = conservative_trade_sim.simulate_trades()
    results2 = aggressive_trade_sim.simulate_trades()

    print("\nStrategy 1 (Conservative) Results:")
    # conservative_trade_sim.plot_results()

    print("\nStrategy 2 (Aggressive) Results:")
    # aggressive_trade_sim.plot_results()

    # Run simulations with short selling disabled
    print("\nResults with Short Selling Disabled:")
    sim_params["short_selling"] = False

    conservative_trade_sim_no_short = ConservativeTradeSimulator(**sim_params)
    aggressive_trade_sim_no_short = AggressiveTradeSimulator(**sim_params)

    results3 = conservative_trade_sim_no_short.simulate_trades(random_seed=42)
    results4 = aggressive_trade_sim_no_short.simulate_trades(random_seed=42)

    print("\nStrategy 1 (Conservative, No Short) Results:")
    # conservative_trade_sim_no_short.plot_results()

    print("\nStrategy 2 (Aggressive, No Short) Results:")
    # aggressive_trade_sim_no_short.plot_results()

    # Print detailed statistics comparison
    def print_strategy_stats(name: str, stats: pd.DataFrame):
        print(f"\n{name}:")
        print(f"Number of paths: {len(stats)}")
        print(f"Average return: {stats['total_return'].mean():.2%}")
        print(f"Return std dev: {stats['total_return'].std():.2%}")
        print(f"Average Sharpe ratio: {stats['sharpe_ratio'].mean():.2f}")
        print(f"Average max drawdown: {stats['max_drawdown'].mean():.2%}")
        print(f"Win rate: {(stats['total_return'] > 0).mean():.1%}")
        print(f"Best return: {stats['total_return'].max():.2%}")
        print(f"Worst return: {stats['total_return'].min():.2%}")

    print("\nDetailed Statistics Comparison:")
    print_strategy_stats(
        "Strategy 1 (Conservative, With Short)", results1["statistics"]
    )
    print_strategy_stats("Strategy 2 (Aggressive, With Short)", results2["statistics"])
    print_strategy_stats("Strategy 1 (Conservative, No Short)", results3["statistics"])
    print_strategy_stats("Strategy 2 (Aggressive, No Short)", results4["statistics"])

    # Example usage of ParameterAnalyzer
    base_params = {
        "initial_capital": 110.0,
        "risk_free_rate": 0.02,
        "days_of_year": 365,
        "short_selling": False,
    }

    param_ranges = {
        "mu": [-0.10, -0.05, -0.01, 0.01, 0.05, 0.10],
        "sigma": [0.01, 0.05, 0.10, 0.20, 0.50],
        "prediction_accuracy": [0.6, 0.7, 0.8, 0.9, 1.0],
        "slippage": [0, 0.0002, 0.0003, 0.0004],
        "trading_interval": [1, 5, 10, 20],
    }

    analyzer = ParameterAnalyzer(
        base_params=base_params, param_ranges=param_ranges, num_paths=50, random_seed=42
    )

    # results = analyzer.analyze_parameters()
    # analyzer.plot_parameter_sensitivity()
    # analyzer.print_summary_table()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class InterestRateSimulator:
    """A class for simulating interest rate movements using various models.

    For MTC simulations, the interest rate is short rate. The overnight rate is recommended.

    """

    def __init__(
        self,
        r0: float = 0.043,  # Initial interest rate
        mu: float = 0.0,  # Mean reversion level (for mean-reverting models)
        sigma: float = 0.02,  # Volatility
        days: int = 365
        * 6,  # 5+1 years of simulation. 1 year buffer for discounting 5 year bond.
        days_of_year: int = 365,
        num_paths: int = 10000,
        model: str = "vasicek",  # "vasicek", "cir", "abm", "gbm"
        kappa: float = 0.1,  # Mean reversion speed
    ):
        """Initialize the interest rate simulator.

        Args:
            r0: Initial interest rate
            mu: Long-term mean (for mean-reverting models)
            sigma: Volatility parameter
            days: Number of days to simulate
            days_of_year: Days in a year (for time scaling)
            num_paths: Number of Monte Carlo paths
            model: Model type - "vasicek", "cir", "abm", or "gbm"
            kappa: Mean reversion speed (for mean-reverting models)
        """
        self.r0 = r0
        self.mu = mu
        self.sigma = sigma
        self.days = days
        self.num_paths = num_paths
        self.model = model
        self.kappa = kappa
        self.days_of_year = days_of_year
        self._dt = 1 / self.days_of_year
        self._sqrt_dt = np.sqrt(self._dt)
        self.rates_df: pd.DataFrame | None = None

    def generate_rates(self) -> pd.DataFrame:
        """Generate interest rate paths using the specified model."""

        rates = np.zeros((self.num_paths, self.days))
        rates[:, 0] = self.r0

        if self.model == "vasicek":
            # Vasicek model: dr = kappa*(mu - r)*dt + sigma*dW
            for t in range(1, self.days):
                dW = np.random.normal(0, 1, size=self.num_paths) * self._sqrt_dt
                rates[:, t] = (
                    rates[:, t - 1]
                    + self.kappa * (self.mu - rates[:, t - 1]) * self._dt
                    + self.sigma * dW
                )

        elif self.model == "cir":
            # Cox-Ingersoll-Ross model: dr = kappa*(mu - r)*dt + sigma*sqrt(r)*dW
            for t in range(1, self.days):
                dW = np.random.normal(0, 1, size=self.num_paths) * self._sqrt_dt
                sqrt_r = np.maximum(np.sqrt(np.abs(rates[:, t - 1])), 1e-8)
                rates[:, t] = (
                    rates[:, t - 1]
                    + self.kappa * (self.mu - rates[:, t - 1]) * self._dt
                    + self.sigma * sqrt_r * dW
                )
                # Ensure non-negative rates
                rates[:, t] = np.maximum(rates[:, t], 0.0001)

        elif self.model == "abm":
            # Arithmetic Brownian Motion: dr = mu*dt + sigma*dW
            for t in range(1, self.days):
                dW = np.random.normal(0, 1, size=self.num_paths) * self._sqrt_dt
                rates[:, t] = rates[:, t - 1] + self.mu * self._dt + self.sigma * dW
        
        elif self.model == "gbm":
            # Geometric Brownian Motion (continuous)
            for t in range(1, self.days):
                dW = np.random.normal(0, 1, size=self.num_paths) * self._sqrt_dt
                rates[:, t] = rates[:, t - 1] * np.exp(
                    (self.mu - 0.5 * self.sigma**2) * self._dt
                    + self.sigma * dW
                )
        
        else:
            raise ValueError(f"Unknown model: {self.model}")

        # Create DataFrame
        try:
            # Try to create a date index for plotting only for short simulations
            if self.days <= 10000:
                dates = pd.date_range(start="today", periods=self.days, freq="D")
                self.rates_df = pd.DataFrame(
                    rates.T,
                    index=dates,
                    columns=[f"path_{i+1}" for i in range(self.num_paths)],
                )
            else:
                # For very long simulations, use numerical index
                self.rates_df = pd.DataFrame(
                    rates.T,
                    index=np.arange(self.days),
                    columns=[f"path_{i+1}" for i in range(self.num_paths)],
                )
        except (OverflowError, pd._libs.tslibs.np_datetime.OutOfBoundsTimedelta):
            self.rates_df = pd.DataFrame(
                rates.T,
                index=np.arange(self.days),
                columns=[f"path_{i+1}" for i in range(self.num_paths)],
            )

        return self.rates_df

    def plot_rate_paths(self, max_paths: int = 50) -> None:
        """Plot interest rate paths."""
        if self.rates_df is None:
            raise ValueError("No rate data available. Run generate_rates() first.")

        plt.figure(figsize=(12, 6))
        paths_to_plot = min(max_paths, len(self.rates_df.columns))
        self.rates_df.iloc[:, :paths_to_plot].plot(
            title=f"Interest Rate Simulation - {self.model.upper()} Model",
            alpha=0.5,
            legend=False,
        )
        plt.xlabel("Date")
        plt.ylabel("Interest Rate")
        plt.grid(True)
        plt.show() 
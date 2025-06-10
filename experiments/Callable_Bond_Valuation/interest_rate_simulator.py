import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize


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
        model: str = "abm",  # "vasicek", "cir", "abm", "gbm"
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

    def calibrate_cir_model(
        self,
        historical_data: pd.DataFrame,
        date_col: str = "date",
        rate_col: str = "rate",
    ) -> dict[str, float]:
        """Calibrate CIR model parameters from historical yield data.

        The CIR model follows: dr = kappa * (theta - r) * dt + sigma * sqrt(r) * dW
        Where theta is the long-term mean (stored as self.mu in this class).

        Args:
            historical_data: DataFrame with historical yield data
            date_col: Name of the date column
            rate_col: Name of the rate column (should be in decimal form, e.g., 0.05 for 5%)

        Returns:
            Dictionary with calibrated parameters: {'kappa', 'theta', 'sigma', 'log_likelihood'}
        """
        # Ensure data is sorted by date in ascending order
        data = historical_data.copy().sort_values(date_col).reset_index(drop=True)

        # Convert rates to numpy array
        rates = data[rate_col].values

        # Check for zero or negative rates (CIR requires positive rates)
        if np.any(rates <= 0):
            raise ValueError(
                "CIR model requires strictly positive rates. Found zero or negative rates in data."
            )

        # Calculate time differences (assuming daily data, convert to years)
        if pd.api.types.is_datetime64_any_dtype(data[date_col]):
            dates = pd.to_datetime(data[date_col])
            dt_values = np.diff(dates).astype("timedelta64[D]").astype(float) / 365.25
        else:
            # Assume uniform daily spacing if dates are not datetime
            dt_values = np.full(len(rates) - 1, 1 / 365.25)

        print(
            f"Calibrating CIR model with {len(dt_values)} data points..."
        )

        def cir_log_likelihood(params: np.ndarray) -> float:
            """Calculate negative log-likelihood for CIR model."""
            kappa, theta, sigma = params

            # Parameter constraints - return large but finite penalty
            penalty = 1e6
            if kappa <= 0 or theta <= 0 or sigma <= 0:
                return penalty

            # Feller condition: 2*kappa*theta >= sigma^2 (with small tolerance)
            if (
                2 * kappa * theta < 0.9 * sigma**2
            ):  # Allow slight violation for numerical stability
                return penalty

            log_likelihood = 0.0
            valid_observations = 0

            # Adjust rates to match dt_values. The last rate is not used.
            for i in range(len(rates) - 1):
                r_t = rates[i]
                r_t1 = rates[i + 1]
                dt = dt_values[i] if i < len(dt_values) else dt_values[-1]

                # Ensure positive rates
                r_t = max(r_t, 1e-6)
                r_t1 = max(r_t1, 1e-6)

                # Avoid extreme parameter values
                kappa_dt = min(kappa * dt, 10)  # Cap to avoid exp overflow

                # CIR Euler approximation with numerical stability
                try:
                    exp_kappa_dt = np.exp(-kappa_dt)

                    # Mean of the transition
                    mean = r_t * exp_kappa_dt + theta * (1 - exp_kappa_dt)

                    # Variance of the transition
                    if kappa > 1e-8:  # Avoid division by very small kappa
                        var_term1 = (sigma**2 * r_t / kappa) * (
                            exp_kappa_dt - np.exp(-2 * kappa_dt)
                        )
                        var_term2 = (theta * sigma**2 / (2 * kappa)) * (
                            1 - exp_kappa_dt
                        ) ** 2
                        variance = var_term1 + var_term2
                    else:
                        # Fallback for very small kappa
                        variance = sigma**2 * r_t * dt

                    # Ensure positive variance with minimum threshold
                    variance = max(variance, 1e-8)

                    # Calculate log-likelihood contribution
                    residual = r_t1 - mean
                    log_likelihood_contrib = (
                        -0.5 * np.log(2 * np.pi * variance)
                        - 0.5 * (residual**2) / variance
                    )

                    # Check for numerical validity
                    if np.isfinite(log_likelihood_contrib):
                        log_likelihood += log_likelihood_contrib
                        valid_observations += 1

                except (OverflowError, ZeroDivisionError, FloatingPointError):
                    # Skip this observation if numerical issues arise
                    continue

            # Penalize if too few valid observations
            if valid_observations < len(rates) * 0.8:
                return penalty

            # Return negative log-likelihood (capped to avoid extreme values)
            result = -log_likelihood
            return min(result, penalty * 0.9)  # Cap to avoid infinite values

        # Initial parameter guesses
        rate_mean = np.mean(rates)
        rate_std = np.std(rates)

        # Method of moments initial estimates
        initial_kappa = 0.1
        initial_theta = rate_mean
        initial_sigma = rate_std * 2

        initial_params = np.array([initial_kappa, initial_theta, initial_sigma])

        # Parameter bounds
        bounds = [
            (0.001, 2.0),  # kappa: mean reversion speed
            (0.001, 0.2),  # theta: long-term mean
            (0.001, 0.5),  # sigma: volatility
        ]

        print(
            f"Initial parameter estimates: kappa={initial_kappa:.4f}, theta={initial_theta:.4f}, sigma={initial_sigma:.4f}"
        )

        # Optimize using multiple starting points for robustness
        best_result = None
        best_likelihood = np.inf

        # Try multiple optimization methods and starting points
        methods = ["L-BFGS-B", "TNC"]
        np.random.seed(42)  # For reproducible results

        for method in methods:
            for attempt in range(5):  # More attempts for better convergence
                try:
                    # Generate diverse starting points
                    if attempt == 0:
                        # Use initial method-of-moments estimates
                        start_params = initial_params.copy()
                    else:
                        # Random perturbation of initial guess
                        perturbation = 1 + 0.3 * (
                            np.random.rand(3) - 0.5
                        )  # ±15% variation
                        start_params = initial_params * perturbation
                        start_params = np.clip(
                            start_params, [b[0] for b in bounds], [b[1] for b in bounds]
                        )

                    # Optimization options (method-specific)
                    if method == "L-BFGS-B":
                        options = {"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-6}
                    elif method == "TNC":
                        options = {"maxfun": 3000, "ftol": 1e-9, "gtol": 1e-6}

                    result = optimize.minimize(
                        cir_log_likelihood,
                        start_params,
                        method=method,
                        bounds=bounds,
                        options=options,
                    )

                    # Check if this is the best result so far
                    if (
                        result.success
                        and np.isfinite(result.fun)
                        and result.fun < best_likelihood
                    ):
                        # Additional validation: ensure parameters make sense
                        kappa_opt, theta_opt, sigma_opt = result.x
                        if (
                            kappa_opt > 0
                            and theta_opt > 0
                            and sigma_opt > 0
                            and 2 * kappa_opt * theta_opt >= 0.8 * sigma_opt**2
                        ):
                            best_result = result
                            best_likelihood = result.fun
                            print(
                                f"  Improved result found with {method}, attempt {attempt + 1}"
                            )

                except Exception as e:
                    print(
                        f"  Optimization attempt {attempt + 1} with {method} failed: {e}"
                    )
                    continue

        if best_result is None or not best_result.success:
            # Fallback to method of moments if optimization fails
            print("  Optimization failed, using method of moments estimates...")

            # Enhanced method of moments for CIR
            dt_mean = np.mean(dt_values)
            rate_changes = np.diff(rates)

            # Calculate sample statistics
            mean_rate = np.mean(rates[:-1])
            mean_change = np.mean(rate_changes)
            var_change = np.var(rate_changes)
            var_rate = np.var(rates[:-1])

            # Method of moments estimates with better numerical stability
            try:
                # Estimate mean reversion parameters
                autocorr = (
                    np.corrcoef(rates[:-1], rates[1:])[0, 1] if len(rates) > 2 else 0.9
                )
                autocorr = max(0.1, min(0.99, autocorr))  # Ensure reasonable range

                # Estimate kappa from autocorrelation
                kappa_est = -np.log(autocorr) / dt_mean
                kappa_est = max(0.01, min(2.0, kappa_est))

                # Estimate theta (long-term mean)
                theta_est = mean_rate
                theta_est = max(0.001, min(0.2, theta_est))

                # Estimate sigma from residual variance
                # For CIR: Var[dr] ≈ sigma² * r * dt
                if mean_rate > 0:
                    sigma_est = np.sqrt(var_change / (mean_rate * dt_mean))
                else:
                    sigma_est = np.sqrt(var_change / dt_mean)

                sigma_est = max(0.001, min(0.5, sigma_est))

                # Verify Feller condition and adjust if necessary
                if 2 * kappa_est * theta_est < sigma_est**2:
                    print(f"  Adjusting sigma to satisfy Feller condition...")
                    sigma_est = np.sqrt(
                        1.5 * kappa_est * theta_est
                    )  # Ensure Feller condition
                    sigma_est = max(0.001, min(0.5, sigma_est))

            except Exception as e:
                print(
                    f"  Method of moments calculation failed: {e}, using default values..."
                )
                kappa_est = 0.1
                theta_est = mean_rate if mean_rate > 0 else 0.03
                sigma_est = 0.02

            calibrated_params = {
                "kappa": kappa_est,
                "theta": theta_est,
                "sigma": sigma_est,
                "log_likelihood": -cir_log_likelihood(
                    [kappa_est, theta_est, sigma_est]
                ),
            }
        else:
            kappa_opt, theta_opt, sigma_opt = best_result.x
            calibrated_params = {
                "kappa": kappa_opt,
                "theta": theta_opt,
                "sigma": sigma_opt,
                "log_likelihood": -best_result.fun,
            }

        # Update instance parameters
        self.kappa = calibrated_params["kappa"]
        self.mu = calibrated_params["theta"]  # Note: mu is used as theta in this class
        self.sigma = calibrated_params["sigma"]
        self.model = "cir"

        print(f"Calibrated CIR parameters:")
        print(f"  kappa (mean reversion speed): {calibrated_params['kappa']:.6f}")
        print(f"  theta (long-term mean): {calibrated_params['theta']:.6f}")
        print(f"  sigma (volatility): {calibrated_params['sigma']:.6f}")
        print(f"  Log-likelihood: {calibrated_params['log_likelihood']:.2f}")

        # Verify Feller condition
        feller_condition = 2 * calibrated_params["kappa"] * calibrated_params["theta"]
        if feller_condition >= calibrated_params["sigma"] ** 2:
            print(
                f"  Feller condition satisfied: 2κθ = {feller_condition:.6f} >= σ² = {calibrated_params['sigma']**2:.6f}"
            )
        else:
            print(
                f"  WARNING: Feller condition violated: 2κθ = {feller_condition:.6f} < σ² = {calibrated_params['sigma']**2:.6f}"
            )
            print("  This may lead to negative interest rates in simulation.")

        return calibrated_params

    def load_and_calibrate_from_csv(
        self,
        csv_path: str,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
        date_col: str = "date",
        rate_col: str = "rate",
    ) -> dict[str, float]:
        """Load historical yield data from CSV and calibrate CIR model.

        Args:
            csv_path: Path to CSV file containing historical yield data
            start_date: Start date of the data. If None, use the first date in the data.
            end_date: End date of the data. If None, use the last date in the data.
            date_col: Name of the date column in CSV
            rate_col: Name of the rate column in CSV

        Returns:
            Dictionary with calibrated parameters

        Example:
            # Your CSV should look like:
            # date,rate
            # 2020-01-01,0.0150
            # 2020-01-02,0.0152
            # ...

            simulator = InterestRateSimulator(model="cir")
            params = simulator.load_and_calibrate_from_csv("5yr_treasury_yields.csv")
        """
        try:
            # Load data from CSV
            historical_data = pd.read_csv(csv_path, parse_dates=[date_col])

            if start_date:
                historical_data = historical_data[
                    historical_data[date_col] >= start_date
                ]
            if end_date:
                historical_data = historical_data[historical_data[date_col] <= end_date]

            # Basic data validation
            print(f"Loaded {len(historical_data)} records")
            print(
                f"Date range: {historical_data[date_col].min()} to {historical_data[date_col].max()}"
            )
            print(f"Rate statistics:")
            print(f"  Mean: {historical_data[rate_col].mean():.4f}")
            print(f"  Std:  {historical_data[rate_col].std():.4f}")
            print(f"  Min:  {historical_data[rate_col].min():.4f}")
            print(f"  Max:  {historical_data[rate_col].max():.4f}")
            print()

            # Check for missing values
            missing_dates = historical_data[date_col].isna().sum()
            missing_rates = historical_data[rate_col].isna().sum()

            if missing_dates > 0:
                print(f"Warning: {missing_dates} missing dates found")
                historical_data = historical_data.dropna(subset=[date_col])

            if missing_rates > 0:
                print(f"Warning: {missing_rates} missing rates found")
                historical_data = historical_data.dropna(subset=[rate_col])

            # Calibrate model
            return self.calibrate_cir_model(historical_data, date_col, rate_col)

        except FileNotFoundError:
            print(f"Error: File not found: {csv_path}")
            print("Please make sure the file path is correct.")
            raise
        except KeyError as e:
            print(f"Error: Column not found in CSV: {e}")
            print(
                f"Available columns: {list(historical_data.columns) if 'historical_data' in locals() else 'Unable to read CSV'}"
            )
            raise
        except Exception as e:
            print(f"Error loading/calibrating data: {e}")
            raise

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
                    (self.mu - 0.5 * self.sigma**2) * self._dt + self.sigma * dW
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

        fig, ax = plt.subplots(figsize=(12, 6))
        paths_to_plot = min(max_paths, len(self.rates_df.columns))
        self.rates_df.iloc[:, :paths_to_plot].plot(
            ax=ax,  # Pass the axes to pandas plot
            title=f"Interest Rate Simulation - {self.model.upper()} Model",
            alpha=0.5,
            legend=False,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Interest Rate")
        ax.grid(True)
        plt.show()


def main():
    # Initialize simulator
    simulator = InterestRateSimulator(model="cir")

    # Load CSV file and calibrate CIR model. CSV format example:
    # date,rate
    # 2020-01-01,0.0150
    simulator.load_and_calibrate_from_csv(
        csv_path="experiments/Callable_Bond_Valuation/5y_treasury_2020_2025.csv",
        start_date=datetime.datetime(2024, 1, 1),
        end_date=datetime.datetime(2025, 5, 27),
    )

    # Generate and plot some paths with calibrated parameters
    print("=== Generating Rate Paths with Calibrated Parameters ===")
    simulator.days = 365 * 5  # 5 years
    simulator.num_paths = 1000

    rates_df = simulator.generate_rates()
    print(f"Generated {len(rates_df.columns)} rate paths for {len(rates_df)} days")

    # Show summary statistics of generated paths
    print(f"Generated rate statistics:")
    print(f"  Mean: {rates_df.mean().mean():.4f}")
    print(f"  Std:  {rates_df.std().mean():.4f}")
    print(f"  Min:  {rates_df.min().min():.4f}")
    print(f"  Max:  {rates_df.max().max():.4f}")

    # Plot first few paths (uncomment to see plots)
    simulator.plot_rate_paths(max_paths=50)


if __name__ == "__main__":
    main()

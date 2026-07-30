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
        seed: int | None = None,  # Random seed for reproducibility
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
            seed: Random seed for reproducible paths (None = non-deterministic)
        """
        self.r0 = r0
        self.mu = mu
        self.sigma = sigma
        self.days = days
        self.num_paths = num_paths
        self.model = model
        self.kappa = kappa
        self.days_of_year = days_of_year
        self.seed = seed
        self._dt = 1 / self.days_of_year
        self._sqrt_dt = np.sqrt(self._dt)
        self._rng: np.random.Generator | None = None
        self.rates_df: pd.DataFrame | None = None

    def calibrate_model(
        self,
        historical_data: pd.DataFrame,
        model: str = "cir",
        date_col: str = "date",
        rate_col: str = "rate",
    ) -> dict[str, float]:
        """Calibrate CIR or Vasicek model parameters from historical yield data.

        Both models share the affine mean-reverting SDE form
            dr = kappa * (theta - r) * dt + sigma * f(r) * dW
        where
            CIR:     f(r) = sqrt(r)  (rates stay positive; Feller condition applies)
            Vasicek: f(r) = 1        (rates may turn negative; no Feller condition)

        `theta` (long-term mean) is stored on the instance as ``self.mu``.

        Args:
            historical_data: DataFrame with historical yield data
            model: "cir" or "vasicek"
            date_col: Name of the date column
            rate_col: Name of the rate column (decimal form, e.g. 0.05 for 5%)

        Returns:
            Dictionary with calibrated parameters: {'kappa', 'theta', 'sigma', 'log_likelihood'}
        """
        if model not in ("cir", "vasicek"):
            raise ValueError(f"Unsupported model '{model}'. Use 'cir' or 'vasicek'.")

        # Ensure data is sorted by date in ascending order
        data = historical_data.copy().sort_values(date_col).reset_index(drop=True)

        # Convert rates to numpy array
        rates = data[rate_col].values

        # Check for zero or negative rates (CIR requires positive rates; Vasicek tolerates them)
        if model == "cir" and np.any(rates <= 0):
            raise ValueError(
                "CIR model requires strictly positive rates. Found zero or negative "
                "rates in data. Filter the data or use model='vasicek'."
            )

        # Calculate time differences (assuming daily data, convert to years)
        if pd.api.types.is_datetime64_any_dtype(data[date_col]):
            dates = pd.to_datetime(data[date_col])
            dt_values = np.diff(dates).astype("timedelta64[D]").astype(float) / 365.25
        else:
            # Assume uniform daily spacing if dates are not datetime
            dt_values = np.full(len(rates) - 1, 1 / 365.25)

        print(f"Calibrating {model.upper()} model with {len(dt_values)} data points...")

        def _conditional_moments(
            r_t: float, kappa: float, theta: float, sigma: float, dt: float
        ) -> tuple[float, float]:
            """Return (mean, variance) of r_{t+dt} conditional on r_t.

            Uses the exact discrete-time transition for both models. The CIR
            variance here is the exact non-central chi-square Gaussian
            approximation (mean = r_t*exp(-k*dt) + theta*(1-exp(-k*dt)),
            variance mixing terms from r_t and theta); Vasicek has a closed-form
            Normal transition.
            """
            kappa_dt = min(kappa * dt, 10.0)  # cap to avoid exp overflow
            exp_k = np.exp(-kappa_dt)
            mean = r_t * exp_k + theta * (1.0 - exp_k)

            if model == "vasicek":
                if kappa > 1e-8:
                    variance = (sigma**2 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa_dt))
                else:
                    variance = sigma**2 * dt
            else:  # cir
                if kappa > 1e-8:
                    variance = (sigma**2 * r_t / kappa) * (
                        exp_k - np.exp(-2.0 * kappa_dt)
                    ) + (theta * sigma**2 / (2.0 * kappa)) * (1.0 - exp_k) ** 2
                else:
                    variance = sigma**2 * r_t * dt

            variance = max(variance, 1e-8)
            return mean, variance

        def neg_log_likelihood(params: np.ndarray) -> float:
            """Negative log-likelihood for either CIR or Vasicek."""
            kappa, theta, sigma = params

            penalty = 1e6
            if kappa <= 0 or theta <= 0 or sigma <= 0:
                return penalty

            # Feller condition applies ONLY to CIR. Allow slight violation for stability.
            if model == "cir" and 2 * kappa * theta < 0.9 * sigma**2:
                return penalty

            log_likelihood = 0.0
            valid_observations = 0

            for i in range(len(rates) - 1):
                r_t = rates[i]
                r_t1 = rates[i + 1]
                dt = dt_values[i] if i < len(dt_values) else dt_values[-1]

                # CIR requires positive rates for the sqrt(r) term
                if model == "cir":
                    r_t = max(r_t, 1e-6)
                    r_t1 = max(r_t1, 1e-6)

                try:
                    mean, variance = _conditional_moments(r_t, kappa, theta, sigma, dt)

                    residual = r_t1 - mean
                    log_likelihood_contrib = (
                        -0.5 * np.log(2 * np.pi * variance)
                        - 0.5 * (residual**2) / variance
                    )

                    if np.isfinite(log_likelihood_contrib):
                        log_likelihood += log_likelihood_contrib
                        valid_observations += 1

                except (OverflowError, ZeroDivisionError, FloatingPointError):
                    continue

            if valid_observations < len(rates) * 0.8:
                return penalty

            result = -log_likelihood
            return min(result, penalty * 0.9)  # Cap to avoid infinite values

        # Initial parameter guesses
        rate_mean = float(np.mean(rates))
        rate_std = float(np.std(rates))

        initial_kappa = 0.1
        initial_theta = max(rate_mean, 0.001) if rate_mean > 0 else 0.03
        initial_sigma = max(rate_std * 2, 0.001)

        initial_params = np.array([initial_kappa, initial_theta, initial_sigma])

        # Parameter bounds
        bounds = [
            (0.001, 5.0),  # kappa: mean reversion speed
            (0.001, 0.07),  # theta: long-term mean
            (0.001, 0.5),  # sigma: volatility
        ]

        print(
            f"Initial parameter estimates: kappa={initial_kappa:.4f}, "
            f"theta={initial_theta:.4f}, sigma={initial_sigma:.4f}"
        )

        # Optimize using multiple starting points for robustness
        best_result = None
        best_likelihood = np.inf

        methods = ["L-BFGS-B", "TNC"]
        np.random.seed(42)  # For reproducible results

        for method in methods:
            for attempt in range(5):  # More attempts for better convergence
                try:
                    if attempt == 0:
                        start_params = initial_params.copy()
                    else:
                        perturbation = 1 + 0.3 * (np.random.rand(3) - 0.5)
                        start_params = initial_params * perturbation
                        start_params = np.clip(
                            start_params,
                            [b[0] for b in bounds],
                            [b[1] for b in bounds],
                        )

                    if method == "L-BFGS-B":
                        options = {"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-6}
                    elif method == "TNC":
                        options = {"maxfun": 3000, "ftol": 1e-9, "gtol": 1e-6}

                    result = optimize.minimize(
                        neg_log_likelihood,
                        start_params,
                        method=method,
                        bounds=bounds,
                        options=options,
                    )

                    if (
                        result.success
                        and np.isfinite(result.fun)
                        and result.fun < best_likelihood
                    ):
                        kappa_opt, theta_opt, sigma_opt = result.x
                        if kappa_opt > 0 and theta_opt > 0 and sigma_opt > 0:
                            # Only CIR needs the Feller sanity check
                            if model == "vasicek" or (
                                2 * kappa_opt * theta_opt >= 0.8 * sigma_opt**2
                            ):
                                best_result = result
                                best_likelihood = result.fun
                                print(
                                    f"  Improved result found with {method}, "
                                    f"attempt {attempt + 1}"
                                )

                except Exception as e:
                    print(
                        f"  Optimization attempt {attempt + 1} with {method} failed: {e}"
                    )
                    continue

        if best_result is None or not best_result.success:
            # Fallback to method of moments if optimization fails
            print("  Optimization failed, using method of moments estimates...")

            dt_mean = np.mean(dt_values)
            rate_changes = np.diff(rates)

            mean_rate = np.mean(rates[:-1])
            var_change = np.var(rate_changes)

            try:
                autocorr = (
                    np.corrcoef(rates[:-1], rates[1:])[0, 1]
                    if len(rates) > 2
                    else 0.9
                )
                autocorr = max(0.1, min(0.99, autocorr))

                kappa_est = -np.log(autocorr) / dt_mean
                kappa_est = max(0.01, min(2.0, kappa_est))

                theta_est = mean_rate if mean_rate > 0 else 0.03
                theta_est = max(0.001, min(0.2, theta_est))

                # Estimate sigma from residual variance:
                #   CIR:     Var[dr] ~ sigma^2 * r * dt  -> sigma = sqrt(Var[dr]/(r_mean*dt))
                #   Vasicek: Var[dr] ~ sigma^2 * dt       -> sigma = sqrt(Var[dr]/dt)
                if model == "cir":
                    sigma_est = (
                        np.sqrt(var_change / (mean_rate * dt_mean))
                        if mean_rate > 0
                        else np.sqrt(var_change / dt_mean)
                    )
                    if 2 * kappa_est * theta_est < sigma_est**2:
                        print("  Adjusting sigma to satisfy Feller condition...")
                        sigma_est = np.sqrt(1.5 * kappa_est * theta_est)
                else:  # vasicek
                    sigma_est = np.sqrt(var_change / dt_mean)

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
                "log_likelihood": -neg_log_likelihood(
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
        self.model = model

        print(f"Calibrated {model.upper()} parameters:")
        print(f"  kappa (mean reversion speed): {calibrated_params['kappa']:.6f}")
        print(f"  theta (long-term mean): {calibrated_params['theta']:.6f}")
        print(f"  sigma (volatility): {calibrated_params['sigma']:.6f}")
        print(f"  Log-likelihood: {calibrated_params['log_likelihood']:.2f}")

        # Verify Feller condition (only for CIR; Vasicek tolerates negative rates)
        if model == "cir":
            feller_condition = 2 * calibrated_params["kappa"] * calibrated_params["theta"]
            if feller_condition >= calibrated_params["sigma"] ** 2:
                print(
                    f"  Feller condition satisfied: 2*kappa*theta = {feller_condition:.6f} "
                    f">= sigma^2 = {calibrated_params['sigma']**2:.6f}"
                )
            else:
                print(
                    f"  WARNING: Feller condition violated: 2*kappa*theta = {feller_condition:.6f} "
                    f"< sigma^2 = {calibrated_params['sigma']**2:.6f}"
                )
                print("  This may lead to negative interest rates in simulation.")
        else:  # vasicek
            print("  Feller condition not applicable (Vasicek allows negative rates).")

        return calibrated_params

    def calibrate_cir_model(
        self,
        historical_data: pd.DataFrame,
        date_col: str = "date",
        rate_col: str = "rate",
    ) -> dict[str, float]:
        """Backward-compatible wrapper for calibrate_model(model='cir').

        New code should call calibrate_model(data, model='cir' | 'vasicek') directly.
        """
        return self.calibrate_model(
            historical_data, model="cir", date_col=date_col, rate_col=rate_col
        )

    def load_and_calibrate_from_csv(
        self,
        csv_path: str,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
        model: str = "cir",
        date_col: str = "date",
        rate_col: str = "rate",
    ) -> dict[str, float]:
        """Load historical yield data from CSV and calibrate a CIR or Vasicek model.

        Args:
            csv_path: Path to CSV file containing historical yield data
            start_date: Start date of the data. If None, use the first date in the data.
            end_date: End date of the data. If None, use the last date in the data.
            model: Model to calibrate - "cir" (default) or "vasicek". CIR requires
                strictly positive rates; Vasicek tolerates negative rates.
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

            # Vasicek
            simulator = InterestRateSimulator(model="vasicek")
            params = simulator.load_and_calibrate_from_csv(
                "sofr_2020_2025.csv", model="vasicek"
            )
        """
        if model not in ("cir", "vasicek"):
            raise ValueError(f"Unsupported model '{model}'. Use 'cir' or 'vasicek'.")

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

            # Calibrate the requested model
            return self.calibrate_model(
                historical_data, model=model, date_col=date_col, rate_col=rate_col
            )

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

        # Use a dedicated Generator so results are reproducible when self.seed is set.
        # Re-creating the Generator on every call ensures two calls with the same
        # seed produce identical paths (useful for fair before/after comparisons).
        self._rng = np.random.default_rng(self.seed)

        if self.model == "vasicek":
            # Vasicek model: dr = kappa*(mu - r)*dt + sigma*dW
            for t in range(1, self.days):
                dW = self._rng.normal(0, 1, size=self.num_paths) * self._sqrt_dt
                rates[:, t] = (
                    rates[:, t - 1]
                    + self.kappa * (self.mu - rates[:, t - 1]) * self._dt
                    + self.sigma * dW
                )

        elif self.model == "cir":
            # Cox-Ingersoll-Ross model: dr = kappa*(mu - r)*dt + sigma*sqrt(r)*dW
            for t in range(1, self.days):
                dW = self._rng.normal(0, 1, size=self.num_paths) * self._sqrt_dt
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
                dW = self._rng.normal(0, 1, size=self.num_paths) * self._sqrt_dt
                rates[:, t] = rates[:, t - 1] + self.mu * self._dt + self.sigma * dW

        elif self.model == "gbm":
            # Geometric Brownian Motion (continuous)
            for t in range(1, self.days):
                dW = self._rng.normal(0, 1, size=self.num_paths) * self._sqrt_dt
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


def calculate_history_stats(
    csv_path: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    date_col: str = "date",
    rate_col: str = "rate",
) -> dict[str, float]:
    trade_day_df = pd.read_csv(csv_path, parse_dates=[date_col])
    print(f"Loaded {len(trade_day_df)} records")
    trade_day_df = trade_day_df[
        (trade_day_df[date_col] >= start_date) & (trade_day_df[date_col] <= end_date)
    ]

    """Statistics of historical yield data in trade days."""
    trade_day_df["dr"] = trade_day_df[rate_col].diff()
    trade_day_df = trade_day_df.dropna()
    years = ((trade_day_df[date_col].max() - trade_day_df[date_col].min()).days + 1) / 365
    days_per_year = len(trade_day_df) / years

    # Calculate trade day statistics
    trade_day_stats = {}
    trade_day_stats["days_per_year"] = days_per_year
    trade_day_stats["mean"] = trade_day_df[rate_col].mean()
    trade_day_stats["std"] = trade_day_df[rate_col].std()
    trade_day_stats["min"] = trade_day_df[rate_col].min()
    trade_day_stats["max"] = trade_day_df[rate_col].max()
    trade_day_stats["dr_mean"] = trade_day_df["dr"].mean()
    trade_day_stats["dr_std"] = trade_day_df["dr"].std()
    trade_day_stats["dr_std_year"] = trade_day_stats["dr_std"] * np.sqrt(days_per_year)
    trade_day_stats["dr_min"] = trade_day_df["dr"].min()
    trade_day_stats["dr_max"] = trade_day_df["dr"].max()

    # Create new dataframe with holidays filled using last trade day's rates
    print(f"\nCreating calendar day dataframe with holidays filled...")

    # Create complete date range (including holidays/weekends)
    complete_date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Set date column as index for easier reindexing
    trade_day_indexed = trade_day_df.set_index(date_col)

    # Reindex to complete date range and forward fill missing values
    calendar_day_df = trade_day_indexed.reindex(complete_date_range)
    calendar_day_df[rate_col] = calendar_day_df[rate_col].ffill()

    # Reset index to get date column back
    calendar_day_df = calendar_day_df.reset_index()
    calendar_day_df.rename(columns={"index": date_col}, inplace=True)

    # Calculate dr for calendar day dataframe
    calendar_day_df["dr"] = calendar_day_df[rate_col].diff()
    calendar_day_df = calendar_day_df.dropna()
    years = ((calendar_day_df[date_col].max() - calendar_day_df[date_col].min()).days + 1) / 365
    calendar_days_per_year = len(calendar_day_df) / years

    # Calculate calendar day statistics
    calendar_day_stats = {}
    calendar_day_stats["days_per_year"] = calendar_days_per_year
    calendar_day_stats["mean"] = calendar_day_df[rate_col].mean()
    calendar_day_stats["std"] = calendar_day_df[rate_col].std()
    calendar_day_stats["min"] = calendar_day_df[rate_col].min()
    calendar_day_stats["max"] = calendar_day_df[rate_col].max()
    calendar_day_stats["dr_mean"] = calendar_day_df["dr"].mean()
    calendar_day_stats["dr_std"] = calendar_day_df["dr"].std()
    calendar_day_stats["dr_std_year"] = calendar_day_stats["dr_std"] * np.sqrt(
        calendar_days_per_year
    )
    calendar_day_stats["dr_min"] = calendar_day_df["dr"].min()
    calendar_day_stats["dr_max"] = calendar_day_df["dr"].max()

    # Print statistics comparison
    print(f"\n=== TRADE DAY STATISTICS ===")
    print(f"Number of observations: {len(trade_day_df)}")
    print(f"Days per year: {trade_day_stats['days_per_year']:.1f}")
    print(f"Rate statistics:")
    print(f"  Mean: {trade_day_stats['mean']:.2%}")
    print(f"  Std:  {trade_day_stats['std']:.2%}")
    print(f"  Min:  {trade_day_stats['min']:.2%}")
    print(f"  Max:  {trade_day_stats['max']:.2%}")
    print(f"Daily rate change (dr) statistics:")
    print(f"  Mean: {trade_day_stats['dr_mean']:.2%}")
    print(f"  Std:  {trade_day_stats['dr_std']:.2%}")
    print(f"  Std (annualized): {trade_day_stats['dr_std_year']:.2%}")
    print(f"  Min:  {trade_day_stats['dr_min']:.2%}")
    print(f"  Max:  {trade_day_stats['dr_max']:.2%}")

    print(f"\n=== CALENDAR DAY STATISTICS (with holidays filled) ===")
    print(f"Number of observations: {len(calendar_day_df)}")
    print(f"Days per year: {calendar_day_stats['days_per_year']:.1f}")
    print(f"Rate statistics:")
    print(f"  Mean: {calendar_day_stats['mean']:.2%}")
    print(f"  Std:  {calendar_day_stats['std']:.2%}")
    print(f"  Min:  {calendar_day_stats['min']:.2%}")
    print(f"  Max:  {calendar_day_stats['max']:.2%}")
    print(f"Daily rate change (dr) statistics:")
    print(f"  Mean: {calendar_day_stats['dr_mean']:.2%}")
    print(f"  Std:  {calendar_day_stats['dr_std']:.2%}")
    print(f"  Std (annualized): {calendar_day_stats['dr_std_year']:.2%}")
    print(f"  Min:  {calendar_day_stats['dr_min']:.2%}")
    print(f"  Max:  {calendar_day_stats['dr_max']:.2%}")

    print(f"\n=== COMPARISON ===")
    print(
        f"Additional days from filling holidays: {len(calendar_day_df) - len(trade_day_df)}"
    )
    print(
        f"Trade day volatility vs Calendar day volatility: {trade_day_stats['dr_std']:.2%} vs {calendar_day_stats['dr_std']:.2%}"
    )
    print(
        f"Annualized volatility ratio (trade/calendar): {trade_day_stats['dr_std_year']/calendar_day_stats['dr_std_year']:.3f}"
    )

    try:
        from ydata_profiling import ProfileReport

        # Generate profile for trade day data
        trade_day_profile = ProfileReport(trade_day_df, title="Trade Day Data Profile")
        trade_day_profile.to_file("experiments/Callable_Bond_Valuation/results/trade_day_profile.html")
        print("\nTrade day profile saved to trade_day_profile.html")

        # Generate profile for calendar day data
        calendar_day_profile = ProfileReport(
            calendar_day_df, title="Calendar Day Data Profile"
        )
        calendar_day_profile.to_file(
            "experiments/Callable_Bond_Valuation/results/calendar_day_profile.html"
        )
        print("Calendar day profile saved to calendar_day_profile.html")

    except ImportError:
        print(
            "\nWarning: ydata_profiling not installed. Install with: pip install ydata-profiling"
        )
    except Exception as e:
        print(f"\nError generating profiles: {e}")

    return trade_day_stats, calendar_day_stats


def main():
    # Initialize simulator
    simulator = InterestRateSimulator(model="cir")

    # Load CSV file and calibrate CIR model. CSV format example:
    # date,rate
    # 2020-01-01,0.0150
    simulator.load_and_calibrate_from_csv(
        csv_path="experiments/Callable_Bond_Valuation/sofr_2020_2025.csv",
        start_date=datetime.datetime(2020, 1, 1),
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

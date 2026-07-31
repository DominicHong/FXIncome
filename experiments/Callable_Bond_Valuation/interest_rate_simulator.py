import datetime
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
        """Calibrate model parameters from historical yield data.

        Supported models:
            CIR:     dr = kappa*(theta - r)*dt + sigma*sqrt(r)*dW
                     (positive rates; Feller condition applies)
            Vasicek: dr = kappa*(theta - r)*dt + sigma*dW
                     (may turn negative)
            ABM:     dr = mu*dt + sigma*dW
                     (constant absolute vol; sigma same units as rate)
            GBM:     dr = mu*r*dt + sigma*r*dW
                     (constant proportional vol; sigma dimensionless)

        For CIR/Vasicek `theta` (long-term mean) is stored as ``self.mu``
        (for Vasicek, θ is a P-measure estimate typically overwritten by
        Q-measure re-anchoring).
        For ABM/GBM ``self.mu`` is the drift parameter.
        CIR uses MLE with multi-start optimisation; Vasicek uses OLS
        regression (Δr ~ r); ABM/GBM use closed-form MLE.

        Args:
            historical_data: DataFrame with historical yield data
            model: "cir" (default), "vasicek", "abm" or "gbm"
            date_col: Name of the date column
            rate_col: Name of the rate column (decimal form, e.g. 0.05 for 5%)

        Returns:
            Dictionary with calibrated parameters.
            CIR/Vasicek: {'kappa', 'theta', 'sigma', 'log_likelihood'}
            ABM/GBM:     {'mu', 'sigma', 'log_likelihood'}
        """
        if model not in ("cir", "vasicek", "abm", "gbm"):
            raise ValueError(f"Unsupported model '{model}'. Use 'cir', 'vasicek', 'abm' or 'gbm'.")

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

        if model in ("abm", "gbm"):
            dr = np.diff(rates)

            if model == "gbm" and np.any(rates <= 0):
                raise ValueError(
                    "GBM model requires strictly positive rates. Found zero or "
                    "negative rates in data. Filter the data or use a different "
                    "model."
                )

            print(
                f"Calibrating {model.upper()} model with {len(dt_values)} "
                f"data points..."
            )

            if model == "abm":
                mu_hat = np.sum(dr) / np.sum(dt_values)
                residuals = dr - mu_hat * dt_values
            else:
                log_returns = np.log(rates[1:] / rates[:-1])
                mu_hat = np.sum(log_returns) / np.sum(dt_values)
                residuals = log_returns - mu_hat * dt_values

            sigma_sq_hat = np.mean(residuals**2 / dt_values)
            sigma_hat = np.sqrt(sigma_sq_hat)

            if model == "gbm":
                mu_hat += 0.5 * sigma_sq_hat

            n = len(dt_values)
            log_likelihood = (
                -0.5 * n * np.log(2 * np.pi)
                - 0.5 * n * np.log(sigma_sq_hat)
                - 0.5 * np.sum(np.log(dt_values))
                - 0.5 * np.sum(residuals**2 / (sigma_sq_hat * dt_values))
            )

            calibrated_params = {
                "mu": mu_hat,
                "sigma": sigma_hat,
                "log_likelihood": float(log_likelihood),
            }

            self.mu = mu_hat
            self.sigma = sigma_hat
            self.model = model

            sigma_note = "(% vol)" if model == "gbm" else "(abs vol)"
            print(
                f"Calibrated {model.upper()} parameters (P-measure, historical):"
            )
            print(f"  mu (drift): {mu_hat:.6f}")
            print(f"  sigma {sigma_note}: {sigma_hat:.6f}")
            print(f"  Log-likelihood: {log_likelihood:.2f}")

            return calibrated_params

        if model == "vasicek":
            dr = np.diff(rates)
            r_levels = rates[:-1]

            dt_avg = np.mean(dt_values)

            # OLS discretisation of dr = κ(θ - r)·dt + σ·dW:
            #   Δr_t = α + β·r_t + ε_t,  ε ~ N(0, σ²_resid)
            #   κ̂ = -β/dt,  θ̂ = -α/β,  σ̂ = std(ε) / √dt
            X = np.column_stack([np.ones(len(dr)), r_levels])
            coeffs, _, _, _ = np.linalg.lstsq(X, dr, rcond=None)
            alpha_hat, beta_hat = coeffs

            residuals = dr - (alpha_hat + beta_hat * r_levels)

            kappa_hat = -beta_hat / dt_avg
            theta_hat = -alpha_hat / beta_hat
            sigma_hat = np.std(residuals, ddof=0) / np.sqrt(dt_avg)

            n = len(dr)
            sigma2_mle = np.mean(residuals**2)
            log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma2_mle) + 1)

            calibrated_params = {
                "kappa": kappa_hat,
                "theta": theta_hat,
                "sigma": sigma_hat,
                "log_likelihood": float(log_likelihood),
            }

            self.kappa = kappa_hat
            self.mu = theta_hat
            self.sigma = sigma_hat
            self.model = model

            r_squared = 1 - np.var(residuals) / np.var(dr)
            half_life_days = (
                np.log(2) / kappa_hat if kappa_hat > 0 else np.inf
            )

            print(
                f"Calibrated {model.upper()} parameters "
                f"(P-measure, historical, OLS):"
            )
            print(f"  kappa (mean reversion speed): {kappa_hat:.6f}")
            print(
                f"  theta (long-term mean, P-measure ref): "
                f"{theta_hat:.4%}"
            )
            print(f"  sigma (annualized vol): {sigma_hat:.4%}")
            print(f"  R-squared: {r_squared:.4f}")
            print(
                f"  Half-life: {half_life_days:.1f} days "
                f"({half_life_days / 365.25:.2f} years)"
            )
            print(f"  Log-likelihood: {log_likelihood:.2f}")

            if kappa_hat <= 0:
                print(
                    "  WARNING: kappa <= 0 — no mean reversion detected. "
                    "Consider model='abm'."
                )

            return calibrated_params

        if model == "cir":
            dr = np.diff(rates)
            r_prev = rates[:-1]

            r_prev_safe = np.maximum(r_prev, 1e-8)
            sqrt_r = np.sqrt(r_prev_safe)
            sqrt_dt = np.sqrt(dt_values)

            # Homoskedastic-transformed regression:
            #   Δr/(√r·√dt) = κθ·(√dt/√r) + (-κ)·(√r·√dt) + σ·ε
            #   κ̂ = -b̂,  θ̂ = â/κ̂,  σ̂ = std(residuals)
            z = dr / (sqrt_r * sqrt_dt)
            w1 = sqrt_dt / sqrt_r
            w2 = sqrt_r * sqrt_dt
            X = np.column_stack([w1, w2])
            coeffs, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
            a_hat, b_hat = coeffs

            residuals = z - (a_hat * w1 + b_hat * w2)

            kappa_hat = -b_hat
            theta_hat = (
                a_hat / kappa_hat if kappa_hat > 1e-12 else float("nan")
            )
            sigma_hat = np.std(residuals, ddof=0)

            n = len(dr)
            sigma2_mle = np.mean(residuals**2)
            log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma2_mle) + 1)

            calibrated_params = {
                "kappa": kappa_hat,
                "theta": theta_hat,
                "sigma": sigma_hat,
                "log_likelihood": float(log_likelihood),
            }

            self.kappa = kappa_hat
            self.mu = theta_hat
            self.sigma = sigma_hat
            self.model = model

            half_life_days = (
                np.log(2) / kappa_hat if kappa_hat > 0 else np.inf
            )

            print(
                f"Calibrated {model.upper()} parameters "
                f"(P-measure, historical, OLS):"
            )
            print(f"  kappa (mean reversion speed): {kappa_hat:.6f}")
            if not np.isnan(theta_hat):
                print(
                    f"  theta (long-term mean, P-measure ref): "
                    f"{theta_hat:.4%}"
                )
            print(f"  sigma (volatility): {sigma_hat:.6f}")
            print(
                f"  Half-life: {half_life_days:.1f} days "
                f"({half_life_days / 365.25:.2f} years)"
            )

            if not np.isnan(theta_hat):
                feller = 2 * kappa_hat * theta_hat
                if feller >= sigma_hat**2:
                    print(
                        f"  Feller: SATISFIED (2κθ = {feller:.6f} "
                        f">= σ² = {sigma_hat**2:.6f})"
                    )
                else:
                    print(
                        f"  Feller: VIOLATED (2κθ = {feller:.6f} "
                        f"< σ² = {sigma_hat**2:.6f})"
                    )

            print(f"  Log-likelihood: {log_likelihood:.2f}")

            if kappa_hat <= 0:
                print(
                    "  WARNING: kappa <= 0 — no mean reversion detected. "
                    "Consider model='abm'."
                )

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
        """Load historical yield data from CSV and calibrate a rate model.

        Args:
            csv_path: Path to CSV file containing historical yield data
            start_date: Start date of the data. If None, use the first date in the data.
            end_date: End date of the data. If None, use the last date in the data.
            model: Model to calibrate - "cir" (default), "vasicek", "abm" or "gbm".
                CIR/GBM require strictly positive rates; Vasicek/ABM tolerate
                negative rates.  GBM sigma is proportional (dimensionless),
                all others use absolute volatility.
            date_col: Name of the date column in CSV
            rate_col: Name of the rate column in CSV

        Returns:
            Dictionary with calibrated parameters.
            CIR/Vasicek: {'kappa', 'theta', 'sigma', 'log_likelihood'}
            ABM/GBM:     {'mu', 'sigma', 'log_likelihood'}

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

            # ABM
            params = simulator.load_and_calibrate_from_csv(
                "sofr_2020_2025.csv", model="abm"
            )
        """
        if model not in ("cir", "vasicek", "abm", "gbm"):
            raise ValueError(f"Unsupported model '{model}'. Use 'cir', 'vasicek', 'abm' or 'gbm'.")

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

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

    def calibrate_cir_model(
        self, 
        historical_data: pd.DataFrame, 
        date_col: str = 'date', 
        rate_col: str = 'rate'
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
        # Ensure data is sorted by date
        data = historical_data.copy().sort_values(date_col).reset_index(drop=True)
        
        # Convert rates to numpy array
        rates = data[rate_col].values
        
        # Calculate time differences (assuming daily data, convert to years)
        if pd.api.types.is_datetime64_any_dtype(data[date_col]):
            dates = pd.to_datetime(data[date_col])
            dt_values = np.diff(dates).astype('timedelta64[D]').astype(float) / 365.25
        else:
            # Assume uniform daily spacing if dates are not datetime
            dt_values = np.full(len(rates) - 1, 1/365.25)
        
        # Remove any zero or negative rates (CIR requires positive rates)
        valid_mask = rates > 0
        rates = rates[valid_mask]
        
        if len(rates) < 10:
            raise ValueError("Insufficient valid data points for calibration")
        
        # Adjust dt_values to match filtered rates
        if len(dt_values) >= len(rates):
            dt_values = dt_values[:len(rates)-1]
        else:
            dt_values = dt_values[:len(rates)-1]
        
        print(f"Calibrating CIR model with {len(rates)} data points...")
        
        def cir_log_likelihood(params: np.ndarray) -> float:
            """Calculate negative log-likelihood for CIR model."""
            kappa, theta, sigma = params
            
            # Parameter constraints
            if kappa <= 0 or theta <= 0 or sigma <= 0:
                return np.inf
            
            # Feller condition: 2*kappa*theta >= sigma^2
            if 2 * kappa * theta < sigma**2:
                return np.inf
            
            log_likelihood = 0.0
            
            for i in range(len(rates) - 1):
                r_t = rates[i]
                r_t1 = rates[i + 1]
                dt = dt_values[i] if i < len(dt_values) else dt_values[-1]
                
                # CIR transition density parameters
                c = 2 * kappa / ((1 - np.exp(-kappa * dt)) * sigma**2)
                q = 2 * kappa * theta / sigma**2 - 1
                u = c * r_t * np.exp(-kappa * dt)
                v = c * r_t1
                
                # Avoid numerical issues
                if u <= 0 or v <= 0:
                    continue
                
                # Log-likelihood contribution (approximation for simplicity)
                # Using Euler approximation for the transition density
                mean = r_t * np.exp(-kappa * dt) + theta * (1 - np.exp(-kappa * dt))
                variance = (sigma**2 * r_t / kappa) * (np.exp(-kappa * dt) - np.exp(-2 * kappa * dt)) + \
                          (theta * sigma**2 / (2 * kappa)) * (1 - np.exp(-kappa * dt))**2
                
                if variance <= 0:
                    continue
                
                # Gaussian approximation to CIR transition density
                log_likelihood += -0.5 * np.log(2 * np.pi * variance) - \
                                0.5 * (r_t1 - mean)**2 / variance
            
            return -log_likelihood  # Return negative for minimization
        
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
            (0.001, 2.0),    # kappa: mean reversion speed
            (0.001, 0.2),    # theta: long-term mean  
            (0.001, 0.5),    # sigma: volatility
        ]
        
        print(f"Initial parameter estimates: kappa={initial_kappa:.4f}, theta={initial_theta:.4f}, sigma={initial_sigma:.4f}")
        
        # Optimize using multiple starting points for robustness
        best_result = None
        best_likelihood = np.inf
        
        for attempt in range(3):
            try:
                # Add some randomness to initial guess
                if attempt > 0:
                    initial_params *= (1 + 0.1 * np.random.randn(3))
                    initial_params = np.clip(initial_params, 
                                           [b[0] for b in bounds], 
                                           [b[1] for b in bounds])
                
                result = optimize.minimize(
                    cir_log_likelihood,
                    initial_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000}
                )
                
                if result.success and result.fun < best_likelihood:
                    best_result = result
                    best_likelihood = result.fun
                    
            except Exception as e:
                print(f"Optimization attempt {attempt + 1} failed: {e}")
                continue
        
        if best_result is None or not best_result.success:
            # Fallback to method of moments if optimization fails
            print("Optimization failed, using method of moments estimates...")
            
            # Simple method of moments for CIR
            dt_mean = np.mean(dt_values)
            rate_changes = np.diff(rates)
            
            # Estimate parameters using sample moments
            mean_rate = np.mean(rates[:-1])
            mean_change = np.mean(rate_changes) / dt_mean
            var_change = np.var(rate_changes) / dt_mean
            
            # Method of moments estimates (simplified)
            theta_est = mean_rate - mean_change / 0.1  # Assume kappa = 0.1 initially
            kappa_est = -mean_change / (mean_rate - theta_est) if mean_rate != theta_est else 0.1
            sigma_est = np.sqrt(var_change / mean_rate) if mean_rate > 0 else 0.02
            
            # Ensure reasonable bounds
            kappa_est = max(0.01, min(2.0, abs(kappa_est)))
            theta_est = max(0.001, min(0.2, abs(theta_est)))
            sigma_est = max(0.001, min(0.5, abs(sigma_est)))
            
            calibrated_params = {
                'kappa': kappa_est,
                'theta': theta_est,
                'sigma': sigma_est,
                'log_likelihood': -cir_log_likelihood([kappa_est, theta_est, sigma_est])
            }
        else:
            kappa_opt, theta_opt, sigma_opt = best_result.x
            calibrated_params = {
                'kappa': kappa_opt,
                'theta': theta_opt,
                'sigma': sigma_opt,
                'log_likelihood': -best_result.fun
            }
        
        # Update instance parameters
        self.kappa = calibrated_params['kappa']
        self.mu = calibrated_params['theta']  # Note: mu is used as theta in this class
        self.sigma = calibrated_params['sigma']
        
        print(f"Calibrated CIR parameters:")
        print(f"  kappa (mean reversion speed): {calibrated_params['kappa']:.6f}")
        print(f"  theta (long-term mean): {calibrated_params['theta']:.6f}")
        print(f"  sigma (volatility): {calibrated_params['sigma']:.6f}")
        print(f"  Log-likelihood: {calibrated_params['log_likelihood']:.2f}")
        
        # Verify Feller condition
        feller_condition = 2 * calibrated_params['kappa'] * calibrated_params['theta']
        if feller_condition >= calibrated_params['sigma']**2:
            print(f"  Feller condition satisfied: 2κθ = {feller_condition:.6f} >= σ² = {calibrated_params['sigma']**2:.6f}")
        else:
            print(f"  WARNING: Feller condition violated: 2κθ = {feller_condition:.6f} < σ² = {calibrated_params['sigma']**2:.6f}")
            print("  This may lead to negative interest rates in simulation.")
        
        return calibrated_params

    def load_and_calibrate_from_csv(
        self, 
        csv_path: str, 
        date_col: str = 'date', 
        rate_col: str = 'rate',
        date_format: str = None
    ) -> dict[str, float]:
        """Load historical yield data from CSV and calibrate CIR model.
        
        Args:
            csv_path: Path to CSV file containing historical yield data
            date_col: Name of the date column in CSV
            rate_col: Name of the rate column in CSV
            date_format: Date format string (e.g., '%Y-%m-%d'). If None, pandas will infer.
            
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
            print(f"Loading historical data from: {csv_path}")
            historical_data = pd.read_csv(csv_path)
            
            # Convert date column to datetime
            if date_format:
                historical_data[date_col] = pd.to_datetime(historical_data[date_col], format=date_format)
            else:
                historical_data[date_col] = pd.to_datetime(historical_data[date_col])
            
            # Basic data validation
            print(f"Loaded {len(historical_data)} records")
            print(f"Date range: {historical_data[date_col].min()} to {historical_data[date_col].max()}")
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
            print(f"Available columns: {list(historical_data.columns) if 'historical_data' in locals() else 'Unable to read CSV'}")
            raise
        except Exception as e:
            print(f"Error loading/calibrating data: {e}")
            raise

    def test_calibration(self) -> None:
        """Test the CIR calibration with sample data."""
        print("=== Testing CIR Model Calibration ===")
        
        # Create sample historical data (you can replace this with your actual data)
        # This simulates 5-year Treasury yields
        np.random.seed(42)  # For reproducible results
        
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        
        # Simulate some realistic 5-year Treasury yield data
        # Starting around 1.5% and trending upward with volatility
        n_days = len(dates)
        base_rate = 0.015  # 1.5%
        trend = np.linspace(0, 0.035, n_days)  # Upward trend to 3.5%
        noise = np.random.normal(0, 0.002, n_days)  # Daily volatility
        
        # Apply some mean reversion to make it more realistic
        rates = np.zeros(n_days)
        rates[0] = base_rate
        
        for i in range(1, n_days):
            # Simple mean-reverting process for sample data
            mean_level = base_rate + trend[i]
            rates[i] = rates[i-1] + 0.1 * (mean_level - rates[i-1]) * (1/365) + noise[i]
            rates[i] = max(rates[i], 0.001)  # Ensure positive
        
        # Create DataFrame
        sample_data = pd.DataFrame({
            'date': dates,
            'rate': rates
        })
        
        print(f"Sample data summary:")
        print(f"  Date range: {sample_data['date'].min()} to {sample_data['date'].max()}")
        print(f"  Rate range: {sample_data['rate'].min():.4f} to {sample_data['rate'].max():.4f}")
        print(f"  Mean rate: {sample_data['rate'].mean():.4f}")
        print(f"  Rate std: {sample_data['rate'].std():.4f}")
        print()
        
        # Calibrate the CIR model
        calibrated_params = self.calibrate_cir_model(sample_data)
        
        return calibrated_params

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
    """Demonstrate CIR model calibration."""
    print("=== CIR Model Calibration Demo ===")
    print("This demo shows two ways to calibrate the CIR model:")
    print("1. Using simulated test data")
    print("2. Loading data from CSV file (example)")
    print()
    
    # Initialize simulator
    simulator = InterestRateSimulator(model="cir")
    
    # Method 1: Test calibration with sample data
    print("METHOD 1: Test calibration with simulated data")
    print("-" * 50)
    calibrated_params = simulator.test_calibration()
    
    # Method 2: Show how to use CSV data (example)
    print("\nMETHOD 2: How to use your own CSV data")
    print("-" * 50)
    print("To use your own historical 5-year Treasury yield data:")
    print("1. Create a CSV file with columns 'date' and 'rate'")
    print("2. Rates should be in decimal form (e.g., 0.05 for 5%)")
    print("3. Use the following code:")
    print()
    print("# Example code:")
    print("simulator = InterestRateSimulator(model='cir')")
    print("params = simulator.load_and_calibrate_from_csv('your_data.csv')")
    print()
    print("# CSV format example:")
    print("# date,rate")
    print("# 2020-01-01,0.0150")
    print("# 2020-01-02,0.0152")
    print("# 2020-01-03,0.0148")
    print("# ...")
    print()
    
    # Uncomment the following lines if you have a CSV file to test:
    # try:
    #     csv_params = simulator.load_and_calibrate_from_csv("5yr_treasury_yields.csv")
    #     print("Successfully calibrated from CSV data!")
    # except:
    #     print("CSV file not found - using simulated data results")
    
    # Generate and plot some paths with calibrated parameters
    print("=== Generating Rate Paths with Calibrated Parameters ===")
    simulator.days = 365 * 2  # 2 years
    simulator.num_paths = 10
    
    rates_df = simulator.generate_rates()
    print(f"Generated {len(rates_df.columns)} rate paths for {len(rates_df)} days")
    
    # Show summary statistics of generated paths
    print(f"Generated rate statistics:")
    print(f"  Mean: {rates_df.mean().mean():.4f}")
    print(f"  Std:  {rates_df.std().mean():.4f}")
    print(f"  Min:  {rates_df.min().min():.4f}")
    print(f"  Max:  {rates_df.max().max():.4f}")
    
    # Plot first few paths (uncomment to see plots)
    # simulator.plot_rate_paths(max_paths=5)


if __name__ == "__main__":
    main() 
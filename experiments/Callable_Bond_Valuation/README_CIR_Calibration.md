# CIR Model Calibration for Interest Rate Simulation

This module provides functionality to calibrate the Cox-Ingersoll-Ross (CIR) interest rate model using historical 5-year Treasury yield data.

## Overview

The CIR model follows the stochastic differential equation:
```
dr = κ(θ - r)dt + σ√r dW
```

Where:
- `κ` (kappa): Mean reversion speed
- `θ` (theta): Long-term mean level  
- `σ` (sigma): Volatility parameter
- `r`: Interest rate
- `dW`: Wiener process

## Usage

### Method 1: Using Your Own CSV Data

1. **Prepare your data**: Create a CSV file with historical 5-year Treasury yields
   ```csv
   date,rate
   2023-01-01,0.0389
   2023-01-02,0.0392
   2023-01-03,0.0395
   ...
   ```
   
   **Important**: 
   - Rates should be in decimal form (e.g., 0.05 for 5%)
   - Dates should be in YYYY-MM-DD format (or specify custom format)

2. **Calibrate the model**:
   ```python
   from interest_rate_simulator import InterestRateSimulator
   
   # Initialize simulator
   simulator = InterestRateSimulator(model="cir")
   
   # Calibrate from CSV
   params = simulator.load_and_calibrate_from_csv("your_5yr_yields.csv")
   
   # The simulator parameters are automatically updated
   print(f"Calibrated parameters:")
   print(f"  κ (kappa): {params['kappa']:.6f}")
   print(f"  θ (theta): {params['theta']:.6f}")  
   print(f"  σ (sigma): {params['sigma']:.6f}")
   ```

3. **Generate rate paths**:
   ```python
   # Generate interest rate paths using calibrated parameters
   simulator.days = 365 * 5  # 5 years
   simulator.num_paths = 1000
   rates_df = simulator.generate_rates()
   
   # Plot sample paths
   simulator.plot_rate_paths(max_paths=10)
   ```

### Method 2: Using DataFrame Data

If you already have data in a pandas DataFrame:

```python
import pandas as pd
from interest_rate_simulator import InterestRateSimulator

# Your DataFrame with columns 'date' and 'rate'
historical_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', '2024-12-31', freq='D'),
    'rate': your_rate_data  # Your historical rates
})

# Calibrate
simulator = InterestRateSimulator(model="cir")
params = simulator.calibrate_cir_model(historical_data, 'date', 'rate')
```

### Method 3: Testing with Simulated Data

For testing purposes, you can use the built-in test method:

```python
from interest_rate_simulator import InterestRateSimulator

simulator = InterestRateSimulator(model="cir")
params = simulator.test_calibration()
```

## Key Features

### Maximum Likelihood Estimation
The calibration uses maximum likelihood estimation with multiple optimization attempts for robustness.

### Feller Condition Check
The calibration automatically checks the Feller condition: `2κθ ≥ σ²`
- If satisfied: The CIR process won't reach zero
- If violated: Warning is displayed (may lead to negative rates)

### Robust Parameter Bounds
- κ (kappa): 0.001 to 2.0
- θ (theta): 0.001 to 0.2 (20%)
- σ (sigma): 0.001 to 0.5

### Fallback to Method of Moments
If maximum likelihood optimization fails, the method automatically falls back to method of moments estimation.

## Output

The calibration returns a dictionary with:
```python
{
    'kappa': 0.123456,      # Mean reversion speed
    'theta': 0.045678,      # Long-term mean
    'sigma': 0.012345,      # Volatility
    'log_likelihood': 1234.56  # Log-likelihood value
}
```

## Example Output

```
Loading historical data from: sample_5yr_yields.csv
Loaded 50 records
Date range: 2023-01-01 00:00:00 to 2023-03-09 00:00:00
Rate statistics:
  Mean: 0.0462
  Std:  0.0044
  Min:  0.0389
  Max:  0.0536

Calibrating CIR model with 50 data points...
Initial parameter estimates: kappa=0.1000, theta=0.0462, sigma=0.0087
Calibrated CIR parameters:
  kappa (mean reversion speed): 0.520595
  theta (long-term mean): 0.200000
  sigma (volatility): 0.010236
  Log-likelihood: 370.06
  Feller condition satisfied: 2κθ = 0.208238 >= σ² = 0.000105
```

## Integration with Callable Bond Valuation

Once calibrated, the CIR model parameters are automatically set in the `InterestRateSimulator` instance and can be used directly with the callable bond valuation framework:

```python
from callable_bond_valuer import CallableBond, CallableBondValuer
from financepy.utils.date import Date

# Calibrate CIR model
simulator = InterestRateSimulator(model="cir")
params = simulator.load_and_calibrate_from_csv("5yr_yields.csv")

# Create callable bond
callable_bond = CallableBond(
    issue_date=Date(1, 1, 2025),
    maturity_date=Date(1, 1, 2030),
    coupon_rate=0.05,
    call_protection_years=1
)

# Value the bond using calibrated interest rate model
valuer = CallableBondValuer(callable_bond, simulator)
bond_value, straight_value, stats = valuer.value_bond(
    valuation_date=Date(1, 1, 2025),
    straight_bond_ytm=0.046
)
```

## Files

- `interest_rate_simulator.py`: Main calibration and simulation functionality
- `sample_5yr_yields.csv`: Example CSV file format
- `callable_bond_valuer.py`: Callable bond valuation using calibrated rates

## Requirements

- pandas
- numpy
- scipy
- matplotlib
- financepy (for callable bond valuation) 
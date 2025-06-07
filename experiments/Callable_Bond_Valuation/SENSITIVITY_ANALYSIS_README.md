# Callable Bond Sensitivity Analysis

This module provides comprehensive sensitivity analysis for callable bond valuation using Monte Carlo simulation.

## Overview

The `CallableBondSensitivityAnalyzer` class analyzes how callable bond values change with respect to four key parameters:

1. **Coupon Rate**: Annual coupon payment rate
2. **Time to Maturity**: Years until bond maturity  
3. **Volatility (Sigma)**: Interest rate volatility parameter
4. **Straight Bond YTM**: Yield to maturity for equivalent non-callable bond

## Key Features

- **Monte Carlo Valuation**: Uses the same valuation approach as `solve_callable_bond_coupon()`
- **Equivalent Rate Calculation**: Automatically computes equivalent initial short rates
- **Comprehensive Output**: Generates plots, CSV files, and detailed reports
- **Call Probability Analysis**: Tracks probability of early call under different scenarios
- **Option Value Quantification**: Measures the value of the embedded call option

## Usage

### Basic Usage

```python
from financepy.utils.date import Date
from callable_bond_sensitivity_analysis import CallableBondSensitivityAnalyzer

# Define base parameters
issue_date = Date(27, 5, 2025)
maturity_date = Date(27, 5, 2030)  # 5 years
base_coupon_rate = 0.046  # 4.6%
base_straight_bond_ytm = 0.046  # 4.6%
base_sigma = 0.0095  # 0.95%

# Create analyzer
analyzer = CallableBondSensitivityAnalyzer(
    base_issue_date=issue_date,
    base_maturity_date=maturity_date,
    base_coupon_rate=base_coupon_rate,
    base_straight_bond_ytm=base_straight_bond_ytm,
    base_sigma=base_sigma,
    num_simulations=1000,
)

# Run full analysis
results = analyzer.run_full_sensitivity_analysis()

# Generate report and plots
report = analyzer.generate_sensitivity_report()
print(report)
analyzer.create_sensitivity_plots()
analyzer.create_call_probability_plots()
```

### Custom Parameter Ranges

```python
# Define custom ranges for targeted analysis
coupon_range = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]
maturity_range = [3, 5, 7, 10]  # Years
sigma_range = [0.005, 0.01, 0.015, 0.02]  # Volatility
ytm_range = [0.025, 0.035, 0.045, 0.055, 0.065]  # YTM

results = analyzer.run_full_sensitivity_analysis(
    coupon_range=coupon_range,
    maturity_years_range=maturity_range,
    sigma_range=sigma_range,
    ytm_range=ytm_range,
)
```

## Output Files

The analysis generates several output files:

### Plots
- `callable_bond_sensitivity_analysis.png`: Main sensitivity plots with bond values and option values
- `callable_bond_call_probability_sensitivity.png`: Call probability sensitivity plots

### Data Files
- `coupon_sensitivity_results.csv`: Coupon rate sensitivity data
- `maturity_sensitivity_results.csv`: Time to maturity sensitivity data
- `sigma_sensitivity_results.csv`: Volatility sensitivity data
- `ytm_sensitivity_results.csv`: YTM sensitivity data

### Report Output
Console output includes:
- Base parameter summary
- Sensitivity ranges for each parameter
- Detailed numerical results tables

## Key Metrics Analyzed

For each parameter variation, the analysis tracks:

- **Callable Bond Value**: Monte Carlo valuation of the callable bond
- **Straight Bond Value**: Value of equivalent non-callable bond
- **Option Value**: Difference between straight and callable bond values
- **Call Probability**: Percentage of simulation paths where bond is called early
- **Standard Deviation**: Variability of callable bond values across simulations

## Methodology

### Valuation Approach

1. **Equivalent Rate Calculation**: For each parameter combination, calculates the equivalent initial short rate that matches the straight bond YTM using the Monte Carlo framework
2. **Rate Path Generation**: Generates interest rate paths using Arithmetic Brownian Motion (ABM) model
3. **Path-Dependent Valuation**: Values the callable bond along each rate path, checking call conditions at each call date
4. **Statistical Aggregation**: Computes mean values and call statistics across all simulation paths

### Call Decision Logic

The issuer calls the bond when:
- Current date is a valid call date (after call protection period)
- Continuation value > Call price × (1 + buffer)
- Buffer of 2% is used to account for transaction costs

### Discounting Method

Uses discrete discounting: `PV = CF / ∏(1 + r_i × dt)` where:
- `CF` = Cash flow amount
- `r_i` = Interest rate in period i
- `dt` = Time step (daily: 1/365)

## Example Results Interpretation

### Coupon Rate Sensitivity
- **Higher coupons** → Higher callable bond values but also higher option values
- **Call probability increases** with coupon rate (bonds with high coupons more likely to be called)

### Time to Maturity Sensitivity  
- **Longer maturity** → Higher option values (more time for rates to fall)
- **Call probability** may increase or decrease depending on rate environment

### Volatility Sensitivity
- **Higher volatility** → Higher option values (more upside for issuer)
- **Call probability** generally increases with volatility

### YTM Sensitivity
- **Lower YTM** → Higher bond values and option values
- **Call probability decreases** as YTM increases (less attractive to call when rates are high)

## Configuration Parameters

### Base Parameters
- `base_issue_date`: Bond issue date
- `base_maturity_date`: Bond maturity date
- `base_coupon_rate`: Annual coupon rate (decimal)
- `base_straight_bond_ytm`: Straight bond YTM (decimal)
- `base_sigma`: Interest rate volatility (decimal)

### Analysis Parameters
- `call_protection_years`: Years before bond becomes callable (default: 1)
- `num_simulations`: Number of Monte Carlo paths (default: 2000)
- `call_premium`: Premium above par for calling (default: 0.0)

### Financial Parameters
- `face_value`: Bond face value (default: 100.0)
- `freq_type`: Coupon frequency (default: Annual)
- `day_count_type`: Day count convention (default: 30E/360)

## Performance Considerations

- **Simulation Count**: More simulations improve accuracy but increase runtime
- **Parameter Ranges**: Larger ranges provide more comprehensive analysis but take longer
- **Parallel Processing**: Consider reducing simulation count for faster iteration during development

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation and analysis
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical plotting
- `financepy`: Bond pricing and date handling
- `scipy`: Optimization (used in underlying callable bond valuer)

## Running the Analysis

To run the complete sensitivity analysis:

```bash
cd experiments/Callable_Bond_Valuation
python callable_bond_sensitivity_analysis.py
```

This will execute the full analysis with default parameters and generate all output files in the current directory. 
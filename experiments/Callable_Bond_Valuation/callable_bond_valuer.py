import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

# Import FinancePy classes
from financepy.products.bonds.bond import Bond, YTMCalcType
from financepy.utils.date import Date
from financepy.utils.frequency import FrequencyTypes
from financepy.utils.day_count import DayCountTypes


class InterestRateSimulator:
    """A class for simulating interest rate movements using various models.

    For MTC simulations, the interest rate is short rate. The overnight rate is recommended.

    """
    
    def __init__(
        self,
        r0: float = 0.043,  # Initial interest rate
        mu: float = 0.0,   # Mean reversion level (for mean-reverting models)
        sigma: float = 0.02,  # Volatility
        days: int = 365 * 6,  # 5+1 years of simulation. 1 year buffer for discounting 5 year bond.
        days_of_year: int = 365,
        num_paths: int = 10000,
        model: str = "vasicek",  # "vasicek", "cir", "gbm"
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
            model: Model type - "vasicek", "cir", or "gbm"
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
                rates[:, t] = (rates[:, t-1] + 
                              self.kappa * (self.mu - rates[:, t-1]) * self._dt +
                              self.sigma * dW)
                              
        elif self.model == "cir":
            # Cox-Ingersoll-Ross model: dr = kappa*(mu - r)*dt + sigma*sqrt(r)*dW
            for t in range(1, self.days):
                dW = np.random.normal(0, 1, size=self.num_paths) * self._sqrt_dt
                sqrt_r = np.maximum(np.sqrt(np.abs(rates[:, t-1])), 1e-8)
                rates[:, t] = (rates[:, t-1] + 
                              self.kappa * (self.mu - rates[:, t-1]) * self._dt +
                              self.sigma * sqrt_r * dW)
                # Ensure non-negative rates
                rates[:, t] = np.maximum(rates[:, t], 0.0001)
                
        elif self.model == "gbm":
            # Geometric Brownian Motion: dr = mu*r*dt + sigma*r*dW
            for t in range(1, self.days):
                dW = np.random.normal(0, 1, size=self.num_paths) * self._sqrt_dt
                rates[:, t] = rates[:, t-1] * np.exp(
                    (self.mu - 0.5 * self.sigma**2) * self._dt +
                    self.sigma * dW
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
                    columns=[f"path_{i+1}" for i in range(self.num_paths)]
                )
            else:
                # For very long simulations, use numerical index
                self.rates_df = pd.DataFrame(
                    rates.T,
                    index=np.arange(self.days),
                    columns=[f"path_{i+1}" for i in range(self.num_paths)]
                )
        except (OverflowError, pd._libs.tslibs.np_datetime.OutOfBoundsTimedelta):
            self.rates_df = pd.DataFrame(
                rates.T,
                index=np.arange(self.days),
                columns=[f"path_{i+1}" for i in range(self.num_paths)]
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
            alpha=0.5, legend=False
        )
        plt.xlabel("Date")
        plt.ylabel("Interest Rate")
        plt.grid(True)
        plt.show()


class CallableBond:
    """A callable bond implementation using FinancePy's Bond class as the base."""
    
    def __init__(
        self,
        issue_date: Date,
        maturity_date: Date,
        coupon_rate: float,
        call_protection_years: int = 1,
        face_value: float = 100.0,
        freq_type: FrequencyTypes = FrequencyTypes.ANNUAL,
        day_count_type: DayCountTypes = DayCountTypes.THIRTY_E_360,
    ):
        """Initialize a callable bond.
        
        Args:
            issue_date: Bond issue date
            maturity_date: Bond maturity date  
            coupon_rate: Annual coupon rate
            call_protection_years: Years before bond becomes callable
            face_value: Face value of the bond
            freq_type: Coupon payment frequency
            day_count_type: Day count convention
        """
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.coupon_rate = coupon_rate
        self.call_protection_years = call_protection_years
        self.face_value = face_value
        self.freq_type = freq_type
        self.day_count_type = day_count_type
        
        # Create the underlying FinancePy Bond
        self.bond = Bond(
            issue_dt=issue_date,
            maturity_dt=maturity_date,
            coupon=coupon_rate,
            freq_type=freq_type,
            dc_type=day_count_type
        )
        
        # Calculate call dates (annual callable dates after protection period)
        self.call_dates = self._calculate_call_dates()
        
    def _calculate_call_dates(self) -> list[Date]:
        """Calculate the dates when the bond can be called."""
        call_dates = []
        
        # Start from first call date (issue date + protection years)
        first_call_date = self.issue_date.add_years(self.call_protection_years)
        
        # Add annual call dates until maturity
        call_date = first_call_date
        while call_date < self.maturity_date:
            call_dates.append(call_date)
            call_date = call_date.add_years(1)
            
        return call_dates
    
    def dirty_price_from_ytm(
        self, 
        settle_date: Date, 
        ytm: float,
        convention: YTMCalcType = YTMCalcType.US_STREET
    ) -> float:
        """Calculate dirty price using FinancePy's implementation."""
        return self.bond.dirty_price_from_ytm(settle_date, ytm, convention)
    
    def yield_to_maturity(
        self,
        settle_date: Date,
        clean_price: float,
        convention: YTMCalcType = YTMCalcType.US_STREET
    ) -> float:
        """Calculate yield to maturity using FinancePy's implementation."""
        return self.bond.yield_to_maturity(settle_date, clean_price, convention)


class CallableBondValuer:
    """Monte Carlo valuation engine for callable bonds."""
    
    def __init__(
        self,
        callable_bond: CallableBond,
        rate_simulator: InterestRateSimulator,
        call_premium: float = 0.0,  # Premium above par for calling
    ):
        """Initialize the callable bond valuer.
        
        Args:
            callable_bond: The callable bond to value
            rate_simulator: Interest rate simulator
            call_premium: Premium above par when calling (usually 0)
        """
        self.callable_bond = callable_bond
        self.rate_simulator = rate_simulator
        self.call_premium = call_premium
        
    def _pv_of_future_cash_flows(
        self,
        cash_flow_dates: list[Date],
        cash_flow_amounts: list[float],
        valuation_date: Date,
        rate_path: np.ndarray,
        discount_mode: str = "discrete"
    ) -> float:
        """
        Calculate the present value of a series of cash flows using a simulated rate path.
        In a continuous time simulation with time step dt, this is approximated by
            $exp(- sum(rate_path[i] * dt))$
        In a discrete time simulation with time step dt, this is approximated by
            $ (prod(1 + rate_path[i]) ^ dt)^{-1}$
        where the product is over the periods from T_val to T_cf.
        Args:
            cash_flow_dates: List of cash flow dates
            cash_flow_amounts: List of cash flow amounts
            valuation_date: Date to value the bond
            rate_path: Numpy array of overnight rates for each day in the simulation
            discount_mode: "discrete" or "continuous"
        Returns:
            Present value of the cash flows
        """
        pv = 0.0
        path_dt = self.rate_simulator._dt
        max_path_idx = len(rate_path)

        for cf_date, cf_amount in zip(cash_flow_dates, cash_flow_amounts):
            if cf_date < valuation_date:
                continue
            elif cf_date == valuation_date:
                pv += cf_amount
                continue
            # Calculate number of days from valuation_date to cf_date
            # rate_path[0] is for the first period from valuation_date
            days_to_cf = int(round((cf_date - valuation_date))) # Date subtraction gives float days

            # Determine the slice of the rate path relevant for this cash flow
            # We need rates from index 0 up to days_to_cf - 1
            end_idx = min(days_to_cf, max_path_idx)
            
            relevant_rates = rate_path[0:end_idx]
            
            # This assumes simple summation; for more precision, one might align dt with actual day count to cf.
            # However, with daily dt, sum(rates*dt) is standard.
            # If end_idx < days_to_cf, it means rate_path is shorter than time to cash flow.
            # The discounting will only cover up to len(rate_path).
            
            # Discrete Mode:
            # $\frac{1}{\prod_{i=1}^{n}(1+r_i \cdot \Delta t) }$
            #
            # Continuous Mode:
            # $\exp\left(-\sum_{i=1}^{n} r_i \cdot \Delta t\right)$

            if discount_mode == "discrete":
                discount_factor = 1.0 / np.prod(1 + relevant_rates * path_dt)
            elif discount_mode == "continuous":
                discount_factor = np.exp(-np.sum(relevant_rates * path_dt))
            else:
                raise ValueError(f"Invalid discount mode: {discount_mode}")
            pv += cf_amount * discount_factor
            
        return pv

    def value_bond(
        self, 
        valuation_date: Date,
        current_rate: float,
        num_simulations: int | None = None
    ) -> tuple[float, float, dict]:
        """Value the callable bond using Monte Carlo simulation.
        
        Args:
            valuation_date: Date to value the bond
            current_rate: Current interest rate
            num_simulations: Number of MC simulations (if None, uses rate_simulator setting)
        
        Returns:
            Tuple of (callable_bond_value, straight_bond_value, additional_stats)
        """
        if num_simulations is not None:
            self.rate_simulator.num_paths = num_simulations
            
        # Generate interest rate paths
        self.rate_simulator.generate_rates()  # Always regenerate for fresh random paths
        rates_df = self.rate_simulator.rates_df
        
        # Calculate straight bond value (no call option)
        straight_bond_value = self.callable_bond.dirty_price_from_ytm(
            valuation_date, current_rate, YTMCalcType.US_STREET
        )
        
        # Monte Carlo valuation of callable bond
        callable_values = []
        call_statistics = {
            'call_frequency': 0,
            'call_dates': [],
            'call_values': []
        }
        
        for path_name in rates_df.columns:
            rate_path = rates_df[path_name].values
            bond_value, was_called, call_info = self._value_single_path(
                valuation_date, rate_path
            )
            callable_values.append(bond_value)
            
            if was_called:
                call_statistics['call_frequency'] += 1
                call_statistics['call_dates'].append(call_info['call_date'])
                call_statistics['call_values'].append(call_info['call_value'])
        
        callable_bond_value = np.mean(callable_values)
        call_statistics['call_frequency'] /= len(callable_values)
        
        additional_stats = {
            'call_probability': call_statistics['call_frequency'],
            'option_value': straight_bond_value - callable_bond_value,
            'callable_values_std': np.std(callable_values),
            'call_statistics': call_statistics
        }
        
        return callable_bond_value, straight_bond_value, additional_stats
    
    
    def _value_single_path(
        self, 
        valuation_date: Date, 
        rate_path: np.ndarray,
    ) -> tuple[float, bool, dict]:
        """Value the bond along a single interest rate path using path-dependent discounting.
        
        Returns:
            Tuple of (bond_value, was_called, call_info)
        """
        # Get all nominal cash flows and dates from the underlying bond
        # FinancePy Bond cash flows are per unit face value, so scale by actual face value
        all_cf_dates = self.callable_bond.bond.cpn_dts
        all_cf_amounts = [cf * self.callable_bond.face_value for cf in self.callable_bond.bond.flow_amounts]
        
        # Fix: FinancePy doesn't include principal in the final payment, so add it manually
        if (len(all_cf_dates) > 0 and 
            all_cf_dates[-1] == self.callable_bond.maturity_date and 
            len(all_cf_amounts) > 0):
            # Check if last payment is just a coupon (needs principal added)
            expected_coupon = self.callable_bond.coupon_rate * self.callable_bond.face_value
            if abs(all_cf_amounts[-1] - expected_coupon) < 0.01:
                all_cf_amounts[-1] += self.callable_bond.face_value
        
        # Check each call date to see if bond should be called
        for call_date in self.callable_bond.call_dates:
            if call_date <= valuation_date:
                continue  # Already past this call date or on valuation date

            # Calculate years from valuation to call date for rate_at_call
            years_to_call = (call_date - valuation_date) / 365.25 # Approximate for indexing
            time_index = min(int(years_to_call * 365), len(rate_path) - 1)
            if time_index < 0: # Should not happen if call_date > valuation_date
                continue
                
            rate_at_call = rate_path[time_index] if time_index < len(rate_path) else rate_path[-1]
            
            # Calculate continuation value (value if not called at this call_date)
            # This uses FinancePy's YTM-based pricing for the bond from call_date onwards,
            # using the simulated rate_at_call as the YTM. This is a common simplification.
            continuation_value = self.callable_bond.dirty_price_from_ytm(
                call_date, rate_at_call, YTMCalcType.US_STREET
            )
            
            # Call value (par + premium)
            call_price = self.callable_bond.face_value + self.call_premium
            
            # Issuer calls if it's beneficial (continuation value > call price)
            if continuation_value > call_price * 1.02:  # 2% buffer
                # Bond is called. Calculate PV of (coupons up to call_date + call_price at call_date)
                
                effective_cf_dates = []
                effective_cf_amounts = []
                coupon_on_call_date = 0.0

                # Process each cash flow from the bond
                for i, (bond_cf_date, cf_amount) in enumerate(zip(all_cf_dates, all_cf_amounts)):
                    if bond_cf_date <= valuation_date:
                        continue  # Skip past cash flows
                    
                    if bond_cf_date < call_date:
                        # This is a coupon payment before call date
                        effective_cf_dates.append(bond_cf_date)
                        effective_cf_amounts.append(cf_amount)
                    elif bond_cf_date == call_date:
                        # If call date coincides with a coupon date, include the coupon
                        # The final payment at maturity typically includes both coupon and principal
                        # We need to extract just the coupon part if this is the maturity date
                        if bond_cf_date == self.callable_bond.maturity_date:
                            # This is the final payment which includes both coupon and principal
                            # Extract just the coupon portion
                            coupon_per_period = (self.callable_bond.bond._coupon / 
                                               self.callable_bond.bond._frequency * 
                                               self.callable_bond.face_value)
                            coupon_on_call_date = coupon_per_period
                        else:
                            # This is just a regular coupon payment
                            coupon_on_call_date = cf_amount
                    # For dates after call_date, we ignore them as the bond is called

                # Add the call payment (call price + any coupon due on call date)
                effective_cf_dates.append(call_date)
                effective_cf_amounts.append(call_price + coupon_on_call_date)
                
                called_bond_value = self._pv_of_future_cash_flows(
                    effective_cf_dates, effective_cf_amounts, valuation_date, rate_path
                )
                return (called_bond_value, True, 
                       {'call_date': call_date, 'call_value': call_price + coupon_on_call_date})
        
        # Bond was not called - value is PV of all remaining straight bond cash flows
        # (coupons + principal at maturity) from valuation_date onwards.
        
        cf_dates_not_called = []
        cf_amounts_not_called = []
        
        for i, (bond_cf_date, cf_amount) in enumerate(zip(all_cf_dates, all_cf_amounts)):
            if bond_cf_date > valuation_date:
                cf_dates_not_called.append(bond_cf_date)
                cf_amounts_not_called.append(cf_amount)
                
        not_called_bond_value = self._pv_of_future_cash_flows(
            cf_dates_not_called, cf_amounts_not_called, valuation_date, rate_path
        )
        return not_called_bond_value, False, {}


def calculate_equivalent_initial_short_rate(
    straight_bond: Bond,
    ytm: float,
    valuation_date: Date,
    discount_mode: str = "discrete",
    tolerance: float = 1e-6,
    max_iterations: int = 50
) -> tuple[float, dict]:
    """Calculate equivalent initial short rate for Monte Carlo simulation.
    
    This function finds a flat short rate that, when used in the Monte Carlo 
    discounting framework (_pv_of_future_cash_flows), produces the same present 
    value as the analytical bond valuation.
    
    Args:
        straight_bond: FinancePy Bond object
        ytm: Yield to maturity of the bond  
        valuation_date: Date to value the bond
        discount_mode: "discrete" or "continuous" discounting mode
        tolerance: Convergence tolerance for the optimization
        max_iterations: Maximum number of optimization iterations
        
    Returns:
        Tuple of (equivalent_short_rate, solution_info)
    """
    
    # Get analytical bond price using FinancePy
    analytical_pv = straight_bond.dirty_price_from_ytm(
        valuation_date, ytm, YTMCalcType.US_STREET
    )
    
    # Extract bond cash flows and dates
    bond_cf_dates = straight_bond.cpn_dts
    bond_cf_amounts = [cf * straight_bond.par for cf in straight_bond.flow_amounts]
    
    # Fix: FinancePy doesn't include principal in the final payment, add it manually
    if (len(bond_cf_dates) > 0 and 
        bond_cf_dates[-1] == straight_bond.maturity_dt and 
        len(bond_cf_amounts) > 0):
        # Check if last payment is just a coupon (needs principal added)
        expected_coupon = straight_bond.cpn * straight_bond.par
        if abs(bond_cf_amounts[-1] - expected_coupon) < 0.01:
            bond_cf_amounts[-1] += straight_bond.par
    
    # Create a dummy rate simulator to get the dt parameter
    # We need this for the _pv_of_future_cash_flows method
    dummy_simulator = InterestRateSimulator(
        r0=ytm,  # Initial guess
        days=365 * 10,  # Sufficient duration for bond maturity
        num_paths=1,
        model="gbm"
    )
    
    # Create a dummy callable bond valuer to access _pv_of_future_cash_flows
    dummy_callable_bond = CallableBond(
        issue_date=valuation_date,
        maturity_date=straight_bond.maturity_dt,
        coupon_rate=straight_bond.cpn,
        call_protection_years=1,
        face_value=straight_bond.par
    )
    
    dummy_valuer = CallableBondValuer(dummy_callable_bond, dummy_simulator)
    
    def objective_function(flat_rate: float) -> float:
        """Objective function: difference between MC PV and analytical PV."""
        
        # Calculate maximum days needed for discounting
        max_days_to_maturity = int((straight_bond.maturity_dt - valuation_date)) + 1
        
        # Create flat rate path
        flat_rate_path = np.full(max_days_to_maturity, flat_rate)
        
        # Calculate PV using Monte Carlo discounting method
        mc_pv = dummy_valuer._pv_of_future_cash_flows(
            bond_cf_dates,
            bond_cf_amounts,
            valuation_date,
            flat_rate_path,
            discount_mode=discount_mode
        )
        
        return mc_pv - analytical_pv
    
    # Test the objective function with the initial guess (YTM)
    print(f"Testing equivalent short rate calculation...")
    print(f"Analytical bond PV: {analytical_pv:.6f}")
    
    initial_error = objective_function(ytm)
    print(f"Initial guess ({ytm:.4%}) error: {initial_error:.6f}")
    
    # Set up bounds for optimization
    lower_bound = max(0.0001, ytm - 0.05)  # At least 1bp, YTM - 5%
    upper_bound = ytm + 0.05  # YTM + 5%
    
    # Test bounds
    f_lower = objective_function(lower_bound)
    f_upper = objective_function(upper_bound)
    
    print(f"f({lower_bound:.4%}) = {f_lower:.6f}")
    print(f"f({upper_bound:.4%}) = {f_upper:.6f}")
    
    # Expand bounds if they don't bracket the root
    if f_lower * f_upper > 0:
        if f_lower > 0:  # Both positive, need lower rate
            lower_bound = max(0.0001, ytm - 0.10)
        else:  # Both negative, need higher rate  
            upper_bound = ytm + 0.10
        
        f_lower = objective_function(lower_bound)
        f_upper = objective_function(upper_bound)
        print(f"Adjusted bounds: f({lower_bound:.4%}) = {f_lower:.6f}, f({upper_bound:.4%}) = {f_upper:.6f}")
    
    # Solve for equivalent short rate using Brent's method
    try:
        equivalent_rate = optimize.brentq(
            objective_function,
            a=lower_bound,
            b=upper_bound,
            xtol=tolerance,
            maxiter=max_iterations
        )
        
        # Calculate final verification
        max_days_to_maturity = int((straight_bond.maturity_dt - valuation_date)) + 1
        flat_rate_path = np.full(max_days_to_maturity, equivalent_rate)
        
        final_mc_pv = dummy_valuer._pv_of_future_cash_flows(
            bond_cf_dates,
            bond_cf_amounts,
            valuation_date,
            flat_rate_path,
            discount_mode=discount_mode
        )
        
        solution_info = {
            'equivalent_short_rate': equivalent_rate,
            'analytical_pv': analytical_pv,
            'mc_pv': final_mc_pv,
            'error': abs(final_mc_pv - analytical_pv),
            'relative_error': abs(final_mc_pv - analytical_pv) / analytical_pv,
            'ytm': ytm,
            'rate_difference': equivalent_rate - ytm,
            'discount_mode': discount_mode,
            'num_cash_flows': len(bond_cf_dates) - 1  # Exclude issue date
        }
        
        return equivalent_rate, solution_info
        
    except ValueError as e:
        raise ValueError(f"Could not find equivalent short rate: {e}")


def solve_callable_bond_coupon(
    issue_date: Date,
    maturity_date: Date,
    target_value: float,
    risk_free_rate: float,
    call_protection_years: int = 1,
    initial_guess: float = 0.045,
    tolerance: float = 1e-4
) -> tuple[float, dict]:
    """Solve for the coupon rate that makes callable bond value equal to target.
    
    Args:
        issue_date: Bond issue date
        maturity_date: Bond maturity date
        target_value: Target bond value
        risk_free_rate: Risk-free rate for discounting
        call_protection_years: Years of call protection
        initial_guess: Initial guess for coupon rate
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (optimal_coupon_rate, solution_info)
    """
    
    def objective_function(coupon_rate: float) -> float:
        """Objective function: difference between callable bond value and target."""
        
        # Create callable bond
        callable_bond = CallableBond(
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_rate=coupon_rate,
            call_protection_years=call_protection_years,
            freq_type=FrequencyTypes.ANNUAL,
            day_count_type=DayCountTypes.THIRTY_E_360
        )
        
        # Create interest rate simulator
        rate_simulator = InterestRateSimulator(
            r0=risk_free_rate,
            mu=0,  # 0 for GBM
            sigma=0.02,  # 2% volatility
            days=365 * 5,  # 5 years
            num_paths=1000,  
            model="gbm",
            kappa=0.1
        )
        
        # Create valuer
        valuer = CallableBondValuer(callable_bond, rate_simulator)
        
        # Value the bond
        callable_value, straight_value, stats = valuer.value_bond(
            issue_date, risk_free_rate, num_simulations=1000
        )
        
        return callable_value - target_value
    
    # Test the bounds first to ensure they bracket the root
    print("Testing optimization bounds...")
    
    # Start with a wider range and test
    lower_bound = 0.01  # 1%
    upper_bound = 0.12  # 12%
    
    f_lower = objective_function(lower_bound)
    f_upper = objective_function(upper_bound)
    
    print(f"f({lower_bound:.1%}) = {f_lower:.4f}")
    print(f"f({upper_bound:.1%}) = {f_upper:.4f}")
    
    # Adjust bounds if needed
    if f_lower * f_upper > 0:
        # Need to find bounds that bracket the root
        if f_lower > 0:  # Both positive, need lower bound
            lower_bound = 0.001
        else:  # Both negative, need higher bound
            upper_bound = 0.20
        
        f_lower = objective_function(lower_bound)
        f_upper = objective_function(upper_bound)
        print(f"Adjusted: f({lower_bound:.1%}) = {f_lower:.4f}, f({upper_bound:.1%}) = {f_upper:.4f}")
    
    # Solve using Brent's method
    try:
        optimal_coupon = optimize.brentq(
            objective_function,
            a=lower_bound,
            b=upper_bound,
            xtol=tolerance,
            maxiter=50
        )
        
        # Get final solution info
        final_callable_bond = CallableBond(
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_rate=optimal_coupon,
            call_protection_years=call_protection_years,
            freq_type=FrequencyTypes.ANNUAL,
            day_count_type=DayCountTypes.THIRTY_E_360
        )
        
        rate_simulator = InterestRateSimulator(
            r0=risk_free_rate,
            mu=0,
            sigma=0.02,
            days=365 * 5,
            num_paths=1000,
            model="gbm",
            kappa=0.1
        )
        
        valuer = CallableBondValuer(final_callable_bond, rate_simulator)
        final_value, straight_value, stats = valuer.value_bond(
            issue_date, risk_free_rate, num_simulations=1000
        )
        
        solution_info = {
            'optimal_coupon': optimal_coupon,
            'callable_bond_value': final_value,
            'straight_bond_value': straight_value,
            'target_value': target_value,
            'error': abs(final_value - target_value),
            'option_value': stats['option_value'],
            'call_probability': stats['call_probability']
        }
        
        return optimal_coupon, solution_info
        
    except ValueError as e:
        raise ValueError(f"Could not find solution: {e}")


def analyze_callable_bond_dynamics(
    issue_date: Date,
    maturity_date: Date,
    risk_free_rate: float,
    coupon_range: list[float] = None
) -> None:
    """Analyze the relationship between coupon rates and callable bond values."""
    
    if coupon_range is None:
        coupon_range = [0.02, 0.03, 0.04, 0.045, 0.046, 0.05, 0.06, 0.08]
    
    print("=== Callable Bond Dynamics Analysis ===")
    print(f"Risk-free rate: {risk_free_rate:.2%}")
    print(f"Coupon Rate | Straight Bond | Callable Bond | Option Value | Call Prob")
    print("-" * 70)
    
    for coupon in coupon_range:
        # Create callable bond
        callable_bond = CallableBond(
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_rate=coupon,
            call_protection_years=1,
            freq_type=FrequencyTypes.ANNUAL,
            day_count_type=DayCountTypes.THIRTY_E_360
        )
        
        # Create rate simulator
        rate_simulator = InterestRateSimulator(
            r0=risk_free_rate,
            mu=0,
            sigma=0.02,
            days=365 * 5,
            num_paths=1000,
            model="gbm",
            kappa=0.1
        )
        
        # Create valuer
        valuer = CallableBondValuer(callable_bond, rate_simulator)
        
        # Value the bond
        callable_value, straight_value, stats = valuer.value_bond(
            issue_date, risk_free_rate, num_simulations=1000
        )
        
        print(f"{coupon:8.2%} | {straight_value:11.4f} | {callable_value:11.4f} | "
              f"{stats['option_value']:10.4f} | {stats['call_probability']:.1%}")


def main():
    # Test the implementation with the specific scenario
    print("=== Callable Bond Valuer Implementation Test ===\n")
    issue_date = Date(27, 5, 2025)
    maturity_date = Date(27, 5, 2030)  # 5 years
    
    # Parameters from the problem
    risk_free_rate = 0.0435  # 4.35%
    straight_bond_coupon = 0.046  # 4.60%
    
    # Create a straight bond for testing
    straight_bond = Bond(
        issue_dt=issue_date,
        maturity_dt=maturity_date,
        coupon=straight_bond_coupon,
        freq_type=FrequencyTypes.ANNUAL,
        dc_type=DayCountTypes.THIRTY_E_360
    )
    target_value = 100

    print("Problem Parameters:")
    print(f"Risk-free rate: {risk_free_rate:.2%}")
    print(f"Straight bond coupon: {straight_bond_coupon:.2%}")
    print(f"Target bond value: {target_value}")
    print(f"Call protection: 1 year")
    print()

    # Test the new equivalent initial short rate calculation
    print("\n=== Testing Equivalent Initial Short Rate Calculation ===")
        
    # Test both discrete and continuous modes
    for discount_mode in ["discrete", "continuous"]:
        print(f"\n--- Testing {discount_mode.title()} Mode ---")
        try:
            equivalent_rate, solution_info = calculate_equivalent_initial_short_rate(
                straight_bond=straight_bond,
                ytm=straight_bond_coupon,  # Use bond coupon as YTM
                valuation_date=issue_date,
                discount_mode=discount_mode,
                tolerance=1e-6
            )
            
            print(f"Input YTM: {solution_info['ytm']:.4%}")
            print(f"Equivalent short rate: {solution_info['equivalent_short_rate']:.4%}")
            print(f"Rate difference: {solution_info['rate_difference']:.4%}")
            print(f"Analytical PV: {solution_info['analytical_pv']:.6f}")
            print(f"Monte Carlo PV: {solution_info['mc_pv']:.6f}")
            print(f"Absolute error: {solution_info['error']:.8f}")
            print(f"Relative error: {solution_info['relative_error']:.8%}")
            print(f"Number of cash flows: {solution_info['num_cash_flows']}")
            
        except Exception as e:
            print(f"Error in {discount_mode} mode: {e}")

    try:
        # Analyze callable bond dynamics first
        analyze_callable_bond_dynamics(issue_date, maturity_date, equivalent_rate)        
        
        # Solve for callable bond coupon using Monte Carlo optimization
        print("\n=== Solving for Callable Bond Coupon ===")
        result = solve_callable_bond_coupon(
            issue_date=issue_date,
            maturity_date=maturity_date,
            target_value=target_value,
            risk_free_rate=equivalent_rate,
            call_protection_years=1
        )
        
        optimal_coupon, solution_info = result
        print("\nSolution Results:")
        print(f"Optimal callable bond coupon: {optimal_coupon:.4%}")
        print(f"Callable bond value: {solution_info['callable_bond_value']:.4f}")
        print(f"Straight bond value: {solution_info['straight_bond_value']:.4f}")
        print(f"Target value: {solution_info['target_value']:.4f}")
        print(f"Pricing error: {solution_info['error']:.6f}")
        print(f"Option value: {solution_info['option_value']:.4f}")
        print(f"Call probability: {solution_info['call_probability']:.2%}")
        
        print(f"\nConclusion: To achieve a bond value of {target_value:.0f}, "
              f"the callable bond should have a coupon rate of {optimal_coupon:.4%}.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        print("Debugging information printed above.") 

if __name__ == "__main__":
    main()

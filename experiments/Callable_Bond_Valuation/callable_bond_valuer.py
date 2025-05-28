import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

# Import FinancePy classes
try:
    from financepy.products.bonds.bond import Bond, YTMCalcType
    from financepy.utils.date import Date
    from financepy.utils.frequency import FrequencyTypes
    from financepy.utils.day_count import DayCountTypes
    FINANCEPY_AVAILABLE = True
except ImportError:
    print("Warning: FinancePy not found. Using mock classes for development.")
    FINANCEPY_AVAILABLE = False
    # Mock classes for development/testing
    class Date:
        def __init__(self, day, month, year):
            self.day, self.month, self.year = day, month, year
        def add_years(self, years): return self
        def __sub__(self, other): return 365
        def __add__(self, days): return self
    
    class Bond:
        def __init__(self, *args, **kwargs): pass
        def dirty_price_from_ytm(self, *args): return 100.0
        def yield_to_maturity(self, *args): return 0.045
    
    class YTMCalcType:
        US_STREET = "US_STREET"
    
    class FrequencyTypes:
        ANNUAL = "ANNUAL"
    
    class DayCountTypes:
        THIRTY_E_360 = "THIRTY_E_360"


class InterestRateSimulator:
    """A class for simulating interest rate movements using various models.
    
    This extends the framework from trade_brown_motion.py to handle interest rate 
    simulation instead of just price simulation, which is more appropriate for 
    callable bond valuation.
    """
    
    def __init__(
        self,
        r0: float = 0.04,  # Initial interest rate
        mu: float = 0.0,   # Mean reversion level (for mean-reverting models)
        sigma: float = 0.02,  # Volatility
        days: int = 365 * 5,  # 5 years of simulation
        days_of_year: int = 365,
        num_paths: int = 1000,
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
            if self.days <= 10000:
                dates = pd.date_range(start="today", periods=self.days, freq="D")
                self.rates_df = pd.DataFrame(
                    rates.T,
                    index=dates,
                    columns=[f"path_{i+1}" for i in range(self.num_paths)]
                )
            else:
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
        straight_bond_value = self._calculate_straight_bond_value(
            valuation_date, current_rate
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
                valuation_date, rate_path, current_rate
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
    
    def _calculate_straight_bond_value(
        self, 
        valuation_date: Date, 
        discount_rate: float
    ) -> float:
        """Calculate the value of the bond without call option."""
        return self.callable_bond.dirty_price_from_ytm(
            valuation_date, discount_rate, YTMCalcType.US_STREET
        )
    
    def _value_single_path(
        self, 
        valuation_date: Date, 
        rate_path: np.ndarray,
        initial_rate: float
    ) -> tuple[float, bool, dict]:
        """Value the bond along a single interest rate path.
        
        Returns:
            Tuple of (bond_value, was_called, call_info)
        """
        if FINANCEPY_AVAILABLE:
            # Calculate years to maturity for proper indexing
            years_to_maturity = (self.callable_bond.maturity_date - valuation_date) / 365.25
        else:
            years_to_maturity = 5  # Default for mock
        
        # Check each call date to see if bond should be called
        for i, call_date in enumerate(self.callable_bond.call_dates):
            if call_date <= valuation_date:
                continue  # Already past this call date
                
            # Calculate years from valuation to call date
            if FINANCEPY_AVAILABLE:
                years_to_call = (call_date - valuation_date) / 365.25
            else:
                years_to_call = 1 + i  # Mock calculation
            
            # Find the rate at this call date (approximate using path index)
            time_index = min(int(years_to_call * 365), len(rate_path) - 1)
            if time_index < 0:
                continue
                
            rate_at_call = rate_path[time_index] if time_index < len(rate_path) else rate_path[-1]
            
            # Calculate continuation value (value if not called)
            continuation_value = self.callable_bond.dirty_price_from_ytm(
                call_date, rate_at_call, YTMCalcType.US_STREET
            )
            
            # Call value (par + premium)
            call_value = self.callable_bond.face_value + self.call_premium
            
            # Issuer calls if it's beneficial (continuation value > call value)
            # Adding a small buffer to make calling decision more realistic
            if continuation_value > call_value * 1.02:  # 2% buffer
                # Bond is called - return discounted call value
                discount_factor = np.exp(-initial_rate * years_to_call)
                return (call_value * discount_factor, True, 
                       {'call_date': call_date, 'call_value': call_value})
        
        # Bond was not called - return discounted final value
        final_rate = initial_rate  # Use initial rate for discounting
        bond_value = self.callable_bond.dirty_price_from_ytm(
            valuation_date, final_rate, YTMCalcType.US_STREET
        )
        
        return bond_value, False, {}


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
            mu=risk_free_rate,  # Mean-reverting to current rate
            sigma=0.02,  # 2% volatility
            days=365 * 5,  # 5 years
            num_paths=100,  # Fewer paths for speed during optimization
            model="vasicek",
            kappa=0.1
        )
        
        # Create valuer
        valuer = CallableBondValuer(callable_bond, rate_simulator)
        
        # Value the bond
        callable_value, straight_value, stats = valuer.value_bond(
            issue_date, risk_free_rate, num_simulations=100
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
            mu=risk_free_rate,
            sigma=0.02,
            days=365 * 5,
            num_paths=1000,
            model="vasicek",
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


def validate_financepy_bond(
    issue_date: Date,
    maturity_date: Date,
    coupon_rate: float,
    discount_rate: float
) -> tuple[float, float]:
    """Validate bond pricing using both our implementation and FinancePy directly.
    
    Returns:
        Tuple of (our_implementation_value, financepy_direct_value)
    """
    
    # Our implementation
    callable_bond = CallableBond(
        issue_date=issue_date,
        maturity_date=maturity_date,
        coupon_rate=coupon_rate,
        freq_type=FrequencyTypes.ANNUAL,
        day_count_type=DayCountTypes.THIRTY_E_360
    )
    our_value = callable_bond.dirty_price_from_ytm(
        issue_date, discount_rate, YTMCalcType.US_STREET
    )
    
    # Direct FinancePy implementation
    financepy_bond = Bond(
        issue_dt=issue_date,
        maturity_dt=maturity_date,
        coupon=coupon_rate,
        freq_type=FrequencyTypes.ANNUAL,
        dc_type=DayCountTypes.THIRTY_E_360
    )
    financepy_value = financepy_bond.dirty_price_from_ytm(
        issue_date, discount_rate, YTMCalcType.US_STREET
    )
    
    return our_value, financepy_value


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
            mu=risk_free_rate,
            sigma=0.02,
            days=365 * 5,
            num_paths=200,
            model="vasicek",
            kappa=0.1
        )
        
        # Create valuer
        valuer = CallableBondValuer(callable_bond, rate_simulator)
        
        # Value the bond
        callable_value, straight_value, stats = valuer.value_bond(
            issue_date, risk_free_rate, num_simulations=200
        )
        
        print(f"{coupon:8.2%} | {straight_value:11.4f} | {callable_value:11.4f} | "
              f"{stats['option_value']:10.4f} | {stats['call_probability']:7.1%}")


def simple_callable_bond_solver(
    issue_date: Date,
    maturity_date: Date,
    target_value: float,
    risk_free_rate: float,
    straight_bond_coupon: float
) -> tuple[float, dict]:
    """Simple approach: find coupon that makes callable bond equal to straight bond value."""
    
    # First, get the straight bond value at the given coupon rate
    straight_bond = CallableBond(
        issue_date=issue_date,
        maturity_date=maturity_date,
        coupon_rate=straight_bond_coupon,
        freq_type=FrequencyTypes.ANNUAL,
        day_count_type=DayCountTypes.THIRTY_E_360
    )
    
    straight_bond_value = straight_bond.dirty_price_from_ytm(
        issue_date, risk_free_rate, YTMCalcType.US_STREET
    )
    
    print(f"Straight bond ({straight_bond_coupon:.2%} coupon) value: {straight_bond_value:.4f}")
    
    def objective_function(coupon_rate: float) -> float:
        """Find callable bond coupon that gives same value as straight bond."""
        
        callable_bond = CallableBond(
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_rate=coupon_rate,
            call_protection_years=1,
            freq_type=FrequencyTypes.ANNUAL,
            day_count_type=DayCountTypes.THIRTY_E_360
        )
        
        rate_simulator = InterestRateSimulator(
            r0=risk_free_rate,
            mu=risk_free_rate,
            sigma=0.015,  # Lower volatility for more stable results
            days=365 * 5,
            num_paths=100,
            model="vasicek",
            kappa=0.05  # Slower mean reversion
        )
        
        valuer = CallableBondValuer(callable_bond, rate_simulator)
        callable_value, _, _ = valuer.value_bond(
            issue_date, risk_free_rate, num_simulations=100
        )
        
        return callable_value - straight_bond_value
    
    # Test a range to find the root
    print("Testing callable bond values for different coupon rates:")
    test_coupons = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
    
    for test_coupon in test_coupons:
        try:
            obj_val = objective_function(test_coupon)
            print(f"Coupon {test_coupon:.2%}: Callable - Straight = {obj_val:.4f}")
        except:
            print(f"Coupon {test_coupon:.2%}: Error in calculation")
    
    # Find the root - based on analysis, we expect the crossover around 2-3%
    try:
        # Use the appropriate range where callable > straight transitions to callable < straight
        optimal_coupon = optimize.brentq(
            objective_function,
            a=0.015,  # Lower bound where callable > straight
            b=0.035,  # Upper bound where callable < straight
            xtol=1e-4,
            maxiter=20
        )
        
        # Get final results
        final_callable_bond = CallableBond(
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_rate=optimal_coupon,
            call_protection_years=1,
            freq_type=FrequencyTypes.ANNUAL,
            day_count_type=DayCountTypes.THIRTY_E_360
        )
        
        rate_simulator = InterestRateSimulator(
            r0=risk_free_rate,
            mu=risk_free_rate,
            sigma=0.015,
            days=365 * 5,
            num_paths=500,
            model="vasicek",
            kappa=0.05
        )
        
        valuer = CallableBondValuer(final_callable_bond, rate_simulator)
        final_callable_value, final_straight_value, final_stats = valuer.value_bond(
            issue_date, risk_free_rate, num_simulations=500
        )
        
        solution_info = {
            'optimal_coupon': optimal_coupon,
            'callable_bond_value': final_callable_value,
            'straight_bond_reference_value': straight_bond_value,
            'straight_bond_at_optimal_coupon': final_straight_value,
            'target_value': target_value,
            'error': abs(final_callable_value - straight_bond_value),
            'option_value': final_stats['option_value'],
            'call_probability': final_stats['call_probability']
        }
        
        return optimal_coupon, solution_info
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # If optimization fails, let's find the closest value manually
        print("Attempting manual search for best match...")
        
        best_coupon = None
        best_error = float('inf')
        best_info = {}
        
        for test_coupon in np.arange(0.01, 0.05, 0.002):
            try:
                obj_val = objective_function(test_coupon)
                if abs(obj_val) < best_error:
                    best_error = abs(obj_val)
                    best_coupon = test_coupon
                    
                    # Get detailed info for best match
                    if best_error < 1.0:  # If reasonable match
                        callable_bond = CallableBond(
                            issue_date=issue_date,
                            maturity_date=maturity_date,
                            coupon_rate=best_coupon,
                            call_protection_years=1,
                            freq_type=FrequencyTypes.ANNUAL,
                            day_count_type=DayCountTypes.THIRTY_E_360
                        )
                        
                        rate_simulator = InterestRateSimulator(
                            r0=risk_free_rate,
                            mu=risk_free_rate,
                            sigma=0.015,
                            days=365 * 5,
                            num_paths=300,
                            model="vasicek",
                            kappa=0.05
                        )
                        
                        valuer = CallableBondValuer(callable_bond, rate_simulator)
                        callable_value, straight_value_at_best, stats = valuer.value_bond(
                            issue_date, risk_free_rate, num_simulations=300
                        )
                        
                        best_info = {
                            'optimal_coupon': best_coupon,
                            'callable_bond_value': callable_value,
                            'straight_bond_reference_value': straight_bond_value,
                            'straight_bond_at_optimal_coupon': straight_value_at_best,
                            'target_value': target_value,
                            'error': best_error,
                            'option_value': stats['option_value'],
                            'call_probability': stats['call_probability']
                        }
            except:
                continue
        
        if best_coupon is not None:
            print(f"Best match found: {best_coupon:.3%} with error: {best_error:.4f}")
            return best_coupon, best_info
        else:
            return None, {}


if __name__ == "__main__":
    # Test the implementation with the specific scenario
    print("=== Callable Bond Valuer Implementation Test ===\n")
    
    # Create dates (using simple Date construction)
    try:
        issue_date = Date(1, 1, 2024)
        maturity_date = Date(1, 1, 2029)  # 5 years
        print("Using FinancePy dates")
    except:
        # Mock dates for testing without FinancePy
        issue_date = "2024-01-01"
        maturity_date = "2029-01-01"
        print("Using mock dates for testing")
    
    # Parameters from the problem
    risk_free_rate = 0.0435  # 4.35%
    straight_bond_coupon = 0.046  # 4.60%
    target_value = 100.0
    
    print("Problem Parameters:")
    print(f"Risk-free rate: {risk_free_rate:.2%}")
    print(f"Straight bond coupon: {straight_bond_coupon:.2%}")
    print(f"Target bond value: {target_value}")
    print(f"Call protection: 1 year")
    print()
    
    try:
        # 2.1: Validate FinancePy bond pricing
        print("=== 2.1: Validating FinancePy Bond Pricing ===")
        our_value, financepy_value = validate_financepy_bond(
            issue_date, maturity_date, straight_bond_coupon, risk_free_rate
        )
        print(f"Our implementation value: {our_value:.4f}")
        print(f"FinancePy direct value: {financepy_value:.4f}")
        print(f"Difference: {abs(our_value - financepy_value):.6f}")
        print(f"Both values > 100: {our_value > 100 and financepy_value > 100}")
        print()
        
        # Analyze callable bond dynamics first
        analyze_callable_bond_dynamics(issue_date, maturity_date, risk_free_rate)
        print()
        
        # 2: Solve for callable bond coupon using simpler approach
        print("=== 2: Solving for Callable Bond Coupon (Simple Approach) ===")
        result = simple_callable_bond_solver(
            issue_date=issue_date,
            maturity_date=maturity_date,
            target_value=target_value,
            risk_free_rate=risk_free_rate,
            straight_bond_coupon=straight_bond_coupon
        )
        
        if result[0] is not None:
            optimal_coupon, solution_info = result
            print("\nSolution Results:")
            print(f"Optimal callable bond coupon: {optimal_coupon:.4%}")
            print(f"Callable bond value: {solution_info['callable_bond_value']:.4f}")
            print(f"Straight bond (4.6%) value: {solution_info['straight_bond_reference_value']:.4f}")
            print(f"Straight bond (optimal coupon) value: {solution_info['straight_bond_at_optimal_coupon']:.4f}")
            print(f"Pricing error: {solution_info['error']:.6f}")
            print(f"Option value: {solution_info['option_value']:.4f}")
            print(f"Call probability: {solution_info['call_probability']:.2%}")
            
            print(f"\nConclusion: To have the same fair value as a 4.60% straight bond, "
                  f"the callable bond should have a coupon rate of {optimal_coupon:.4%}.")
        else:
            print("Could not find a solution.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        print("Debugging information printed above.") 
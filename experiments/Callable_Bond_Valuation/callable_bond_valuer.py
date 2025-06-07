import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

# Import FinancePy classes
from financepy.products.bonds.bond import Bond, YTMCalcType
from financepy.utils.date import Date
from financepy.utils.frequency import FrequencyTypes
from financepy.utils.day_count import DayCountTypes

from experiments.Callable_Bond_Valuation.interest_rate_simulator import (
    InterestRateSimulator,
)


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
            dc_type=day_count_type,
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

    def cashflow_since_date(self, date: Date) -> tuple[list[Date], list[float]]:
        """
        Get bond cash flows since a given date(exclusive).
        Cash flow on valuation date is received by the bond holder on the previous date.
        FinancePy cash flows are per unit face value.
        The cashflow returned by this method is scaled by actual face value.
        Args:
            date: Date to get cash flows since
        Returns:
            Tuple of (list of cash flow dates, list of cash flow amounts)
        """
        # Get financepy bond cash flows and dates from the underlying bond
        # FinancePy Bond cash flows are per unit face value, so scale by actual face value
        financepy_cf_dates = self.bond.cpn_dts
        financepy_cf_amounts = [cf * self.face_value for cf in self.bond.flow_amounts]
        cf_dates = []
        cf_amounts = []
        for cf_date, cf_amount in zip(financepy_cf_dates, financepy_cf_amounts):
            if cf_date > date:
                cf_dates.append(cf_date)
                cf_amounts.append(cf_amount)

        # FinancePy doesn't include principal in the final payment, add it manually
        cf_amounts[-1] += self.face_value

        return cf_dates, cf_amounts

    def dirty_price_from_ytm(
        self,
        settle_date: Date,
        ytm: float,
        convention: YTMCalcType = YTMCalcType.US_STREET,
    ) -> float:
        """Calculate dirty price using FinancePy's implementation."""
        return self.bond.dirty_price_from_ytm(settle_date, ytm, convention)

    def yield_to_maturity(
        self,
        settle_date: Date,
        clean_price: float,
        convention: YTMCalcType = YTMCalcType.US_STREET,
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

    @staticmethod
    def pv_of_future_cash_flows(
        cash_flow_dates: list[Date],
        cash_flow_amounts: list[float],
        valuation_date: Date,
        rate_path: np.ndarray,
        dt: float,
        discount_mode: str = "discrete",
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
            dt: Time step (unit: year) for the rate path
            discount_mode: "discrete" or "continuous"
        Returns:
            Present value of the cash flows
        """
        pv = 0.0
        max_path_idx = len(rate_path)

        for cf_date, cf_amount in zip(cash_flow_dates, cash_flow_amounts):
            # Cash flow on valuation date is received by the bond holder on the previous date
            if cf_date <= valuation_date:
                continue

            # Calculate number of days from valuation_date to cf_date
            # rate_path[0] is for the first period from valuation_date
            days_to_cf = int(
                round((cf_date - valuation_date))
            )  # Date subtraction gives float days

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
                discount_factor = 1.0 / np.prod(1 + relevant_rates * dt)
            elif discount_mode == "continuous":
                discount_factor = np.exp(-np.sum(relevant_rates * dt))
            else:
                raise ValueError(f"Invalid discount mode: {discount_mode}")
            pv += cf_amount * discount_factor

        return pv

    @classmethod
    def calculate_equivalent_initial_short_rate(
        cls,
        straight_bond: Bond,
        dt: float,
        ytm: float,
        valuation_date: Date,
        discount_mode: str = "discrete",
        tolerance: float = 1e-6,
        max_iterations: int = 50,
    ) -> tuple[float, dict]:
        """Calculate equivalent initial short rate for Monte Carlo simulation.

        This method finds a flat short rate that, when used in the Monte Carlo
        discounting framework (CallableBondValuer.pv_of_future_cash_flows),
        produces the same present value as the analytical bond valuation.

        Args:
            straight_bond: FinancePy Bond object
            dt: Time step (unit: year) for the rate path
            ytm: Yield to maturity of the bond, for calculating the analytical bond price
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

        # Create a CallableBond from the straight bond to use cashflow_since_date()
        callable_bond = CallableBond(
            issue_date=straight_bond.issue_dt,
            maturity_date=straight_bond.maturity_dt,
            coupon_rate=straight_bond.cpn,
            call_protection_years=0,  # No call protection for straight bond equivalent
            face_value=straight_bond.par,
            freq_type=straight_bond.freq_type,
            day_count_type=straight_bond.dc_type,
        )

        # Use cashflow_since_date() to get bond cash flows
        bond_cf_dates, bond_cf_amounts = callable_bond.cashflow_since_date(
            valuation_date
        )

        def objective_function(flat_rate: float) -> float:
            """Objective function: difference between MC PV and analytical PV."""

            # Calculate maximum days needed for discounting
            max_days_to_maturity = int((straight_bond.maturity_dt - valuation_date)) + 1

            # Create flat rate path
            flat_rate_path = np.full(max_days_to_maturity, flat_rate)

            # Calculate PV using Monte Carlo discounting method
            mc_pv = cls.pv_of_future_cash_flows(
                bond_cf_dates,
                bond_cf_amounts,
                valuation_date,
                flat_rate_path,
                dt,
                discount_mode=discount_mode,
            )

            return mc_pv - analytical_pv

        # Set up bounds for optimization
        lower_bound = max(0.0001, ytm - 0.05)  # At least 1bp, YTM - 5%
        upper_bound = ytm + 0.05  # YTM + 5%

        # Test bounds
        f_lower = objective_function(lower_bound)
        f_upper = objective_function(upper_bound)

        # Expand bounds if they don't bracket the root
        if f_lower * f_upper > 0:
            if f_lower > 0:  # Both positive, need lower rate
                lower_bound = max(0.0001, ytm - 0.10)
            else:  # Both negative, need higher rate
                upper_bound = ytm + 0.10

            f_lower = objective_function(lower_bound)
            f_upper = objective_function(upper_bound)

        # Solve for equivalent short rate using Brent's method
        try:
            equivalent_rate = optimize.brentq(
                objective_function,
                a=lower_bound,
                b=upper_bound,
                xtol=tolerance,
                maxiter=max_iterations,
            )

            # Calculate final verification
            max_days_to_maturity = int((straight_bond.maturity_dt - valuation_date)) + 1
            flat_rate_path = np.full(max_days_to_maturity, equivalent_rate)

            final_mc_pv = cls.pv_of_future_cash_flows(
                bond_cf_dates,
                bond_cf_amounts,
                valuation_date,
                flat_rate_path,
                dt,
                discount_mode=discount_mode,
            )

            solution_info = {
                "equivalent_short_rate": equivalent_rate,
                "analytical_pv": analytical_pv,
                "mc_pv": final_mc_pv,
                "error": abs(final_mc_pv - analytical_pv),
                "relative_error": abs(final_mc_pv - analytical_pv) / analytical_pv,
                "ytm": ytm,
                "rate_difference": equivalent_rate - ytm,
                "discount_mode": discount_mode,
                "num_cash_flows": len(bond_cf_dates),
            }

            return equivalent_rate, solution_info

        except ValueError as e:
            raise ValueError(f"Could not find equivalent short rate: {e}")

    def value_bond(
        self,
        valuation_date: Date,
        straight_bond_ytm: float,
        num_simulations: int | None = None,
    ) -> tuple[float, float, dict]:
        """Value the callable bond using Monte Carlo simulation.

        Args:
            valuation_date: Date to value the bond
            straight_bond_ytm: Yield to maturity for straight bond
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
            valuation_date, straight_bond_ytm, YTMCalcType.US_STREET
        )

        # Monte Carlo valuation of callable bond
        callable_values = []
        call_statistics = {"call_frequency": 0, "call_dates": [], "call_values": []}

        for path_name in rates_df.columns:
            rate_path = rates_df[path_name].values
            bond_value, was_called, call_info = self._value_single_path(
                valuation_date, rate_path
            )
            callable_values.append(bond_value)

            if was_called:
                call_statistics["call_frequency"] += 1
                call_statistics["call_dates"].append(call_info["call_date"])
                call_statistics["call_values"].append(call_info["call_value"])

        callable_bond_value = np.mean(callable_values)
        call_statistics["call_frequency"] /= len(callable_values)

        additional_stats = {
            "call_probability": call_statistics["call_frequency"],
            "option_value": straight_bond_value - callable_bond_value,
            "callable_values_std": np.std(callable_values),
            "call_statistics": call_statistics,
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
        # Get all bond cash flows after valuation_date
        bond_cf_dates, bond_cf_amounts = self.callable_bond.cashflow_since_date(
            valuation_date
        )

        # Check each call date to see if bond should be called
        for call_date in self.callable_bond.call_dates:
            if call_date <= valuation_date:
                continue  # Already past this call date or on valuation date

            # Calculate continuation value (value if not called at this call_date)
            # Use Monte Carlo discounting for consistency with the overall framework

            # Get cash flows from call_date to maturity
            continuation_cf_dates, continuation_cf_amounts = (
                self.callable_bond.cashflow_since_date(call_date)
            )

            # Extract rate path from call_date onwards
            call_time_index = max(
                0, min(int(call_date - valuation_date), len(rate_path) - 1)
            )
            rate_path_from_call = rate_path[call_time_index:]

            # Calculate continuation value using Monte Carlo discounting
            continuation_value = self.pv_of_future_cash_flows(
                continuation_cf_dates,
                continuation_cf_amounts,
                call_date,
                rate_path_from_call,
                self.rate_simulator._dt,
                discount_mode="discrete",
            )

            # Call value (par + premium)
            call_price = self.callable_bond.face_value + self.call_premium

            # Issuer calls if it's beneficial (continuation value > call price)
            if continuation_value > call_price * 1.02:  # 2% buffer
                # Bond is called. Calculate PV of (coupons up to call_date + call_price at call_date)

                effective_cf_dates = []
                effective_cf_amounts = []
                coupon_on_call_date = 0.0

                # Get all bond cash flows from valuation_date onwards
                cf_dates, cf_amounts = self.callable_bond.cashflow_since_date(
                    valuation_date
                )

                # Process each cash flow from the bond
                for cf_date, cf_amount in zip(cf_dates, cf_amounts):
                    if cf_date < call_date:
                        # This is a coupon payment before call date
                        effective_cf_dates.append(cf_date)
                        effective_cf_amounts.append(cf_amount)
                    elif cf_date == call_date:
                        # If call date coincides with a coupon date, include the coupon.
                        # The final payment at maturity includes both coupon and principal.
                        # We need to extract just the coupon part if this is the maturity date.
                        if cf_date == self.callable_bond.maturity_date:
                            # This is the final payment which includes both coupon and principal
                            # Extract just the coupon portion
                            coupon_on_call_date = (
                                self.callable_bond.bond._coupon
                                / self.callable_bond.bond._frequency
                                * self.callable_bond.face_value
                            )
                        else:
                            # This is just a regular coupon payment
                            coupon_on_call_date = cf_amount
                    # For dates after call_date, we ignore them as the bond is called

                # Add the call payment (call price + any coupon due on call date)
                effective_cf_dates.append(call_date)
                effective_cf_amounts.append(call_price + coupon_on_call_date)

                called_bond_value = self.pv_of_future_cash_flows(
                    effective_cf_dates,
                    effective_cf_amounts,
                    valuation_date,
                    rate_path,
                    self.rate_simulator._dt,
                )
                return (
                    called_bond_value,
                    True,
                    {
                        "call_date": call_date,
                        "call_value": call_price + coupon_on_call_date,
                    },
                )

        # Bond was not called - value is PV of all remaining straight bond cash flows
        # (coupons + principal at maturity) from valuation_date onwards.

        # Use the already filtered cash flows (bond_cf_dates, bond_cf_amounts)
        not_called_bond_value = self.pv_of_future_cash_flows(
            bond_cf_dates,
            bond_cf_amounts,
            valuation_date,
            rate_path,
            self.rate_simulator._dt,
        )
        return not_called_bond_value, False, {}


def solve_callable_bond_coupon(
    rate_simulator: InterestRateSimulator,
    straight_bond: Bond,
    straight_bond_ytm: float,
    call_protection_years: int = 1,
    tolerance: float = 1e-4,
) -> tuple[float, dict]:
    """Solve for the coupon rate that makes callable bond value equal to 100.

    Args:
        rate_simulator: InterestRateSimulator instance for Monte Carlo simulation
        straight_bond: Straight bond object to construct callable bond from
        straight_bond_ytm: Fair ytm for straight bond
        call_protection_years: Years of call protection
        tolerance: Convergence tolerance

    Returns:
        Tuple of (optimal_coupon_rate, solution_info)
    """
    target_value = 100.0

    def objective_function(coupon_rate: float) -> float:
        """Objective function: difference between callable bond value and target."""

        # Create callable bond with different coupon rates
        callable_bond = CallableBond(
            issue_date=straight_bond.issue_dt,
            maturity_date=straight_bond.maturity_dt,
            coupon_rate=coupon_rate,
            call_protection_years=call_protection_years,
            freq_type=straight_bond.freq_type,
            day_count_type=straight_bond.dc_type,
        )
        # Calculate equivalent initial short rate for the different coupon rates and same straight bond ytm
        # Given the same straight bond ytm, the equivalent rates seem to be the same for all coupon rates.
        equivalent_rate, _ = (
            CallableBondValuer.calculate_equivalent_initial_short_rate(
                straight_bond=callable_bond.bond,
                dt=rate_simulator._dt,
                ytm=straight_bond_ytm,
                valuation_date=callable_bond.bond.issue_dt,
                discount_mode="discrete",
                tolerance=1e-6,
            )
        )

        # Set the equivalent rate in the rate simulator
        rate_simulator.r0 = equivalent_rate

        # Create valuer
        valuer = CallableBondValuer(callable_bond, rate_simulator)

        # Value the bond
        callable_value, _, _ = valuer.value_bond(
            valuation_date=straight_bond.issue_dt,
            straight_bond_ytm=straight_bond_ytm,
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
        print(
            f"Adjusted: f({lower_bound:.1%}) = {f_lower:.4f}, f({upper_bound:.1%}) = {f_upper:.4f}"
        )

    # Solve optimal coupon using Brent's method
    try:
        optimal_coupon = optimize.brentq(
            objective_function,
            a=lower_bound,
            b=upper_bound,
            xtol=tolerance,
            maxiter=100,
        )

        # Get final solution info
        final_callable_bond = CallableBond(
            issue_date=straight_bond.issue_dt,
            maturity_date=straight_bond.maturity_dt,
            coupon_rate=optimal_coupon,
            call_protection_years=call_protection_years,
            freq_type=straight_bond.freq_type,
            day_count_type=straight_bond.dc_type,
        )

        valuer = CallableBondValuer(final_callable_bond, rate_simulator)
        final_value, straight_value, stats = valuer.value_bond(
            valuation_date=straight_bond.issue_dt,
            straight_bond_ytm=straight_bond.cpn,
        )

        solution_info = {
            "optimal_coupon": optimal_coupon,
            "callable_bond_value": final_value,
            "straight_bond_value": straight_value,
            "target_value": target_value,
            "error": abs(final_value - target_value),
            "option_value": stats["option_value"],
            "call_probability": stats["call_probability"],
        }

        return optimal_coupon, solution_info

    except ValueError as e:
        raise ValueError(f"Could not find solution: {e}")


def analyze_callable_bond_dynamics(
    rate_simulator: InterestRateSimulator,
    straight_bond: Bond,
    straight_bond_ytm: float,
    call_protection_years: int = 1,
    num_simulations: int | None = None,
    coupon_range: list[float] = None,
) -> None:
    """Analyze the relationship between coupon rates and callable bond values."""

    if coupon_range is None:
        coupon_range = [
            0.02,
            0.03,
            0.04,
            0.045,
            0.046,
            0.047,
            0.048,
            0.049,
            0.050,
            0.051,
            0.052,
            0.053,
            0.054,
            0.055,
            0.06,
            0.08,
        ]

    print("=== Callable Bond Dynamics Analysis ===")
    print(f"Straight bond YTM: {straight_bond_ytm:.2%}")
    print(f"Coupon Rate | Straight Bond | Callable Bond | Option Value | Call Prob")
    print("-" * 70)

    for coupon in coupon_range:
        # Create callable bond
        callable_bond = CallableBond(
            issue_date=straight_bond.issue_dt,
            maturity_date=straight_bond.maturity_dt,
            coupon_rate=coupon,
            call_protection_years=call_protection_years,
            freq_type=straight_bond.freq_type,
            day_count_type=straight_bond.dc_type,
        )

        # Calculate equivalent initial short rate for different coupon rates
        equivalent_rate, solution_info = (
            CallableBondValuer.calculate_equivalent_initial_short_rate(
                straight_bond=callable_bond.bond,
                dt=rate_simulator._dt,
                ytm=straight_bond_ytm,
                valuation_date=callable_bond.bond.issue_dt,
                discount_mode="discrete",
                tolerance=1e-6,
            )
        )

        # Set the equivalent rate in the rate simulator
        rate_simulator.r0 = equivalent_rate

        # Create valuer
        valuer = CallableBondValuer(callable_bond, rate_simulator)

        if num_simulations is None:
            num_simulations = rate_simulator.num_paths

        # Value the bond
        callable_value, straight_value, stats = valuer.value_bond(
            valuation_date=straight_bond.issue_dt,
            straight_bond_ytm=straight_bond_ytm,
            num_simulations=num_simulations,
        )

        print(
            f"{coupon:8.2%} | {straight_value:11.4f} | {callable_value:11.4f} | "
            f"{stats['option_value']:10.4f} | {stats['call_probability']:.1%}"
        )


def main():
    # Test the implementation with the specific scenario
    print("=== Callable Bond Valuer Implementation Test ===\n")
    issue_date = Date(27, 5, 2025)
    maturity_date = Date(27, 5, 2030)  # 5 years
    days = int(maturity_date - issue_date) + 2
    days_of_year = 365

    straight_bond_coupon = 0.046  # 4.60%
    straight_bond_ytm = straight_bond_coupon  # 4.60%

    # Create a straight bond for testing
    straight_bond = Bond(
        issue_dt=issue_date,
        maturity_dt=maturity_date,
        coupon=straight_bond_coupon,
        freq_type=FrequencyTypes.ANNUAL,
        dc_type=DayCountTypes.THIRTY_E_360,
    )

    print("Problem Parameters:")
    print(f"Straight bond coupon: {straight_bond_coupon:.2%}")
    print(f"Target bond value: 100")
    print(f"Call protection: 1 year")
    print()

    # Create rate simulator with equivalent rate
    rate_simulator = InterestRateSimulator(
        r0=0,  # It will be set to the equivalent rate in solve_callable_bond_coupon() and analyze_callable_bond_dynamics()
        mu=0,
        sigma=0.0095,
        days=days,
        days_of_year=days_of_year,
        num_paths=5000,
        model="abm",
        kappa=0.1,
    )

    try:
        # Analyze callable bond dynamics first
        analyze_callable_bond_dynamics(
            rate_simulator=rate_simulator,
            straight_bond=straight_bond,
            straight_bond_ytm=straight_bond_ytm,
            call_protection_years=1,
        )

        # Solve for callable bond coupon using Monte Carlo optimization
        print("\n=== Solving for Callable Bond Coupon ===")

        optimal_coupon, solution_info = solve_callable_bond_coupon(
            rate_simulator=rate_simulator,
            straight_bond=straight_bond,
            straight_bond_ytm=straight_bond_ytm,
            call_protection_years=1,
        )

        print("\nSolution Results:")
        print(f"Optimal callable bond coupon: {optimal_coupon:.4%}")
        print(f"Callable bond value: {solution_info['callable_bond_value']:.4f}")
        print(f"Straight bond value: {solution_info['straight_bond_value']:.4f}")
        print(f"Target value: {solution_info['target_value']:.4f}")
        print(f"Pricing error: {solution_info['error']:.6f}")
        print(f"Option value: {solution_info['option_value']:.4f}")
        print(f"Call probability: {solution_info['call_probability']:.2%}")

        print(
            f"\nConclusion: To achieve a bond value of 100, "
            f"the callable bond should have a coupon rate of {optimal_coupon:.4%}."
        )

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

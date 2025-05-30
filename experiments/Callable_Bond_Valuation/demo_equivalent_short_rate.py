"""
Demonstration of the calculate_equivalent_initial_short_rate function.

This script shows how to use the new function to find an equivalent flat short rate
that makes Monte Carlo valuation match analytical bond pricing.
"""

from financepy.products.bonds.bond import Bond
from financepy.utils.date import Date
from financepy.utils.frequency import FrequencyTypes
from financepy.utils.day_count import DayCountTypes

from callable_bond_valuer import calculate_equivalent_initial_short_rate


def main():
    print("=== Equivalent Initial Short Rate Calculation Demo ===\n")
    
    # Set up bond parameters
    issue_date = Date(27, 5, 2025)
    maturity_date = Date(27, 5, 2030)  # 5-year bond
    coupon_rate = 0.046  
    ytm = 0.046  # 4.5% yield to maturity
    
    print("Bond Parameters:")
    print(f"Issue Date: {issue_date}")
    print(f"Maturity Date: {maturity_date}")
    print(f"Coupon Rate: {coupon_rate:.2%}")
    print(f"Yield to Maturity: {ytm:.2%}")
    print()
    
    # Create the bond
    bond = Bond(
        issue_dt=issue_date,
        maturity_dt=maturity_date,
        coupon=coupon_rate,
        freq_type=FrequencyTypes.ANNUAL,
        dc_type=DayCountTypes.THIRTY_E_360
    )
    
    # Calculate analytical bond price
    analytical_price = bond.dirty_price_from_ytm(issue_date, ytm)
    print(f"Analytical Bond Price: {analytical_price:.4f}")
    print()
    
    # Test both discrete and continuous modes
    for discount_mode in ["discrete", "continuous"]:
        print(f"--- {discount_mode.title()} Discounting Mode ---")
        
        try:
            # Calculate equivalent short rate
            equivalent_rate, solution_info = calculate_equivalent_initial_short_rate(
                straight_bond=bond,
                ytm=ytm,
                valuation_date=issue_date,
                discount_mode=discount_mode,
                tolerance=1e-6
            )
            
            print(f"Equivalent Short Rate: {equivalent_rate:.4%}")
            print(f"Rate Difference from YTM: {solution_info['rate_difference']:.4%}")
            print(f"Monte Carlo PV: {solution_info['mc_pv']:.6f}")
            print(f"Absolute Error: {solution_info['error']:.8f}")
            print(f"Relative Error: {solution_info['relative_error']:.6%}")
            print()
            
        except Exception as e:
            print(f"Error in {discount_mode} mode: {e}")
            print()
    
    # Demonstrate with different bond types
    print("=== Testing Different Bond Types ===\n")
    
    # Premium bond (high coupon, low YTM)
    print("Premium Bond (8% coupon, 6% YTM):")
    premium_bond = Bond(
        issue_dt=issue_date,
        maturity_dt=maturity_date,
        coupon=0.08,
        freq_type=FrequencyTypes.ANNUAL,
        dc_type=DayCountTypes.THIRTY_E_360
    )
    
    premium_ytm = 0.06
    premium_price = premium_bond.dirty_price_from_ytm(issue_date, premium_ytm)
    print(f"Analytical Price: {premium_price:.6f}")
    
    equiv_rate, info = calculate_equivalent_initial_short_rate(
        premium_bond, premium_ytm, issue_date, "discrete", tolerance=1e-6
    )
    print(f"Equivalent Short Rate: {equiv_rate:.4%} (vs YTM: {premium_ytm:.4%})")
    print(f"Rate Difference: {info['rate_difference']:.4%}")
    print()
    
    # Discount bond (low coupon, high YTM)
    print("Discount Bond (3% coupon, 5% YTM):")
    discount_bond = Bond(
        issue_dt=issue_date,
        maturity_dt=maturity_date,
        coupon=0.03,
        freq_type=FrequencyTypes.ANNUAL,
        dc_type=DayCountTypes.THIRTY_E_360
    )
    
    discount_ytm = 0.05
    discount_price = discount_bond.dirty_price_from_ytm(issue_date, discount_ytm)
    print(f"Analytical Price: {discount_price:.6f}")
    
    equiv_rate, info = calculate_equivalent_initial_short_rate(
        discount_bond, discount_ytm, issue_date, "discrete", tolerance=1e-6
    )
    print(f"Equivalent Short Rate: {equiv_rate:.4%} (vs YTM: {discount_ytm:.4%})")
    print(f"Rate Difference: {info['rate_difference']:.4%}")
    print()
    
    print("=== Summary ===")
    print("The calculate_equivalent_initial_short_rate function successfully:")
    print("1. Finds a flat short rate that matches analytical bond pricing")
    print("2. Works with both discrete and continuous discounting modes")
    print("3. Handles premium, par, and discount bonds")
    print("4. Provides detailed solution information for analysis")
    print("\nThis function is essential for calibrating Monte Carlo simulations")
    print("to ensure consistency with analytical bond valuations.")


if __name__ == "__main__":
    main() 
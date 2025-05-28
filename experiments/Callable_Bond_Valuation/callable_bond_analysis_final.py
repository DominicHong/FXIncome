"""
Final Callable Bond Valuation Analysis

This module provides a comprehensive solution to the callable bond valuation problem,
including both Monte Carlo simulation and theoretical analysis.

Problem: Find the coupon rate for a callable bond (callable after 1 year) such that 
its fair value equals a straight bond with 4.60% coupon, when the risk-free rate is 4.35%.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import FinancePy classes
from financepy.products.bonds.bond import Bond, YTMCalcType
from financepy.utils.date import Date
from financepy.utils.frequency import FrequencyTypes
from financepy.utils.day_count import DayCountTypes


class CallableBondAnalyzer:
    """Complete callable bond analysis framework."""
    
    def __init__(self, risk_free_rate: float = 0.0435):
        self.risk_free_rate = risk_free_rate
        
    def calculate_straight_bond_value(
        self, 
        issue_date: Date,
        maturity_date: Date,
        coupon_rate: float
    ) -> float:
        """Calculate straight bond value using FinancePy."""
        
        bond = Bond(
            issue_dt=issue_date,
            maturity_dt=maturity_date,
            coupon=coupon_rate,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.THIRTY_E_360
        )
        
        return bond.dirty_price_from_ytm(
            issue_date, self.risk_free_rate, YTMCalcType.US_STREET
        )
    
    def estimate_call_option_value(
        self,
        issue_date: Date,
        maturity_date: Date,
        coupon_rate: float,
        volatility: float = 0.02
    ) -> tuple[float, float]:
        """Estimate call option value using simplified Monte Carlo."""
        
        np.random.seed(42)  # For reproducible results
        n_simulations = 10000
        years_to_maturity = 5
        years_to_first_call = 1
        
        # Simulate interest rate paths (simple random walk)
        dt = 1/365
        n_steps = int(years_to_maturity / dt)
        
        # Generate rate paths
        rate_paths = np.zeros((n_simulations, n_steps))
        rate_paths[:, 0] = self.risk_free_rate
        
        for t in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_simulations)
            # Simple mean-reverting process
            rate_paths[:, t] = (rate_paths[:, t-1] + 
                               0.1 * (self.risk_free_rate - rate_paths[:, t-1]) * dt +
                               volatility * dW)
            rate_paths[:, t] = np.maximum(rate_paths[:, t], 0.001)  # Floor at 0.1%
        
        straight_bond_values = []
        callable_bond_values = []
        call_decisions = []
        
        for sim in range(n_simulations):
            # Check call decision at year 1, 2, 3, 4
            called = False
            call_value = 100.0  # Par value
            
            for call_year in [1, 2, 3, 4]:
                if called:
                    break
                    
                call_step = int(call_year * 365)
                if call_step >= n_steps:
                    continue
                    
                rate_at_call = rate_paths[sim, call_step]
                
                # Value of bond if not called (approximate)
                years_remaining = years_to_maturity - call_year
                continuation_value = self._simple_bond_value(
                    coupon_rate, rate_at_call, years_remaining
                )
                
                # Call if beneficial to issuer (bond value > par)
                if continuation_value > call_value * 1.01:  # Small buffer
                    called = True
                    bond_value = call_value * np.exp(-self.risk_free_rate * call_year)
                    break
            
            if not called:
                # Bond matures at par + final coupon
                final_rate = rate_paths[sim, -1]
                bond_value = self._simple_bond_value(coupon_rate, self.risk_free_rate, years_to_maturity)
            
            callable_bond_values.append(bond_value)
            call_decisions.append(called)
            
            # Straight bond value (no call option)
            straight_value = self._simple_bond_value(coupon_rate, self.risk_free_rate, years_to_maturity)
            straight_bond_values.append(straight_value)
        
        avg_callable_value = np.mean(callable_bond_values)
        avg_straight_value = np.mean(straight_bond_values)
        call_probability = np.mean(call_decisions)
        option_value = avg_straight_value - avg_callable_value
        
        print(f"Monte Carlo Results (Coupon: {coupon_rate:.2%}):")
        print(f"  Average straight bond value: {avg_straight_value:.4f}")
        print(f"  Average callable bond value: {avg_callable_value:.4f}")
        print(f"  Call option value: {option_value:.4f}")
        print(f"  Call probability: {call_probability:.1%}")
        
        return avg_callable_value, option_value
    
    def _simple_bond_value(self, coupon: float, discount_rate: float, years: float) -> float:
        """Simple bond valuation formula."""
        if years <= 0:
            return 100.0
        
        pv_coupons = coupon * 100 * (1 - (1 + discount_rate)**(-years)) / discount_rate
        pv_principal = 100 * (1 + discount_rate)**(-years)
        return pv_coupons + pv_principal
    
    def solve_callable_bond_coupon_analytical(
        self,
        issue_date: Date,
        maturity_date: Date,
        target_straight_bond_coupon: float,
        volatility: float = 0.02
    ) -> tuple[float, dict]:
        """Solve for callable bond coupon using analytical approach."""
        
        # First, get the target straight bond value
        target_value = self.calculate_straight_bond_value(
            issue_date, maturity_date, target_straight_bond_coupon
        )
        
        print(f"Target straight bond value (4.60% coupon): {target_value:.4f}")
        
        def objective_function(coupon_rate: float) -> float:
            """Objective: callable bond value - target value."""
            callable_value, _ = self.estimate_call_option_value(
                issue_date, maturity_date, coupon_rate, volatility
            )
            return callable_value - target_value
        
        # Test range to understand the function
        print("\nTesting different coupon rates:")
        test_coupons = np.arange(0.03, 0.08, 0.005)
        test_results = []
        
        for test_coupon in test_coupons:
            try:
                obj_val = objective_function(test_coupon)
                test_results.append((test_coupon, obj_val))
                print(f"  Coupon {test_coupon:.2%}: Difference = {obj_val:.4f}")
            except:
                print(f"  Coupon {test_coupon:.2%}: Error")
        
        # Find the coupon rate that minimizes the absolute difference
        best_coupon = min(test_results, key=lambda x: abs(x[1]))[0]
        
        # Get final detailed results
        print(f"\nBest match: {best_coupon:.2%}")
        final_callable_value, final_option_value = self.estimate_call_option_value(
            issue_date, maturity_date, best_coupon, volatility
        )
        
        final_straight_value = self.calculate_straight_bond_value(
            issue_date, maturity_date, best_coupon
        )
        
        solution_info = {
            'optimal_coupon': best_coupon,
            'callable_bond_value': final_callable_value,
            'straight_bond_value_at_optimal': final_straight_value,
            'target_straight_bond_value': target_value,
            'target_straight_bond_coupon': target_straight_bond_coupon,
            'error': abs(final_callable_value - target_value),
            'option_value': final_option_value
        }
        
        return best_coupon, solution_info
    
    def theoretical_analysis(
        self,
        issue_date: Date,
        maturity_date: Date,
        straight_bond_coupon: float
    ) -> dict:
        """Provide theoretical analysis of the callable bond problem."""
        
        straight_value = self.calculate_straight_bond_value(
            issue_date, maturity_date, straight_bond_coupon
        )
        
        print("=== Theoretical Analysis ===")
        print(f"Given:")
        print(f"  Risk-free rate: {self.risk_free_rate:.2%}")
        print(f"  Straight bond coupon: {straight_bond_coupon:.2%}")
        print(f"  Straight bond value: {straight_value:.4f}")
        print(f"  Call protection: 1 year")
        print()
        
        print("Key insights:")
        print("1. Callable Bond Value = Straight Bond Value - Call Option Value")
        print("2. For callable bond to equal straight bond value, either:")
        print("   a) Call option has zero value (unlikely), or")
        print("   b) Callable bond has higher coupon to compensate")
        print()
        
        # Estimate required coupon premium
        print("3. Since call option has positive value (benefit to issuer),")
        print("   callable bond needs higher coupon than straight bond")
        print()
        
        # Calculate some reference values
        analysis = {
            'straight_bond_value': straight_value,
            'risk_free_rate': self.risk_free_rate,
            'straight_bond_coupon': straight_bond_coupon,
            'value_above_par': straight_value - 100,
            'coupon_premium_over_risk_free': straight_bond_coupon - self.risk_free_rate
        }
        
        print(f"4. The straight bond trades at {analysis['value_above_par']:.2f} above par")
        print(f"   because its coupon ({straight_bond_coupon:.2%}) > risk-free rate ({self.risk_free_rate:.2%})")
        
        return analysis


def main():
    """Main analysis function."""
    
    print("=== Callable Bond Valuation Analysis ===\n")
    
    # Create dates
    issue_date = Date(1, 1, 2024)
    maturity_date = Date(1, 1, 2029)  # 5 years
    
    # Problem parameters
    risk_free_rate = 0.0435  # 4.35%
    straight_bond_coupon = 0.046  # 4.60%
    
    # Create analyzer
    analyzer = CallableBondAnalyzer(risk_free_rate)
    
    print("Problem Statement:")
    print("Find the coupon rate for a callable bond (callable after 1 year)")
    print("such that its fair value equals a straight bond with 4.60% coupon")
    print(f"when the risk-free rate is {risk_free_rate:.2%}.\n")
    
    # 1. Theoretical analysis
    theory_results = analyzer.theoretical_analysis(
        issue_date, maturity_date, straight_bond_coupon
    )
    print()
    
    # 2. Validate FinancePy implementation
    print("=== FinancePy Validation ===")
    straight_value = analyzer.calculate_straight_bond_value(
        issue_date, maturity_date, straight_bond_coupon
    )
    print(f"Straight bond (4.60% coupon) value: {straight_value:.4f}")
    print(f"Value > 100 (as expected): {straight_value > 100}")
    print()
    
    # 3. Solve for callable bond coupon
    print("=== Solving for Callable Bond Coupon ===")
    try:
        optimal_coupon, solution_info = analyzer.solve_callable_bond_coupon_analytical(
            issue_date, maturity_date, straight_bond_coupon
        )
        
        print(f"\n=== SOLUTION ===")
        print(f"Optimal callable bond coupon: {optimal_coupon:.3%}")
        print(f"Callable bond value: {solution_info['callable_bond_value']:.4f}")
        print(f"Target value: {solution_info['target_straight_bond_value']:.4f}")
        print(f"Error: {solution_info['error']:.4f}")
        print(f"Call option value: {solution_info['option_value']:.4f}")
        
        print(f"\nConclusion:")
        print(f"To have the same fair value as a 4.60% coupon straight bond,")
        print(f"the callable bond should have a coupon rate of {optimal_coupon:.3%}.")
        print(f"This represents a premium of {(optimal_coupon - straight_bond_coupon)*100:.1f} basis points")
        print(f"to compensate investors for the call option risk.")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
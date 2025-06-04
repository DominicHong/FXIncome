import pytest
import numpy as np
from experiments.Callable_Bond_Valuation.callable_bond_valuer import (
    CallableBond,
    CallableBondValuer,
    InterestRateSimulator,
)

from financepy.utils.date import Date
from financepy.utils.frequency import FrequencyTypes
from financepy.utils.day_count import DayCountTypes
from financepy.products.bonds.bond import YTMCalcType, Bond


class TestCallableBondValuer:
    """Test suite for the CallableBondValuer implementation."""

    @pytest.fixture
    def test_dates(self):
        """Fixture providing test dates."""
        return {
            "issue_date": Date(27, 5, 2025),
            "maturity_date": Date(27, 5, 2030),
        }
    @pytest.fixture
    def daily_dt(self):
        return 1/365
    
    @pytest.fixture
    def callable_bond(self, test_dates):
        """Fixture providing a test callable bond."""
        return CallableBond(
            issue_date=test_dates["issue_date"],
            maturity_date=test_dates["maturity_date"],
            coupon_rate=0.05,  # 5%
            call_protection_years=1,
            freq_type=FrequencyTypes.ANNUAL,
            day_count_type=DayCountTypes.THIRTY_E_360,
        )

    @pytest.fixture
    def rate_simulator(self):
        """Fixture providing a test rate simulator."""
        return InterestRateSimulator(
            r0=0.0435, sigma=0.02, days=365 * 5, num_paths=1, model="gbm"
        )

    @pytest.fixture
    def valuer(self, callable_bond, rate_simulator):
        """Fixture providing a test callable bond valuer."""
        return CallableBondValuer(callable_bond, rate_simulator)

    def test_pv_calculation_discrete_mode(self, valuer, test_dates, daily_dt):
        """Test present value calculation in discrete mode."""
        discount_rate = 0.0435
        flat_rate_path = np.full(365 * 5, discount_rate)

        # Test PV of a simple 5% coupon at year 1
        test_cf_dates = [test_dates["issue_date"].add_years(1)]
        test_cf_amounts = [5.0]  # 5% of 100 face value

        calculated_pv = valuer.pv_of_future_cash_flows(
            test_cf_dates,
            test_cf_amounts,
            test_dates["issue_date"],
            flat_rate_path,
            daily_dt,
            discount_mode="discrete",
        )
        # Use daily compounding to match the implementation's behavior
        # Implementation uses (1 + rate/365)^365 for daily compounding
        expected_pv = 5.0 / ((1 + discount_rate/365) ** 365)  # Daily discrete compounding

        assert (
            abs(calculated_pv - expected_pv) < 1e-4
        ), f"Discrete PV calculation failed. Expected: {expected_pv:.6f}, Got: {calculated_pv:.6f}"

    def test_pv_calculation_continuous_mode(self, valuer, test_dates, daily_dt):
        """Test present value calculation in continuous mode."""
        discount_rate = 0.0435
        flat_rate_path = np.full(365 * 5, discount_rate)

        # Test PV of a simple 5% coupon at year 1
        test_cf_dates = [test_dates["issue_date"].add_years(1)]
        test_cf_amounts = [5.0]  # 5% of 100 face value

        calculated_pv = valuer.pv_of_future_cash_flows(
            test_cf_dates,
            test_cf_amounts,
            test_dates["issue_date"],
            flat_rate_path,
            daily_dt,
            discount_mode="continuous",
        )
        expected_pv = 5.0 * np.exp(-discount_rate * 1.0)  # Continuous compounding

        assert (
            abs(calculated_pv - expected_pv) < 1e-4
        ), f"Continuous PV calculation failed. Expected: {expected_pv:.6f}, Got: {calculated_pv:.6f}"

    def test_straight_bond_valuation_vs_financepy(
        self, valuer, callable_bond, test_dates, daily_dt
    ):
        """Test that our straight bond valuation matches FinancePy within reasonable tolerance."""
        discount_rate = 0.0435
        flat_rate_path = np.full(365 * 5, discount_rate)

        # Get bond cash flows and scale them
        bond_cf_dates = callable_bond.bond.cpn_dts
        bond_cf_amounts = [
            cf * callable_bond.face_value
            for cf in callable_bond.bond.flow_amounts
        ]

        # Add principal to the final payment (FinancePy quirk)
        if (
            len(bond_cf_dates) > 0
            and bond_cf_dates[-1] == callable_bond.maturity_date
            and len(bond_cf_amounts) > 0
        ):
            bond_cf_amounts[-1] += callable_bond.face_value

        # Discrete mode for comparison
        discrete_pv = valuer.pv_of_future_cash_flows(
            bond_cf_dates,
            bond_cf_amounts,
            test_dates["issue_date"],
            flat_rate_path,
            daily_dt,
            discount_mode="discrete",
        )
        # Continuous mode for comparison
        continuous_pv = valuer.pv_of_future_cash_flows(
            bond_cf_dates,
            bond_cf_amounts,
            test_dates["issue_date"],
            flat_rate_path,
            daily_dt,
            discount_mode="continuous",
        )

        financepy_pv = callable_bond.dirty_price_from_ytm(
            test_dates["issue_date"], discount_rate, YTMCalcType.US_STREET
        )

        print(f"Discrete PV: {discrete_pv:.4f}, Continuous PV: {continuous_pv:.4f}, FinancePy PV: {financepy_pv:.4f}")
        
        # Allow for some difference due to different day count conventions and methods
        tolerance = 0.5  # 0.5/100 tolerance
        assert (
            abs(discrete_pv - financepy_pv) < tolerance
        ), f"Bond valuation differs too much from FinancePy. Our: {discrete_pv:.4f}, FinancePy: {financepy_pv:.4f}, Diff: {abs(discrete_pv - financepy_pv):.4f}"

        assert (
            abs(continuous_pv - financepy_pv) < tolerance
        ), f"Bond valuation differs too much from FinancePy. Our: {continuous_pv:.4f}, FinancePy: {financepy_pv:.4f}, Diff: {abs(continuous_pv - financepy_pv):.4f}"
        
    def test_pv_calculation_with_cash_flow_on_valuation_date(
        self, valuer, test_dates, daily_dt
    ):
        """Test that cash flows on the valuation date should not be added to the PV."""
        flat_rate_path = np.full(365 * 5, 0.05)

        # Cash flow on valuation date is only received by the bond holder on the previous date
        test_cf_dates = [test_dates["issue_date"]]  # Same as valuation date
        test_cf_amounts = [10.0]

        calculated_pv = valuer.pv_of_future_cash_flows(
            test_cf_dates, test_cf_amounts, test_dates["issue_date"], flat_rate_path, daily_dt
        )

        assert (
            abs(calculated_pv - 0.0) < 1e-6
        ), f"Cash flow on valuation date should not be added to the PV. Expected: 0.0, Got: {calculated_pv:.6f}"

    def test_pv_calculation_ignores_past_cash_flows(self, valuer, test_dates, daily_dt):
        """Test that past cash flows are ignored."""
        flat_rate_path = np.full(365 * 5, 0.05)

        # Cash flow in the past should be ignored
        past_date = test_dates["issue_date"].add_days(-30)  # 30 days before valuation
        test_cf_dates = [past_date]
        test_cf_amounts = [100.0]

        calculated_pv = valuer.pv_of_future_cash_flows(
            test_cf_dates, test_cf_amounts, test_dates["issue_date"], flat_rate_path, daily_dt
        )

        assert (
            abs(calculated_pv) < 1e-6
        ), f"Past cash flows should be ignored. Expected: 0.0, Got: {calculated_pv:.6f}"

    def test_invalid_discount_mode_raises_error(self, valuer, test_dates, daily_dt):
        """Test that invalid discount mode raises ValueError."""
        flat_rate_path = np.full(365 * 5, 0.05)
        test_cf_dates = [test_dates["issue_date"].add_years(1)]
        test_cf_amounts = [5.0]

        with pytest.raises(ValueError, match="Invalid discount mode"):
            valuer.pv_of_future_cash_flows(
                test_cf_dates,
                test_cf_amounts,
                test_dates["issue_date"],
                flat_rate_path,
                daily_dt,
                discount_mode="invalid_mode",
            )

    def test_calculate_equivalent_initial_short_rate_discrete(self, test_dates, daily_dt):
        """Test equivalent initial short rate calculation in discrete mode."""
        # Create a straight bond
        straight_bond = Bond(
            issue_dt=test_dates["issue_date"],
            maturity_dt=test_dates["maturity_date"],
            coupon=0.05,  # 5% coupon
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.THIRTY_E_360
        )
        
        ytm = 0.045  # 4.5% YTM
        
        # Calculate equivalent short rate
        equivalent_rate, solution_info = CallableBondValuer.calculate_equivalent_initial_short_rate(
            straight_bond=straight_bond,
            dt=daily_dt,
            ytm=ytm,
            valuation_date=test_dates["issue_date"],
            discount_mode="discrete",
            tolerance=1e-6
        )
        
        # Verify the solution
        assert solution_info['error'] < 1e-4, f"Solution error too large: {solution_info['error']}"
        assert solution_info['relative_error'] < 1e-6, f"Relative error too large: {solution_info['relative_error']}"
        assert solution_info['ytm'] == ytm, "YTM should match input"
        assert solution_info['discount_mode'] == "discrete", "Discount mode should match"
        assert solution_info['num_cash_flows'] > 0, "Should have cash flows"
        
        # The equivalent rate should be close to but not necessarily equal to YTM
        assert abs(equivalent_rate - ytm) < 0.02, f"Equivalent rate {equivalent_rate:.4%} too far from YTM {ytm:.4%}"

    def test_calculate_equivalent_initial_short_rate_continuous(self, test_dates, daily_dt):
        """Test equivalent initial short rate calculation in continuous mode."""
        # Create a straight bond
        straight_bond = Bond(
            issue_dt=test_dates["issue_date"],
            maturity_dt=test_dates["maturity_date"],
            coupon=0.04,  # 4% coupon
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.THIRTY_E_360
        )
        
        ytm = 0.04  # 4% YTM (par bond)
        
        # Calculate equivalent short rate
        equivalent_rate, solution_info = CallableBondValuer.calculate_equivalent_initial_short_rate(
            straight_bond=straight_bond,
            dt=daily_dt,
            ytm=ytm,
            valuation_date=test_dates["issue_date"],
            discount_mode="continuous",
            tolerance=1e-6
        )
        
        # Verify the solution
        assert solution_info['error'] < 1e-4, f"Solution error too large: {solution_info['error']}"
        assert solution_info['relative_error'] < 1e-6, f"Relative error too large: {solution_info['relative_error']}"
        assert solution_info['ytm'] == ytm, "YTM should match input"
        assert solution_info['discount_mode'] == "continuous", "Discount mode should match"
        
        # For a par bond, the equivalent rate should be very close to YTM
        assert abs(equivalent_rate - ytm) < 0.005, f"Equivalent rate {equivalent_rate:.4%} should be close to YTM {ytm:.4%} for par bond"

    def test_calculate_equivalent_initial_short_rate_premium_bond(self, test_dates, daily_dt):
        """Test equivalent initial short rate calculation for a premium bond."""
        # Create a premium bond (high coupon, low YTM)
        straight_bond = Bond(
            issue_dt=test_dates["issue_date"],
            maturity_dt=test_dates["maturity_date"],
            coupon=0.08,  # 8% coupon
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.THIRTY_E_360
        )
        
        ytm = 0.06  # 6% YTM (premium bond)
        
        # Calculate equivalent short rate
        equivalent_rate, solution_info = CallableBondValuer.calculate_equivalent_initial_short_rate(
            straight_bond=straight_bond,
            dt=daily_dt,
            ytm=ytm,
            valuation_date=test_dates["issue_date"],
            discount_mode="discrete",
            tolerance=1e-6
        )
        
        # Verify the solution
        assert solution_info['error'] < 1e-4, f"Solution error too large: {solution_info['error']}"
        assert solution_info['relative_error'] < 1e-6, f"Relative error too large: {solution_info['relative_error']}"
        
        # Bond should be trading at premium
        assert solution_info['analytical_pv'] > 100, "Bond should be at premium"

    def test_calculate_equivalent_initial_short_rate_discount_bond(self, test_dates, daily_dt):
        """Test equivalent initial short rate calculation for a discount bond."""
        # Create a discount bond (low coupon, high YTM)
        straight_bond = Bond(
            issue_dt=test_dates["issue_date"],
            maturity_dt=test_dates["maturity_date"],
            coupon=0.03,  # 3% coupon
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.THIRTY_E_360
        )
        
        ytm = 0.05  # 5% YTM (discount bond)
        
        # Calculate equivalent short rate
        equivalent_rate, solution_info = CallableBondValuer.calculate_equivalent_initial_short_rate(
            straight_bond=straight_bond,
            dt=daily_dt,
            ytm=ytm,
            valuation_date=test_dates["issue_date"],
            discount_mode="continuous",
            tolerance=1e-6
        )
        
        # Verify the solution
        assert solution_info['error'] < 1e-4, f"Solution error too large: {solution_info['error']}"
        assert solution_info['relative_error'] < 1e-6, f"Relative error too large: {solution_info['relative_error']}"
        
        # Bond should be trading at discount
        assert solution_info['analytical_pv'] < 100, "Bond should be at discount"

    def test_calculate_equivalent_initial_short_rate_invalid_inputs(self, test_dates, daily_dt):
        """Test equivalent initial short rate calculation with invalid inputs."""
        straight_bond = Bond(
            issue_dt=test_dates["issue_date"],
            maturity_dt=test_dates["maturity_date"],
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.THIRTY_E_360
        )
        
        # Test invalid discount mode
        with pytest.raises(ValueError, match="Invalid discount mode"):
            CallableBondValuer.calculate_equivalent_initial_short_rate(
                straight_bond=straight_bond,
                dt=daily_dt,
                ytm=0.05,
                valuation_date=test_dates["issue_date"],
                discount_mode="invalid_mode"
            )


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__])

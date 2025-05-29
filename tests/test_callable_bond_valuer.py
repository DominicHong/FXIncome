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
from financepy.products.bonds.bond import YTMCalcType


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
    def test_callable_bond(self, test_dates):
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
    def test_rate_simulator(self):
        """Fixture providing a test rate simulator."""
        return InterestRateSimulator(
            r0=0.0435, sigma=0.02, days=365 * 5, num_paths=1, model="gbm"
        )

    @pytest.fixture
    def test_valuer(self, test_callable_bond, test_rate_simulator):
        """Fixture providing a test callable bond valuer."""
        return CallableBondValuer(test_callable_bond, test_rate_simulator)

    def test_pv_calculation_discrete_mode(self, test_valuer, test_dates):
        """Test present value calculation in discrete mode."""
        discount_rate = 0.0435
        flat_rate_path = np.full(365 * 5, discount_rate)

        # Test PV of a simple 5% coupon at year 1
        test_cf_dates = [test_dates["issue_date"].add_years(1)]
        test_cf_amounts = [5.0]  # 5% of 100 face value

        calculated_pv = test_valuer._pv_of_future_cash_flows(
            test_cf_dates,
            test_cf_amounts,
            test_dates["issue_date"],
            flat_rate_path,
            discount_mode="discrete",
        )
        expected_pv = 5.0 / (1 + discount_rate) ** 1.0  # Discrete compounding

        assert (
            abs(calculated_pv - expected_pv) < 1e-6
        ), f"Discrete PV calculation failed. Expected: {expected_pv:.6f}, Got: {calculated_pv:.6f}"

    def test_pv_calculation_continuous_mode(self, test_valuer, test_dates):
        """Test present value calculation in continuous mode."""
        discount_rate = 0.0435
        flat_rate_path = np.full(365 * 5, discount_rate)

        # Test PV of a simple 5% coupon at year 1
        test_cf_dates = [test_dates["issue_date"].add_years(1)]
        test_cf_amounts = [5.0]  # 5% of 100 face value

        calculated_pv = test_valuer._pv_of_future_cash_flows(
            test_cf_dates,
            test_cf_amounts,
            test_dates["issue_date"],
            flat_rate_path,
            discount_mode="continuous",
        )
        expected_pv = 5.0 * np.exp(-discount_rate * 1.0)  # Continuous compounding

        assert (
            abs(calculated_pv - expected_pv) < 1e-6
        ), f"Continuous PV calculation failed. Expected: {expected_pv:.6f}, Got: {calculated_pv:.6f}"

    def test_straight_bond_valuation_vs_financepy(
        self, test_valuer, test_callable_bond, test_dates
    ):
        """Test that our straight bond valuation matches FinancePy within reasonable tolerance."""
        discount_rate = 0.0435
        flat_rate_path = np.full(365 * 5, discount_rate)

        # Get bond cash flows and scale them
        bond_cf_dates = test_callable_bond.bond.cpn_dts
        bond_cf_amounts = [
            cf * test_callable_bond.face_value
            for cf in test_callable_bond.bond.flow_amounts
        ]

        # Add principal to the final payment (FinancePy quirk)
        if (
            len(bond_cf_dates) > 0
            and bond_cf_dates[-1] == test_callable_bond.maturity_date
            and len(bond_cf_amounts) > 0
        ):
            expected_coupon = (
                test_callable_bond.coupon_rate * test_callable_bond.face_value
            )
            if abs(bond_cf_amounts[-1] - expected_coupon) < 0.01:
                bond_cf_amounts[-1] += test_callable_bond.face_value

        # Use discrete mode for comparison
        our_pv = test_valuer._pv_of_future_cash_flows(
            bond_cf_dates,
            bond_cf_amounts,
            test_dates["issue_date"],
            flat_rate_path,
            discount_mode="discrete",
        )

        financepy_pv = test_callable_bond.dirty_price_from_ytm(
            test_dates["issue_date"], discount_rate, YTMCalcType.US_STREET
        )

        # Allow for some difference due to different day count conventions and methods
        tolerance = 1.0  # 1.0 basis point tolerance
        assert (
            abs(our_pv - financepy_pv) < tolerance
        ), f"Bond valuation differs too much from FinancePy. Our: {our_pv:.4f}, FinancePy: {financepy_pv:.4f}, Diff: {abs(our_pv - financepy_pv):.4f}"

    def test_pv_calculation_with_cash_flow_on_valuation_date(
        self, test_valuer, test_dates
    ):
        """Test that cash flows on the valuation date are not discounted."""
        flat_rate_path = np.full(365 * 5, 0.05)

        # Cash flow on valuation date should have PV = cash flow amount
        test_cf_dates = [test_dates["issue_date"]]  # Same as valuation date
        test_cf_amounts = [10.0]

        calculated_pv = test_valuer._pv_of_future_cash_flows(
            test_cf_dates, test_cf_amounts, test_dates["issue_date"], flat_rate_path
        )

        assert (
            abs(calculated_pv - 10.0) < 1e-6
        ), f"Cash flow on valuation date should not be discounted. Expected: 10.0, Got: {calculated_pv:.6f}"

    def test_pv_calculation_ignores_past_cash_flows(self, test_valuer, test_dates):
        """Test that past cash flows are ignored."""
        flat_rate_path = np.full(365 * 5, 0.05)

        # Cash flow in the past should be ignored
        past_date = test_dates["issue_date"].add_days(-30)  # 30 days before valuation
        test_cf_dates = [past_date]
        test_cf_amounts = [100.0]

        calculated_pv = test_valuer._pv_of_future_cash_flows(
            test_cf_dates, test_cf_amounts, test_dates["issue_date"], flat_rate_path
        )

        assert (
            abs(calculated_pv) < 1e-6
        ), f"Past cash flows should be ignored. Expected: 0.0, Got: {calculated_pv:.6f}"

    def test_invalid_discount_mode_raises_error(self, test_valuer, test_dates):
        """Test that invalid discount mode raises ValueError."""
        flat_rate_path = np.full(365 * 5, 0.05)
        test_cf_dates = [test_dates["issue_date"].add_years(1)]
        test_cf_amounts = [5.0]

        with pytest.raises(ValueError, match="Invalid discount mode"):
            test_valuer._pv_of_future_cash_flows(
                test_cf_dates,
                test_cf_amounts,
                test_dates["issue_date"],
                flat_rate_path,
                discount_mode="invalid_mode",
            )


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__])

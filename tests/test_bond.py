from financepy.utils import *
from financepy.products.bonds.bond import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


class TestBond:

    @pytest.fixture(scope='class')
    def global_data(self):
        #  220008 is a 30 year treasury bond, 2 coupon payments per year.
        treasury = Bond(
            issue_date=Date(15, 4, 2022),
            maturity_date=Date(15, 4, 2052),
            coupon=0.0332,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            accrual_type=DayCountTypes.ACT_ACT_ISDA
        )
        #  210210 is a 10 year CDB bond, 1 coupon payment per year.
        bond = Bond(
            issue_date=Date(7, 6, 2021),
            maturity_date=Date(7, 6, 2031),
            coupon=0.0341,
            freq_type=FrequencyTypes.ANNUAL,
            accrual_type=DayCountTypes.ACT_ACT_ISDA
        )
        #  220211 is a 1 year CDB bond, 1 coupon payment per year.
        bond_1y = Bond(
            issue_date=Date(28, 7, 2022),
            maturity_date=Date(28, 7, 2023),
            coupon=0.0174,
            freq_type=FrequencyTypes.ANNUAL,
            accrual_type=DayCountTypes.ACT_ACT_ISDA
        )
        # 229936 is a 3 months treasure with 0 coupon per year.
        bill = Bond(
            issue_date=Date(25, 7, 2022),
            maturity_date=Date(24, 10, 2022),
            coupon=0,
            freq_type=FrequencyTypes.ZERO,
            accrual_type=DayCountTypes.ZERO
        )
        curve_df = pd.DataFrame([[0, 2], [30, 2.5], [90, 2.8], [365, 3.0], [730, 3.2], [822, 3.21], [1095, 3.5]],
                                columns=['days', 'rate'])
        settlement_date = Date(8, 8, 2022)
        return {'treasury': treasury,
                'bond': bond,
                'bond_1y': bond_1y,
                'bill': bill,
                'curve': curve_df,
                'date': settlement_date}

    def test_ytm_treasury(self, global_data):
        bond = global_data['treasury']
        assess_date = global_data['date']
        clean_price = 104.3456
        assert bond.yield_to_maturity(assess_date, clean_price, YTMCalcType.US_STREET) * 100 == \
               pytest.approx(3.0950)

    def test_ytm_cdb_10y(self, global_data):
        bond = global_data['bond']
        assess_date = global_data['date']
        clean_price = 103.0155
        assert bond.yield_to_maturity(assess_date, clean_price, YTMCalcType.US_STREET) * 100 == \
               pytest.approx(3.0150)

    def test_ytm_cdb_1y(self, global_data):
        bond = global_data['bond_1y']
        assess_date = global_data['date']
        clean_price = 99.8887
        assert bond.yield_to_maturity(assess_date, clean_price, YTMCalcType.US_STREET) * 100 == \
               pytest.approx(1.8559, abs=1e-4)

    def test_ytm_zero_coupon(self, global_data):
        bond = global_data['bill']
        assess_date = global_data['date']
        clean_price = 99.6504
        assert bond.yield_to_maturity(assess_date, clean_price, YTMCalcType.ZERO) * 100 == \
               pytest.approx(1.3997, abs=1e-2)

    def test_accrued_zero_coupon(self, global_data):
        bond = global_data['bill']
        assess_date = global_data['date']
        assert bond.calc_accrued_interest(assess_date) == pytest.approx(0.055231, abs=1e-3)

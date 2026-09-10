"""Callable bond valuation using FinancePy's BondEmbeddedOption.

This module prices a callable bond with FinancePy's
`financepy.products.bonds.bond_embedded_option.BondEmbeddedOption` class,
which values the bond with an embedded (Bermudan) call using short-rate
tree models: Hull-White (HWTree), Black-Karasinski (BKTree) and
Black-Derman-Toy (BDTTree).

The scenario matches callable_bond_valuer.py:
  * 10y bond issued 27-May-2025, 4.6% annual coupon, 30E/360
  * 1 year call protection, then annual call dates at par (100)
  * hypothetical issuer whose straight bond trades at par

Market data (us_market_rates_2020_2025.csv):
  * SOFR overnight plus on-the-run Treasury par yields (tsy1y/3y/5y/7y/10y)

The primary discount curve is the market curve built by market_curve.py:
  * Treasury zero curve bootstrapped from the par yields (par-yield
    assumption: semiannual coupon = yield, price = 100)
  * shifted by a constant issuer Z-spread calibrated so the straight
    bond prices at exactly 100

A flat curve at ln(1 + ytm) is retained as a comparison baseline. The
Longstaff-Schwartz Monte Carlo cross-check from callable_bond_valuer.py
uses the market-curve anchoring (r0 = SOFR + spread, theta = 10y
Treasury par + spread).
"""

import datetime
import os

import matplotlib
matplotlib.use("Agg")  # no blocking plt.show() from the MC solver's plots

import numpy as np
import pandas as pd
from scipy import optimize

from financepy.products.bonds.bond import Bond, YTMCalcType
from financepy.products.bonds.bond_embedded_option import BondEmbeddedOption
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes
from financepy.utils.frequency import FrequencyTypes
from financepy.market.curves.discount_curve import DiscountCurve
from financepy.market.curves.flat_discount_curve import FlatDiscountCurve
from financepy.models.hw_tree import HWTree
from financepy.models.bk_tree import BKTree
from financepy.models.bdt_tree import BDTTree

from interest_rate_simulator import InterestRateSimulator
from market_curve import (
    MarketCurveBuilder,
    bond_pv_with_spread,
    build_risky_curve,
    calibrate_spread_to_bond,
    load_market_rates,
)
from callable_bond_valuer import (
    CallableBond,
    CallableBondValuer,
    solve_callable_bond_coupon,
)


def build_call_schedule(
    issue_dt: Date,
    maturity_dt: Date,
    call_protection_years: int,
    call_price: float,
) -> tuple[list[Date], np.ndarray]:
    """Annual call dates from the end of the protection period to maturity.

    Returns (call_dates, call_prices) as required by BondEmbeddedOption.
    """
    call_dts = []
    call_dt = issue_dt.add_years(call_protection_years)
    while call_dt < maturity_dt:
        call_dts.append(call_dt)
        call_dt = call_dt.add_years(1)
    call_prices = np.array([call_price] * len(call_dts))
    return call_dts, call_prices


def value_with_financepy(
    beo: BondEmbeddedOption,
    valuation_dt: Date,
    discount_curve: DiscountCurve,
    model,
) -> tuple[float, float, float]:
    """Value the callable bond with the given tree model.

    Returns (callable_bond_value, straight_bond_value, option_value).
    Values are dirty prices per 100 face. On the issue date accrued
    interest is zero, so dirty == clean.
    """
    v_with, v_pure = beo.value(valuation_dt, discount_curve, model)
    return v_with, v_pure, v_pure - v_with


def solve_callable_bond_coupon_financepy(
    issue_date: Date,
    maturity_date: Date,
    discount_curve: DiscountCurve,
    call_protection_years: int,
    call_price: float,
    model_factory,
    num_time_steps: int,
    freq_type: FrequencyTypes = FrequencyTypes.ANNUAL,
    dc_type: DayCountTypes = DayCountTypes.THIRTY_E_360,
    tolerance: float = 1e-6,
) -> tuple[float, dict]:
    """Solve for the coupon that makes the FinancePy tree callable value 100.

    The discount curve is held fixed (the market curve); only the coupon
    varies, exactly as in solve_callable_bond_coupon() in
    callable_bond_valuer.py.
    """
    target_value = 100.0
    call_dts, call_prices = build_call_schedule(
        issue_date, maturity_date, call_protection_years, call_price
    )

    def objective_function(coupon_rate: float) -> float:
        beo = BondEmbeddedOption(
            issue_dt=issue_date,
            maturity_dt=maturity_date,
            coupon=coupon_rate,
            freq_type=freq_type,
            accrual_dc_type=dc_type,
            call_dts=call_dts,
            call_prices=call_prices,
            put_dts=[],
            put_prices=np.array([]),
        )
        v_with, _, _ = value_with_financepy(
            beo, issue_date, discount_curve, model_factory(num_time_steps)
        )
        return v_with - target_value

    print("Testing optimization bounds...")
    lower_bound, upper_bound = 0.01, 0.12
    f_lower = objective_function(lower_bound)
    f_upper = objective_function(upper_bound)
    print(f"f({lower_bound:.1%}) = {f_lower:.4f}")
    print(f"f({upper_bound:.1%}) = {f_upper:.4f}")

    if f_lower * f_upper > 0:
        if f_lower > 0:
            lower_bound = 0.001
        else:
            upper_bound = 0.20
        f_lower = objective_function(lower_bound)
        f_upper = objective_function(upper_bound)
        print(f"Adjusted: f({lower_bound:.1%}) = {f_lower:.4f}, "
              f"f({upper_bound:.1%}) = {f_upper:.4f}")

    optimal_coupon = optimize.brentq(
        objective_function, a=lower_bound, b=upper_bound,
        xtol=tolerance, maxiter=100,
    )

    beo = BondEmbeddedOption(
        issue_dt=issue_date,
        maturity_dt=maturity_date,
        coupon=optimal_coupon,
        freq_type=freq_type,
        accrual_dc_type=dc_type,
        call_dts=call_dts,
        call_prices=call_prices,
        put_dts=[],
        put_prices=np.array([]),
    )
    v_with, v_pure, v_opt = value_with_financepy(
        beo, issue_date, discount_curve, model_factory(num_time_steps)
    )

    solution_info = {
        "optimal_coupon": optimal_coupon,
        "callable_bond_value": v_with,
        "straight_bond_value": v_pure,
        "target_value": target_value,
        "error": abs(v_with - target_value),
        "option_value": v_opt,
    }
    return optimal_coupon, solution_info


def main():
    print("=== Callable Bond Valuation with FinancePy BondEmbeddedOption ===\n")

    # ------------------------------------------------------------------
    # 1. Scenario - identical to callable_bond_valuer.py main()
    # ------------------------------------------------------------------
    issue_date = Date(27, 5, 2025)
    maturity_date = Date(27, 5, 2035)
    straight_bond_coupon = 0.046
    straight_bond_ytm = straight_bond_coupon
    call_protection_years = 1
    call_price = 100.0

    print("Problem Parameters:")
    print(f"  Issue date:          {issue_date}")
    print(f"  Maturity date:       {maturity_date}")
    print(f"  Straight bond coupon:{straight_bond_coupon:.2%}")
    print(f"  Straight bond YTM:   {straight_bond_ytm:.2%}")
    print(f"  Call protection:     {call_protection_years} year")
    print(f"  Call price:          {call_price}")
    print()

    # ------------------------------------------------------------------
    # 2. Calibrate sigma/kappa from the SOFR overnight-rate history
    #    (P-measure, as in the Monte Carlo pipeline). sigma/kappa feed
    #    the tree models.
    # ------------------------------------------------------------------
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "us_market_rates_2020_2025.csv",
    )

    rate_simulator = InterestRateSimulator(
        r0=0.044951,
        mu=0.044951,
        sigma=0.0117,
        days=int(maturity_date - issue_date) + 2,
        days_of_year=365,
        num_paths=1000,
        model="vasicek",
        kappa=1.0,
    )

    # Calibrate on the SOFR OVERNIGHT rate, not on a Treasury yield series.
    # The tree models need short-rate vol/mean-reversion; fitting the
    # Vasicek OLS to a long-maturity yield series overestimates the
    # mean-reversion speed relative to the short rate, which damps the
    # tree's long-rate volatility and prices the call at near zero.
    calib_start = datetime.datetime(2020, 1, 1)
    calib_end = datetime.datetime(2025, 5, 27)
    print(f"Calibrating rate model to historical SOFR overnight rates "
          f"({rate_simulator.model})...")
    rate_simulator.load_and_calibrate_from_csv(
        csv_path=csv_path,
        start_date=calib_start,
        end_date=calib_end,
        model=rate_simulator.model,
        rate_col="sofr_rate",  # column in the merged us_market_rates file
    )
    print(
        f"Calibrated: kappa={rate_simulator.kappa:.4f}, "
        f"theta(mu)={rate_simulator.mu:.4%}, sigma={rate_simulator.sigma:.4%}\n"
    )

    sigma = rate_simulator.sigma
    kappa = rate_simulator.kappa

    # Vol sanity check: the tree prices the embedded call off the
    # model-implied long-rate vol sigma*(1-exp(-k*T))/(k*T). Compare it
    # with the historical par-yield vol of a matching tenor over the
    # same window - a gross mismatch means kappa/sigma are not fit for
    # purpose and the option values below are meaningless.
    hist_df = pd.read_csv(csv_path, parse_dates=["date"])
    calib_df = hist_df[
        (hist_df["date"] >= calib_start) & (hist_df["date"] <= calib_end)
    ]
    market_10y_vol = calib_df["tsy10y_rate"].diff().std() * np.sqrt(252)
    T_check = 10.0
    implied_10y_vol = (
        sigma * (1.0 - np.exp(-kappa * T_check)) / (kappa * T_check)
        if kappa > 1e-12
        else sigma
    )
    print("Vol sanity check (10y horizon):")
    print(f"  Model-implied 10y zero-rate vol: {implied_10y_vol:.4%}")
    print(f"  Historical 10y par-yield vol:     {market_10y_vol:.4%}\n")

    # ------------------------------------------------------------------
    # 3. Market discount curve (primary): Treasury zero curve bootstrapped
    #    from the on-the-run par yields at the issue date, shifted by the
    #    issuer Z-spread calibrated so the straight bond prices at 100.
    #    The flat curve at ln(1 + ytm) is kept as a comparison baseline.
    # ------------------------------------------------------------------
    straight_bond = Bond(
        issue_dt=issue_date,
        maturity_dt=maturity_date,
        coupon=straight_bond_coupon,
        freq_type=FrequencyTypes.ANNUAL,
        accrual_dc_type=DayCountTypes.THIRTY_E_360,
    )

    market_rates = load_market_rates(csv_path, issue_date)
    builder = MarketCurveBuilder(issue_date, market_rates["par_yields"])
    builder.print_summary()

    spread, straight_at_spread = calibrate_spread_to_bond(
        builder, straight_bond, target_price=100.0
    )
    discount_curve = build_risky_curve(builder, spread)
    straight_on_treasury = bond_pv_with_spread(builder, straight_bond, 0.0)
    print(f"\nIssuer Z-spread: {spread:.4%}")
    print(f"Straight bond PV on Treasury curve: {straight_on_treasury:.4f}")
    print(f"Straight bond PV at spread:         {straight_at_spread:.4f} "
          f"(target 100)")

    flat_curve = FlatDiscountCurve(issue_date, np.log(1.0 + straight_bond_ytm))

    # ------------------------------------------------------------------
    # 4. Build the BondEmbeddedOption
    # ------------------------------------------------------------------
    call_dts, call_prices = build_call_schedule(
        issue_date, maturity_date, call_protection_years, call_price
    )
    print(f"Call dates ({len(call_dts)}): "
          f"{call_dts[0]} ... {call_dts[-1]}\n")

    beo = BondEmbeddedOption(
        issue_dt=issue_date,
        maturity_dt=maturity_date,
        coupon=straight_bond_coupon,
        freq_type=FrequencyTypes.ANNUAL,
        accrual_dc_type=DayCountTypes.THIRTY_E_360,
        call_dts=call_dts,
        call_prices=call_prices,
        put_dts=[],
        put_prices=np.array([]),
    )

    # ------------------------------------------------------------------
    # 5. Value with each tree model and check convergence in time steps
    #
    #    Primary: the market curve (Treasury bootstrap + issuer spread).
    #    Baseline: the flat curve at ln(1 + ytm).
    #
    # Vol unit conventions:
    #   * HWTree: dr = (theta - a*r)*dt + sigma*dW  -> ABSOLUTE short-rate
    #     vol; the Vasicek sigma calibrated above maps directly onto it.
    #   * BKTree/BDTTree: d(ln r) = (theta - a*ln r)*dt + sigma*dW  ->
    #     PROPORTIONAL short-rate vol, so the calibrated absolute sigma
    #     must be rescaled by the current short-rate level (sigma / r0).
    # With the correct units all three trees agree on a meaningful
    # option value and converge cleanly in the number of steps. The HW
    # tree is the natural primary model here: it is the Gaussian
    # short-rate model whose a (mean reversion) and sigma map directly
    # onto the Vasicek parameters calibrated from SOFR history.
    # ------------------------------------------------------------------
    sofr0 = market_rates["sofr"]
    sigma_log = sigma / sofr0
    print(f"Proportional vol for lognormal trees (sigma/SOFR): {sigma_log:.4f}\n")

    models = {
        "HW (Vasicek-equivalent)": lambda n: HWTree(sigma, kappa, n),
        "BK (Black-Karasinski)": lambda n: BKTree(sigma_log, kappa, n),
        "BDT (Black-Derman-Toy)": lambda n: BDTTree(sigma_log, n),
    }

    print("\n=== FinancePy Tree Valuation (market curve) ===")
    print(f"{'Model':<26}{'Steps':>6}{'Callable':>10}{'Straight':>10}"
          f"{'Option':>10}")
    print("-" * 62)

    results = {}
    for name, factory in models.items():
        for steps in (100, 200, 400):
            model = factory(steps)
            v_with, v_pure, v_opt = value_with_financepy(
                beo, issue_date, discount_curve, model
            )
            results[(name, steps)] = (v_with, v_pure, v_opt)
            print(f"{name:<26}{steps:>6}{v_with:>10.4f}{v_pure:>10.4f}"
                  f"{v_opt:>10.4f}")

    # NB: HW's long-rate volatility is damped by mean reversion
    # (roughly 1/kappa), so the option value is very sensitive to kappa.
    print("\nKappa (mean reversion) sensitivity - HWTree, 400 steps:")
    print(f"{'kappa':>8}{'Callable':>10}{'Option':>10}")
    for k_test in (0.1, 1.0, 3.0, kappa):
        v_with, v_pure, v_opt = value_with_financepy(
            beo, issue_date, discount_curve, HWTree(sigma, k_test, 400)
        )
        note = " (calibrated)" if abs(k_test - kappa) < 1e-6 else ""
        print(f"{k_test:>8.4f}{v_with:>10.4f}{v_opt:>10.4f}{note}")

    print("\n=== FinancePy Tree Valuation (flat baseline) ===")
    print(f"{'Model':<26}{'Steps':>6}{'Callable':>10}{'Straight':>10}"
          f"{'Option':>10}")
    print("-" * 62)

    flat_results = {}
    for name, factory in models.items():
        for steps in (100, 200, 400):
            model = factory(steps)
            v_with, v_pure, v_opt = value_with_financepy(
                beo, issue_date, flat_curve, model
            )
            flat_results[(name, steps)] = (v_with, v_pure, v_opt)
            print(f"{name:<26}{steps:>6}{v_with:>10.4f}{v_pure:>10.4f}"
                  f"{v_opt:>10.4f}")

    # ------------------------------------------------------------------
    # 6. Cross-check against the Longstaff-Schwartz Monte Carlo valuer,
    #    anchored to the market curve (r0 = SOFR + spread, theta = 10y
    #    Treasury + spread). The straight-bond reference remains 100, the
    #    same basis as the tree's spread-adjusted curve.
    # ------------------------------------------------------------------
    print("\n=== Cross-check: Longstaff-Schwartz Monte Carlo ===")
    print("(market-curve anchoring; straight bond reference = 100)")

    CallableBondValuer.anchor_to_market_curve(
        rate_simulator=rate_simulator,
        straight_bond=straight_bond,
        csv_path=csv_path,
    )
    print()

    callable_bond = CallableBond(
        issue_date=issue_date,
        maturity_date=maturity_date,
        coupon_rate=straight_bond_coupon,
        call_protection_years=call_protection_years,
        freq_type=FrequencyTypes.ANNUAL,
        day_count_type=DayCountTypes.THIRTY_E_360,
    )
    valuer = CallableBondValuer(callable_bond, rate_simulator)

    mc_value, mc_straight, mc_stats = valuer.value_bond(
        valuation_date=issue_date,
        straight_bond_ytm=straight_bond_ytm,
        num_simulations=2000,
        seed=42,
    )
    print(f"  MC callable bond value: {mc_value:.4f}")
    print(f"  MC straight bond value: {mc_straight:.4f}")
    print(f"  MC option value:        {mc_stats['option_value']:.4f}")
    print(f"  MC call probability:    {mc_stats['call_probability']:.2%}")

    # ------------------------------------------------------------------
    # 7. Summary
    #
    #    The tree result (HW) is exact backward induction on the Q-measure
    #    short-rate tree fitted to the market curve and is the reference
    #    value. With short-rate-calibrated vol parameters (and the
    #    lognormal trees on proportional sigma) the embedded call carries
    #    meaningful time value, so the callable price sits below the
    #    straight price at the par coupon.
    #    The LSM Monte Carlo value is a low-biased estimator with a
    #    quadratic regression basis and a constant-theta approximation
    #    to the curve, so its option value can differ from the tree
    #    price; it is a cross-check, not the reference.
    # ------------------------------------------------------------------
    print("\n=== Summary (market curve) ===")
    print("FinancePy BondEmbeddedOption (HWTree, 400 steps):")
    v_with, v_pure, v_opt = results[("HW (Vasicek-equivalent)", 400)]
    print(f"  Callable bond value: {v_with:.4f}")
    print(f"  Straight bond value: {v_pure:.4f}")
    print(f"  Embedded call value: {v_opt:.4f}")
    print(f"  MC option value:     {mc_stats['option_value']:.4f}")

    # ------------------------------------------------------------------
    # 8. Equivalent coupon: the coupon that makes the callable bond value
    #    exactly 100, given the market curve. Solved with each method:
    #    HWTree, BDTTree (tree, market curve) and LSM Monte Carlo
    #    (market-curve anchoring).
    # ------------------------------------------------------------------
    print("\n=== Solving for Equivalent Callable Bond Coupon (target = 100) ===")
    print("(market curve: Treasury bootstrap + issuer Z-spread)")

    hw_coupon, hw_info = solve_callable_bond_coupon_financepy(
        issue_date=issue_date,
        maturity_date=maturity_date,
        discount_curve=discount_curve,
        call_protection_years=call_protection_years,
        call_price=call_price,
        model_factory=lambda n: HWTree(sigma, kappa, n),
        num_time_steps=400,
    )
    print(f"HWTree  equivalent coupon: {hw_coupon:.4%}")

    bdt_coupon, bdt_info = solve_callable_bond_coupon_financepy(
        issue_date=issue_date,
        maturity_date=maturity_date,
        discount_curve=discount_curve,
        call_protection_years=call_protection_years,
        call_price=call_price,
        model_factory=lambda n: BDTTree(sigma_log, n),
        num_time_steps=400,
    )
    print(f"BDTTree equivalent coupon: {bdt_coupon:.4%}")

    print("\nSolving with LSM Monte Carlo (this takes a few minutes)...")
    rate_simulator.num_paths = 1500
    rate_simulator.seed = 42
    mc_coupon, mc_coupon_info = solve_callable_bond_coupon(
        rate_simulator=rate_simulator,
        straight_bond=straight_bond,
        straight_bond_ytm=straight_bond_ytm,
        call_protection_years=call_protection_years,
        anchor_mode="market_curve",
        csv_path=csv_path,
    )
    print(f"LSM MC   equivalent coupon: {mc_coupon:.4%}")

    print("\n=== Equivalent Coupon Comparison ===")
    print(f"  HWTree (Hull-White):   {hw_coupon:.4%}")
    print(f"  BDTTree (Black-D-T):    {bdt_coupon:.4%}")
    print(f"  LSM Monte Carlo:        {mc_coupon:.4%}")


if __name__ == "__main__":
    main()

"""Market discount-curve construction from the merged US market-rates CSV.

At a given valuation date this module builds:

  * A Treasury zero (discount) curve bootstrapped from the on-the-run
    Treasury par yields (tsy1y/3y/5y/7y/10y_rate) in
    us_market_rates_2020_2025.csv. A par yield defines its own cash
    flows by construction: semiannual coupon = par yield, price = 100,
    principal at maturity - no additional security data is required.
    Interpolation between tenor knots is flat forward rates.

  * A risky (issuer) curve = Treasury curve shifted by a constant credit
    spread (Z-spread) calibrated so that a given straight bond prices at
    a target price. The issuer curve keeps the straight bond at par even
    when the Treasury curve implies a different price.
"""

import os

import numpy as np
import pandas as pd
from scipy import optimize

from financepy.market.curves.discount_curve import DiscountCurve
from financepy.market.curves.interpolator import InterpTypes
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes

# Tenor knots (years) of the Treasury par yield curve.
TENORS = (1.0, 3.0, 5.0, 7.0, 10.0)

# Par-yield cash-flow convention: semiannual coupons, par 100.
PAYMENT_STEP = 0.5
PAR = 100.0

# Output grid step (years) of the constructed DiscountCurve.
GRID_STEP = 0.25

PAR_COLUMNS = {
    1.0: "tsy1y_rate",
    3.0: "tsy3y_rate",
    5.0: "tsy5y_rate",
    7.0: "tsy7y_rate",
    10.0: "tsy10y_rate",
}


def load_market_rates(csv_path: str, valuation_dt: Date) -> dict:
    """Load the market rates at (or just before) the valuation date.

    Returns a dict with:
      * "sofr": the SOFR overnight rate (decimal)
      * "par_yields": {tenor_years: par yield (decimal)}
      * "as_of": the pd.Timestamp of the data row used
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    target = pd.Timestamp(valuation_dt.y, valuation_dt.m, valuation_dt.d)
    df = df[df["date"] <= target]
    if df.empty:
        raise ValueError(f"No market data on or before {target}")
    row = df.iloc[-1]

    par_yields = {}
    for tenor, col in PAR_COLUMNS.items():
        value = row[col]
        if pd.isna(value):
            raise ValueError(
                f"Par yield column '{col}' missing on {row['date'].date()}"
            )
        par_yields[tenor] = float(value)

    if pd.isna(row["sofr_rate"]):
        raise ValueError(f"SOFR missing on {row['date'].date()}")

    return {
        "sofr": float(row["sofr_rate"]),
        "par_yields": par_yields,
        "as_of": row["date"],
    }


class MarketCurveBuilder:
    """Bootstrap a Treasury discount curve from par yields.

    The curve is piecewise flat-forward: within each segment
    [t_prev, T] the forward rate is constant, and the discount factor
    DF(T) at each tenor knot is solved sequentially from the par
    equation  100 = sum(c * DF(t_i)) + 100 * DF(T).
    """

    def __init__(self, valuation_dt: Date, par_yields: dict):
        self.valuation_dt = valuation_dt
        self.par_yields = dict(par_yields)

        # segments[i] = (t_start, df_start, forward) covering [t_start, next)
        self.segments = []  # type: list[tuple[float, float, float]]
        self._df_at_knots = {0.0: 1.0}

        self._bootstrap()
        self.treasury_curve = self._build_financepy_curve()

    # ------------------------------------------------------------------

    def df_at(self, times) -> np.ndarray:
        """Discount factors at arbitrary times (years) from the bootstrapped curve."""
        times = np.atleast_1d(np.asarray(times, dtype=float))
        dfs = np.empty_like(times)
        for i, t in enumerate(times):
            dfs[i] = self._df_scalar(float(t))
        return dfs

    def zero_rate_at(self, times) -> np.ndarray:
        """Continuously compounded zero rates implied by the curve."""
        dfs = self.df_at(times)
        rates = np.where(dfs > 0.0, -np.log(dfs) / np.maximum(times, 1e-12), 0.0)
        return rates

    # ------------------------------------------------------------------

    def _df_scalar(self, t: float) -> float:
        if t <= 0.0:
            return 1.0
        seg = self.segments
        for i, (t_start, df_start, fwd) in enumerate(seg):
            t_end = seg[i + 1][0] if i + 1 < len(seg) else np.inf
            if t <= t_end:
                return df_start * np.exp(-fwd * (t - t_start))
        # beyond the last knot: flat forward extrapolation
        t_start, df_start, fwd = seg[-1]
        return df_start * np.exp(-fwd * (t - t_start))

    def _bootstrap(self) -> None:
        prev_t, prev_df = 0.0, 1.0
        for T in TENORS:
            y = self.par_yields[T]
            c = y * PAR / 2.0  # semiannual coupon per 100 face

            def price_par(df_T: float) -> float:
                """Par-bond price error given DF(T), flat forward on (prev_t, T]."""
                pv = 0.0
                t = PAYMENT_STEP
                while t < T - 1e-9:
                    if t <= prev_t:
                        df_t = self._df_scalar(t)
                    else:
                        # flat forward on (prev_t, T]
                        df_t = prev_df * (df_T / prev_df) ** ((t - prev_t) / (T - prev_t))
                    pv += c * df_t
                    t += PAYMENT_STEP
                pv += (PAR + c) * df_T
                return pv - PAR

            # DF(T) decreases as yields rise; root in (0, prev_df]
            df_T = optimize.brentq(
                price_par, 1e-12, prev_df, xtol=1e-14, maxiter=200
            )
            fwd = -np.log(df_T / prev_df) / (T - prev_t)
            self.segments.append((prev_t, prev_df, fwd))
            self._df_at_knots[T] = df_T
            prev_t, prev_df = T, df_T

    def _build_financepy_curve(self) -> DiscountCurve:
        grid_times = np.arange(0.0, TENORS[-1] + GRID_STEP / 2.0, GRID_STEP)
        df_dates = [self.valuation_dt.add_years(t) for t in grid_times]
        # Compute the DFs at the exact ACT/365F times of the date grid, so
        # the stored (times, dfs) pairs are internally consistent.
        actual_times = np.array(
            [(dt - self.valuation_dt) / 365.0 for dt in df_dates], dtype=float
        )
        grid_dfs = self.df_at(actual_times)
        return DiscountCurve(
            self.valuation_dt,
            df_dates,
            grid_dfs,
            interp_type=InterpTypes.FLAT_FWD_RATES,
            time_dc_type=DayCountTypes.ACT_365F,
        )

    # ------------------------------------------------------------------

    def par_reprice_errors(self) -> dict:
        """Price each par bond (semiannual, coupon = par yield) on the curve."""
        errors = {}
        for T, y in self.par_yields.items():
            c = y * PAR / 2.0
            pv = 0.0
            t = PAYMENT_STEP
            while t < T - 1e-9:
                pv += c * self._df_scalar(t)
                t += PAYMENT_STEP
            pv += (PAR + c) * self._df_scalar(T)
            errors[T] = pv - PAR
        return errors

    def print_summary(self) -> None:
        print("=== Bootstrapped Treasury Curve (flat-forward, par-yield assumption) ===")
        print(f"{'Tenor':>6} {'Par yield':>10} {'Zero rate':>10} {'DF':>10}")
        print("-" * 38)
        for T in TENORS:
            df_T = self._df_at_knots[T]
            zero = -np.log(df_T) / T
            print(f"{T:>5.0f}y {self.par_yields[T]:>10.4%} {zero:>10.4%} {df_T:>10.6f}")
        errors = self.par_reprice_errors()
        print(f"Max par-bond repricing error: {max(abs(e) for e in errors.values()):.2e}")


def bond_pv_with_spread(
    builder: MarketCurveBuilder,
    straight_bond,
    spread: float,
) -> float:
    """Dirty price (per 100 face) of a FinancePy Bond under treasury curve
    + constant spread.

    FinancePy's flow_amounts are per unit face value, so they are scaled
    by par; the principal is paid at maturity.
    """
    valuation_dt = builder.valuation_dt
    cpn_dts = straight_bond.cpn_dts
    flow_amounts = np.asarray(straight_bond.flow_amounts, dtype=float)
    par = straight_bond.par

    times = np.array([(dt - valuation_dt) / 365.0 for dt in cpn_dts], dtype=float)
    t_mat = (straight_bond.maturity_dt - valuation_dt) / 365.0
    dfs = builder.df_at(times) * np.exp(-spread * times)
    df_mat = float(builder.df_at(t_mat) * np.exp(-spread * t_mat))
    return float(np.sum(flow_amounts * par * dfs) + par * df_mat)


def calibrate_spread_to_bond(
    builder: MarketCurveBuilder,
    straight_bond,
    target_price: float = 100.0,
) -> tuple[float, float]:
    """Constant Z-spread such that the straight bond prices at target_price.

    Returns (spread, bond_price_at_spread).
    """
    def objective(s: float) -> float:
        return bond_pv_with_spread(builder, straight_bond, s) - target_price

    spread = optimize.brentq(objective, -0.20, 0.20, xtol=1e-10, maxiter=200)
    return spread, bond_pv_with_spread(builder, straight_bond, spread)


def build_risky_curve(
    builder: MarketCurveBuilder,
    spread: float,
) -> DiscountCurve:
    """Treasury curve shifted by a constant Z-spread: DF(t) * exp(-s*t)."""
    valuation_dt = builder.valuation_dt
    grid_times = np.arange(0.0, TENORS[-1] + GRID_STEP / 2.0, GRID_STEP)
    df_dates = [valuation_dt.add_years(t) for t in grid_times]
    actual_times = np.array(
        [(dt - valuation_dt) / 365.0 for dt in df_dates], dtype=float
    )
    risky_dfs = builder.df_at(actual_times) * np.exp(-spread * actual_times)
    return DiscountCurve(
        valuation_dt,
        df_dates,
        risky_dfs,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        time_dc_type=DayCountTypes.ACT_365F,
    )

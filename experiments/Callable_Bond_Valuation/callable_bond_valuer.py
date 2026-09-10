import datetime
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

# Import FinancePy classes
from financepy.products.bonds.bond import Bond, YTMCalcType
from financepy.utils.date import Date
from financepy.utils.frequency import FrequencyTypes
from financepy.utils.day_count import DayCountTypes

from interest_rate_simulator import InterestRateSimulator


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
            accrual_dc_type=day_count_type,
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

            if end_idx < days_to_cf:
                raise ValueError(
                    "rate_path is shorter than the time to a cash flow date "
                    f"(path length={max_path_idx}, days_to_cf={days_to_cf}). "
                    "Increase InterestRateSimulator.days to cover the full bond life."
                )

            relevant_rates = rate_path[0:end_idx]

            # This assumes simple summation; for more precision, one might align dt with actual day count to cf.
            # However, with daily dt, sum(rates*dt) is standard.

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
            day_count_type=straight_bond.accrual_dc_type,
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

    @classmethod
    def anchor_to_risk_neutral_measure(
        cls,
        rate_simulator: InterestRateSimulator,
        straight_bond: Bond,
        straight_bond_ytm: float,
    ) -> tuple[float, dict]:
        """Re-anchor an InterestRateSimulator from P-measure to Q-measure.

        We compute the "equivalent short rate" - the flat rate that
        reproduces today's analytical straight-bond PV inside this MC
        discounting framework - and set BOTH r0 and theta (mu) to it for
        mean-reverting models (Vasicek/CIR).  This ensures:

        * The rate starts at its long-term mean: no deterministic drift
          component at initiation.
        * Mean reversion keeps the rate near this level on average.
        * The straight-bond PV under the MC simulation approximately
          equals the analytical PV the calibration was designed to match.

        `kappa` and `sigma` are KEPT from the historical P-measure
        calibration - the standard simplifying assumption that volatility
        and mean-reversion speed are (approximately) measure-invariant.

        Caveat: when liquid swaption or bond-option data are available,
        calibrate κ and σ to those Q-measure prices rather than relying on
        historical P-measure estimates.  The values used here are a
        practical fallback.

        ABM/GBM are not mean-reverting (mu is a drift, not a reversion
        level), so their `mu` is left untouched; only `r0` is re-anchored.

        Args:
            rate_simulator: Simulator whose kappa/sigma/mu(theta) have already
                been calibrated to historical data (P-measure). Mutated in
                place: r0 (and mu, for mean-reverting models) is overwritten.
            straight_bond: FinancePy Bond used to derive the Q-measure theta.
            straight_bond_ytm: Fair YTM of the straight bond on the issue date.

        Returns:
            Tuple of (equivalent_short_rate, calibration_info).
        """
        print("Calibrating equivalent rate (r0 & Q-theta) to straight bond...")
        equivalent_rate, calib_info = (
            cls.calculate_equivalent_initial_short_rate(
                straight_bond=straight_bond,
                dt=rate_simulator._dt,
                ytm=straight_bond_ytm,
                valuation_date=straight_bond.issue_dt,
                discount_mode="discrete",
                tolerance=1e-6,
            )
        )
        print(f"Equivalent short rate: {equivalent_rate:.4%}")
        print(f"Calibration error: {calib_info['error']:.6f}")

        rate_simulator.r0 = equivalent_rate

        if rate_simulator.model in ("vasicek", "cir"):
            rate_simulator.mu = equivalent_rate
            print(f"Anchored r0 & theta (mu) = {equivalent_rate:.4%}")
        else:
            print(f"Anchored r0 = {equivalent_rate:.4%} (mu kept from P-measure)")

        return equivalent_rate, calib_info

    @classmethod
    def anchor_to_market_curve(
        cls,
        rate_simulator: InterestRateSimulator,
        straight_bond: Bond,
        csv_path: str | None = None,
    ) -> tuple[float, dict]:
        """Anchor the simulator to the market: r0 = SOFR + Z-spread,
        theta = 10y Treasury par yield + Z-spread.

        The Treasury zero curve is bootstrapped from the on-the-run par
        yields (tsy1y/3y/5y/7y/10y_rate) in the merged market-data CSV,
        and the issuer's constant Z-spread is calibrated so that the
        straight bond prices at 100 (its observed issue price).
        kappa/sigma remain the historical
        P-measure estimates (measure-invariance assumption), exactly as
        in anchor_to_risk_neutral_measure().

        Args:
            rate_simulator: Simulator whose kappa/sigma have already been
                calibrated. r0 (and mu, for mean-reverting models) is
                overwritten in place.
            straight_bond: FinancePy Bond used to calibrate the Z-spread.
            csv_path: Path to us_market_rates_2020_2025.csv. Defaults to
                the file next to this module.

        Returns:
            Tuple of (spread, anchoring_info).
        """
        from market_curve import (
            MarketCurveBuilder,
            calibrate_spread_to_bond,
            load_market_rates,
        )

        if csv_path is None:
            csv_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "us_market_rates_2020_2025.csv",
            )

        valuation_dt = straight_bond.issue_dt

        rates = load_market_rates(csv_path, valuation_dt)
        builder = MarketCurveBuilder(valuation_dt, rates["par_yields"])
        spread, price_at_spread = calibrate_spread_to_bond(
            builder, straight_bond, target_price=100.0
        )

        r0 = rates["sofr"] + spread
        theta = rates["par_yields"][10.0] + spread

        rate_simulator.r0 = r0
        if rate_simulator.model in ("vasicek", "cir"):
            rate_simulator.mu = theta
            print(
                f"Market-curve anchoring: r0 = SOFR + spread = {r0:.4%}, "
                f"theta = 10y par + spread = {theta:.4%}"
            )
        else:
            print(
                f"Market-curve anchoring: r0 = SOFR + spread = {r0:.4%} "
                f"(mu kept from P-measure)"
            )
        print(
            f"  SOFR = {rates['sofr']:.4%} (as of {rates['as_of'].date()}), "
            f"Z-spread = {spread:.4%}, straight PV at spread = "
            f"{price_at_spread:.4f}"
        )

        anchoring_info = {
            "spread": spread,
            "r0": r0,
            "theta": theta,
            "sofr": rates["sofr"],
            "par_yields": rates["par_yields"],
            "straight_pv_at_spread": price_at_spread,
        }
        return spread, anchoring_info

    def value_bond(
        self,
        valuation_date: Date,
        straight_bond_ytm: float,
        num_simulations: int | None = None,
        discount_mode: str = "discrete",
        seed: int | None = None,
    ) -> tuple[float, float, dict]:
        """Value the callable bond using Longstaff-Schwartz Monte Carlo.

        The LSM algorithm estimates the conditional continuation value at
        each call date via least squares regression across paths (using
        only the in-the-money samples), then back-propagates the optimal
        exercise policy. This avoids look-ahead bias and produces a
        (low-biased) estimate of the Bermudan callable price.

        Args:
            valuation_date: Date to value the bond
            straight_bond_ytm: Yield to maturity used for the analytical
                straight-bond reference value (no call option)
            num_simulations: Number of MC paths (overrides rate_simulator.num_paths)
            discount_mode: "discrete" or "continuous" intra-path discounting
            seed: Optional random seed forwarded to the rate simulator

        Returns:
            Tuple of (callable_bond_value, straight_bond_value, additional_stats)
        """
        if num_simulations is not None:
            self.rate_simulator.num_paths = num_simulations
        if seed is not None:
            self.rate_simulator.seed = seed

        # Generate interest rate paths. The simulator honours its own seed.
        self.rate_simulator.generate_rates()
        rates_df = self.rate_simulator.rates_df
        R = rates_df.values.T  # shape [num_paths, num_days]; R[p, d] = rate for day d -> d+1

        # Analytical straight-bond reference value.
        straight_bond_value = self.callable_bond.dirty_price_from_ytm(
            valuation_date, straight_bond_ytm, YTMCalcType.US_STREET
        )

        num_paths, num_days = R.shape
        dt = self.rate_simulator._dt

        # Discount-factor array G[p, d] = PV at valuation_date (index 0) of 1 unit
        # paid at index d on path p. PV(a -> b) = G[p, b] / G[p, a].
        if discount_mode == "discrete":
            P = np.ones((num_paths, num_days))
            # P[p, d] = prod_{s=0..d-1} (1 + R[p, s] * dt)
            P[:, 1:] = np.cumprod(1.0 + R[:, :-1] * dt, axis=1)
            G = 1.0 / P
        elif discount_mode == "continuous":
            LD = np.zeros((num_paths, num_days))
            LD[:, 1:] = -np.cumsum(R[:, :-1] * dt, axis=1)
            G = np.exp(LD)
        else:
            raise ValueError(f"Invalid discount mode: {discount_mode}")

        # Bond cash flows strictly after valuation_date. The last entry of
        # cf_amounts includes the principal (cashflow_since_date adds face_value).
        cf_dates, cf_amounts_list = self.callable_bond.cashflow_since_date(
            valuation_date
        )
        cf_indices = np.array(
            [int(round(cf_date - valuation_date)) for cf_date in cf_dates],
            dtype=int,
        )
        cf_amounts = np.array(cf_amounts_list, dtype=float)

        # The simulated horizon must cover every cash flow; otherwise the on-path
        # discounting would silently under-count periods.
        if cf_indices.size > 0 and np.any(cf_indices >= num_days):
            raise ValueError(
                "Simulated rate path does not cover the latest bond cash flow date "
                f"(num_days={num_days}, max cf index={int(cf_indices.max())}). "
                "Increase InterestRateSimulator.days."
            )

        # Future call dates and their integer day indices from valuation_date.
        future_call_dates = [
            cd for cd in self.callable_bond.call_dates if cd > valuation_date
        ]
        if not future_call_dates:
            # No remaining call opportunity: callable degrades to the straight bond.
            return straight_bond_value, straight_bond_value, {
                "call_probability": 0.0,
                "option_value": 0.0,
                "callable_values_std": 0.0,
                "call_statistics": {
                    "call_frequency": 0,
                    "call_dates": [],
                    "call_values": [],
                },
            }

        call_indices = np.array(
            [int(round(cd - valuation_date)) for cd in future_call_dates],
            dtype=int,
        )
        if np.any(call_indices >= num_days):
            raise ValueError(
                "Simulated rate path does not cover one or more call dates "
                f"(num_days={num_days}, max call index={int(call_indices.max())}). "
                "Increase InterestRateSimulator.days."
            )

        M = len(call_indices)
        K = self.callable_bond.face_value + self.call_premium  # issuer's call price

        # Pure coupon paid on each call date (principal excluded). FinancePy's
        # flow_amounts never include principal, so we can take the coupon directly.
        # NB: FinancePy's Date class is not hashable, so we cannot use a dict keyed
        # by Date. Iterate and compare call dates against coupon dates instead.
        cpn_dts = self.callable_bond.bond.cpn_dts
        cpn_flows = [
            cf * self.callable_bond.face_value
            for cf in self.callable_bond.bond.flow_amounts
        ]
        coupon_at_call = np.zeros(M, dtype=float)
        for k, cd in enumerate(future_call_dates):
            for cpn_dt, cpn_amt in zip(cpn_dts, cpn_flows):
                if cpn_dt == cd:
                    coupon_at_call[k] = cpn_amt
                    break

        # exercised_at[p] = -1 means "hold to maturity under current policy"
        # exercised_at[p] = k means "exercise at call date index k"
        exercised_at = np.full(num_paths, -1, dtype=int)

        # Pre-compute the "hold-to-maturity" continuation value at every call date:
        # htm_pv[p, k] = PV at t_k of all bond CFs strictly after t_k (full
        # straight-bond profile, including principal), on path p.
        htm_pv = np.zeros((num_paths, M))
        for k in range(M):
            t_k = call_indices[k]
            future_cf_mask = cf_indices > t_k
            if not np.any(future_cf_mask):
                continue
            future_cf_idx = cf_indices[future_cf_mask]
            future_cf_amt = cf_amounts[future_cf_mask]
            # PV at t_k = sum_j cf_j * G[p, cf_idx_j] / G[p, t_k]
            G_future = G[:, future_cf_idx]  # [num_paths, J]
            weighted = G_future * future_cf_amt[np.newaxis, :]
            htm_pv[:, k] = weighted.sum(axis=1) / G[:, t_k]

        # Longstaff-Schwartz backward induction. Start from the last future call
        # date and work backwards. At each step k, regress the continuation value
        # (under the already-decided future policy) on the state r_{t_k}, using
        # only in-the-money paths (C > K), then decide whether to exercise early.
        min_itm_samples = 3  # need at least 3 ITM paths for 3 regressors
        for k in range(M - 1, -1, -1):
            t_k = call_indices[k]
            G_tk = G[:, t_k]

            # Continuation C[p] at t_k depends on whether this path is already
            # committed to exercising at some later call date j_idx > k.
            C = np.zeros(num_paths)

            # Case A: hold to maturity - reuse pre-computed profile.
            case_A = exercised_at == -1
            if np.any(case_A):
                C[case_A] = htm_pv[case_A, k]

            # Case B: already committed to a later call date j > k. The
            # continuation under current policy is "coupons strictly in (t_k, t_j)
            # + (K + coupon at t_j) at t_j", discounted back to t_k.
            case_B = ~case_A
            if np.any(case_B):
                paths_B = np.where(case_B)[0]
                j_vals = exercised_at[paths_B]
                for j in np.unique(j_vals):
                    t_j = call_indices[j]
                    coup_t_j = coupon_at_call[j]
                    sel = paths_B[j_vals == j]
                    pv_sel = np.zeros(len(sel))
                    cps_mask = (cf_indices > t_k) & (cf_indices < t_j)
                    if np.any(cps_mask):
                        cps_idx = cf_indices[cps_mask]
                        cps_amt = cf_amounts[cps_mask]
                        # Discount from each coupon date to t_k: G[p, cf]/G[p, t_k]
                        pv_sel += (
                            G[sel][:, cps_idx] * cps_amt[np.newaxis, :]
                        ).sum(axis=1) / G_tk[sel]
                    # Call payout at t_j: (K + coup_t_j) * G[p, t_j] / G[p, t_k]
                    pv_sel += (K + coup_t_j) * (G[sel, t_j] / G_tk[sel])
                    C[sel] = pv_sel

            # ITM mask for the issuer: continuation liability > call price K.
            # Only these paths have a non-trivial exercise decision; OTM paths
            # keep their existing (defer / hold) policy.
            itm = C > K
            if np.sum(itm) < min_itm_samples:
                continue

            # Regress C on basis [1, r, r^2] using only ITM samples.
            r_tk = R[:, t_k]
            X_itm = np.column_stack(
                [np.ones(np.sum(itm)), r_tk[itm], r_tk[itm] ** 2]
            )
            y_itm = C[itm]
            coeffs, _, _, _ = np.linalg.lstsq(X_itm, y_itm, rcond=None)
            C_hat = X_itm @ coeffs

            # Exercise at t_k on paths where the fitted continuation exceeds K.
            # Going backwards, exercising earlier overrides any later policy.
            ex_now = np.zeros(num_paths, dtype=bool)
            ex_now[itm] = C_hat > K
            exercised_at[ex_now] = k

        # Final bondholder PV per path under the reconstructed optimal policy.
        pv = np.zeros(num_paths)

        never_mask = exercised_at == -1
        if np.any(never_mask):
            # PV = sum of all CFs after valuation_date, discounted to t0.
            pv[never_mask] = (
                G[never_mask][:, cf_indices] * cf_amounts[np.newaxis, :]
            ).sum(axis=1)

        exrc_mask = ~never_mask
        if np.any(exrc_mask):
            paths_e = np.where(exrc_mask)[0]
            k_vals = exercised_at[paths_e]
            for k in np.unique(k_vals):
                t_k = call_indices[k]
                coup_t_k = coupon_at_call[k]
                sel = paths_e[k_vals == k]
                pv_sel = np.zeros(len(sel))
                # Coupons strictly before t_k.
                cps_mask = (cf_indices > 0) & (cf_indices < t_k)
                if np.any(cps_mask):
                    cps_idx = cf_indices[cps_mask]
                    cps_amt = cf_amounts[cps_mask]
                    pv_sel += (
                        G[sel][:, cps_idx] * cps_amt[np.newaxis, :]
                    ).sum(axis=1)
                # Call payout at t_k: (K + coup_t_k) discounted from t_k to t0.
                pv_sel += (K + coup_t_k) * G[sel, t_k]
                pv[sel] = pv_sel

        callable_bond_value = float(np.mean(pv))
        call_probability = float(np.sum(exrc_mask) / num_paths)

        exrc_paths = np.where(exrc_mask)[0]
        exrc_date_list = [future_call_dates[exercised_at[p]] for p in exrc_paths]
        exrc_value_list = [
            K + coupon_at_call[exercised_at[p]] for p in exrc_paths
        ]

        additional_stats = {
            "call_probability": call_probability,
            "option_value": float(straight_bond_value - callable_bond_value),
            "callable_values_std": float(np.std(pv)),
            "call_statistics": {
                "call_frequency": call_probability,
                "call_dates": exrc_date_list,
                "call_values": exrc_value_list,
            },
        }

        return callable_bond_value, straight_bond_value, additional_stats


def solve_callable_bond_coupon(
    rate_simulator: InterestRateSimulator,
    straight_bond: Bond,
    straight_bond_ytm: float,
    call_protection_years: int = 1,
    tolerance: float = 1e-4,
    anchor_mode: str = "equivalent_rate",
    csv_path: str | None = None,
) -> tuple[float, dict]:
    """Solve for the coupon rate that makes callable bond value equal to 100.

    Before the optimisation the rate simulator is re-anchored for
    risk-neutral pricing. Two anchoring modes are supported:

      * "equivalent_rate": r0 and theta (mu) are both set to the flat
        rate reproducing the straight-bond PV at straight_bond_ytm
        (the flat-curve calibration).
      * "market_curve": r0 = SOFR + issuer Z-spread, theta = 10y
        Treasury par yield + Z-spread, where the Treasury zero curve is
        bootstrapped from the par yields in the merged market-data CSV
        and the spread is calibrated so the straight bond prices at 100.

    kappa/sigma are kept from the historical P-measure calibration in
    both modes.

    Args:
        rate_simulator: InterestRateSimulator instance for Monte Carlo simulation
        straight_bond: Straight bond object to construct callable bond from
        straight_bond_ytm: Fair ytm for straight bond
        call_protection_years: Years of call protection
        tolerance: Convergence tolerance
        anchor_mode: "equivalent_rate" or "market_curve"
        csv_path: Path to us_market_rates_2020_2025.csv (market_curve mode)

    Returns:
        Tuple of (optimal_coupon_rate, solution_info)
    """
    target_value = 100.0

    # Re-anchor the simulator from P-measure to Q-measure. See the
    # docstring above for the two supported anchoring schemes.
    if anchor_mode == "equivalent_rate":
        CallableBondValuer.anchor_to_risk_neutral_measure(
            rate_simulator=rate_simulator,
            straight_bond=straight_bond,
            straight_bond_ytm=straight_bond_ytm,
        )
    elif anchor_mode == "market_curve":
        CallableBondValuer.anchor_to_market_curve(
            rate_simulator=rate_simulator,
            straight_bond=straight_bond,
            csv_path=csv_path,
        )
    else:
        raise ValueError(
            f"Unknown anchor_mode '{anchor_mode}'. "
            "Use 'equivalent_rate' or 'market_curve'."
        )

    def objective_function(coupon_rate: float) -> float:
        """Objective function: difference between callable bond value and target."""

        # Create callable bond with the trial coupon rate
        callable_bond = CallableBond(
            issue_date=straight_bond.issue_dt,
            maturity_date=straight_bond.maturity_dt,
            coupon_rate=coupon_rate,
            call_protection_years=call_protection_years,
            freq_type=straight_bond.freq_type,
            day_count_type=straight_bond.accrual_dc_type,
        )

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
            day_count_type=straight_bond.accrual_dc_type,
        )

        valuer = CallableBondValuer(final_callable_bond, rate_simulator)
        final_value, straight_value, stats = valuer.value_bond(
            valuation_date=straight_bond.issue_dt,
            straight_bond_ytm=straight_bond_ytm,
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
        rate_simulator.generate_rates()
        rate_simulator.plot_rate_paths(max_paths=50)

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
    anchor_mode: str = "equivalent_rate",
    csv_path: str | None = None,
) -> None:
    """Analyze the relationship between coupon rates and callable bond values.

    anchor_mode/csv_path select the Q-measure anchoring, as documented in
    solve_callable_bond_coupon().
    """

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

    # Re-anchor the simulator from P-measure to Q-measure.
    print()
    if anchor_mode == "equivalent_rate":
        CallableBondValuer.anchor_to_risk_neutral_measure(
            rate_simulator=rate_simulator,
            straight_bond=straight_bond,
            straight_bond_ytm=straight_bond_ytm,
        )
    elif anchor_mode == "market_curve":
        CallableBondValuer.anchor_to_market_curve(
            rate_simulator=rate_simulator,
            straight_bond=straight_bond,
            csv_path=csv_path,
        )
    else:
        raise ValueError(
            f"Unknown anchor_mode '{anchor_mode}'. "
            "Use 'equivalent_rate' or 'market_curve'."
        )
    print()

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
            day_count_type=straight_bond.accrual_dc_type,
        )

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
    # Run the sample scenario
    print("=== Callable Bond Valuer Implementation Test ===\n")
    issue_date = Date(27, 5, 2025)
    maturity_date = Date(27, 5, 2035)  # 10 years
    days = int(maturity_date - issue_date) + 2
    days_of_year = 365

    straight_bond_coupon = 0.046  # 4.6%
    straight_bond_ytm = straight_bond_coupon  

    # Create a straight bond for testing
    straight_bond = Bond(
        issue_dt=issue_date,
        maturity_dt=maturity_date,
        coupon=straight_bond_coupon,
        freq_type=FrequencyTypes.ANNUAL,
        accrual_dc_type=DayCountTypes.THIRTY_E_360,
    )

    print("Problem Parameters:")
    print(f"Straight bond coupon: {straight_bond_coupon:.2%}")
    print(f"Target bond value: 100")
    print(f"Call protection: 1 year")
    print()

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "us_market_rates_2020_2025.csv",
    )

    # Short-rate model. The r0/mu/sigma/kappa values passed below are
    # PLACEHOLDERS; they go through THREE initialisation layers, in this order:
    #
    #   Layer 1 - placeholders here (all four are dummy values, needed only
    #             so the simulator can be constructed).
    #
    #   Layer 2 - P-measure historical calibration: load_and_calibrate_from_csv()
    #             runs OLS on the SOFR overnight-rate series (sofr_rate
    #             column of us_market_rates_2020_2025.csv) and overwrites
    #             kappa, sigma AND mu (mu is stored as `theta` in
    #             InterestRateSimulator). r0 is left at its placeholder.
    #             These are real-world (P) parameters.
    #
    #   Layer 3 - Q-measure re-anchoring: inside solve_callable_bond_coupon() /
    #             analyze_callable_bond_dynamics(), via
    #             CallableBondValuer.anchor_to_market_curve() with
    #             anchor_mode="market_curve":
    #               * The Treasury zero curve is bootstrapped from the
    #                 on-the-run par yields (tsy1y/3y/5y/7y/10y) at the
    #                 issue date.
    #               * The issuer Z-spread is calibrated so the straight
    #                 bond prices at 100 (its observed issue price).
    #               * r0 := SOFR + spread, theta (mu) := 10y Treasury par
    #                 yield + spread.
    #             The alternative anchor_mode="equivalent_rate" keeps the
    #             flat-curve anchoring (r0 & theta := the flat
    #             rate reproducing the straight-bond PV at its YTM).
    #
    #             κ (mean-reversion speed) and σ (volatility) are KEPT from
    #             Layer 2 under the MEASURE-INVARIANCE ASSUMPTION: under a
    #             Girsanov change of measure from P to Q, σ (the diffusion
    #             coefficient) does not change, and κ (the speed of pullback)
    #             is unaffected by the market price of risk — that premium is
    #             absorbed entirely into θ.
    #
    #             CAVEAT: σ and κ are P-measure estimates from the SOFR
    #             overnight time series, NOT market-implied Q-measure
    #             estimates. For production pricing they should ideally be
    #             calibrated to liquid swaption or bond-option prices (which
    #             directly reflect the Q measure). The historical estimates
    #             here are a practical fallback when implied data is
    #             unavailable.
    rate_simulator = InterestRateSimulator(
        r0=0.044951,              # placeholder -> Layer 3 overwrites with equivalent_rate
        mu=0.044951,              # placeholder -> Layer 2 (P-theta) -> Layer 3 (Q-theta)
        sigma=0.0117,             # placeholder -> Layer 2 (P-sigma, kept in Layer 3)
        days=days,
        days_of_year=days_of_year,
        num_paths=5000,
        model="vasicek",
        kappa=1.0,                # placeholder -> Layer 2 (P-kappa, kept in Layer 3)
    )

    print(
        f"Calibrating rate model to historical SOFR overnight rates "
        f"({rate_simulator.model})..."
    )
    rate_simulator.load_and_calibrate_from_csv(
        csv_path=csv_path,
        start_date=datetime.datetime(2020, 1, 1),
        end_date=datetime.datetime(2025, 5, 27),
        model=rate_simulator.model,
        rate_col="sofr_rate",  # column in the merged us_market_rates file
    )
    print(
        f"Calibrated: kappa={rate_simulator.kappa:.4f}, "
        f"theta(mu)={rate_simulator.mu:.4%}, sigma={rate_simulator.sigma:.4%}\n"
    )

    # Analyze callable bond dynamics first
    analyze_callable_bond_dynamics(
        rate_simulator=rate_simulator,
        straight_bond=straight_bond,
        straight_bond_ytm=straight_bond_ytm,
        call_protection_years=1,
        anchor_mode="market_curve",
        csv_path=csv_path,
    )

    # Solve for callable bond coupon using Monte Carlo optimization
    print("\n=== Solving for Callable Bond Coupon ===")

    optimal_coupon, solution_info = solve_callable_bond_coupon(
        rate_simulator=rate_simulator,
        straight_bond=straight_bond,
        straight_bond_ytm=straight_bond_ytm,
        call_protection_years=1,
        anchor_mode="market_curve",
        csv_path=csv_path,
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


if __name__ == "__main__":
    main()

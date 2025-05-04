import pandas as pd
import numpy as np
import datetime
import sqlite3
import bisect

from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine
from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Interval

from fxincome import const, logger, utils
from dataclasses import dataclass, field


@dataclass
class SymbolPosition:
    symbol: str  # Symbol = code + exchange. All positions must be >= 0
    position: float


@dataclass
class PositionCollection:
    positions: list[SymbolPosition] = field(
        default_factory=list
    )  # A list of SymbolPosition(symbol, position)

    def total_position(self) -> float:
        return sum(p.position for p in self.positions)

    def get_position(self, symbol: str) -> float:
        for p in self.positions:
            if p.symbol == symbol:
                return p.position
        return 0

    def add_bond(self, symbol: str, position: float):
        for p in self.positions:
            if p.symbol == symbol:
                p.position += position
                break
        else:
            self.positions.append(SymbolPosition(symbol, position))

    def substract_bond(self, symbol: str, position: float):
        for p in self.positions:
            if p.symbol == symbol:
                p.position -= position
                if p.position < 0:
                    raise ValueError(f"Position for {symbol} is negative: {p.position}")
                break

    # Check if the positions held are the same as the tenor positions.
    def same_as_tenor_positions(self, tenor_positions: list) -> bool:
        tenor_symbol_pos: list[SymbolPosition] = []
        # Unravel the tenor positions into a flat list of symbol positions.
        for tenor_pos_collection in tenor_positions:
            tenor_symbol_pos.extend(
                tenor_pos_collection.positions
            )  # Use extend to flatten the list

        return (
            all(pos in tenor_symbol_pos for pos in self.positions)
            and all(pos in self.positions for pos in tenor_symbol_pos)
            and len(self.positions) == len(tenor_symbol_pos)
        )


class IndexStrategy(StrategyTemplate):

    TOTAL_SIZE = 6e6  # Bond Size. 100 face value per unit size. 6 million size -> 600 million face value.
    CASH_AVAILABLE = 630e6  # If use more than available cash, you need to borrow cash.
    MAX_CASH = 630e6  # Maximum cash for backtrader. Order will be rejected if cash is insufficient.
    TOTAL_POS = 6  # Total units of positions.

    # Fixed parameters to be shown and set in the UI
    parameters = ["TOTAL_SIZE", "CASH_AVAILABLE", "MAX_CASH", "TOTAL_POS"]

    # Variables to be shown in the UI
    variables = [
        "low_percentile",  # Low percentile threshold of spread. Default 25th percentile(0.25)
        "high_percentile",  # High percentile threshold of spread. Default 75th percentile(0.75)
        "select_mode",  # Bond for buying selection mode. Default "match_ttm"
        "min_volume",  # Mininum trade volume of a bond to be selected. Default 1 billion
        "lookback_days",  # Period of historical data to be used for analysis. Default 3*250 trade days
    ]

    def __init__(
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict,
    ):
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)
        self.low_pctl = setting.get("low_percentile", 0.25)
        self.high_pctl = setting.get("high_percentile", 0.75)
        self.select_mode = setting.get("select_mode", "match_ttm")  # "max_ytm", "max_vol", "match_ttm"
        self.min_vol = setting.get("min_volume", 1e9)
        self.lookback_days = setting.get("lookback_days", 3 * 250)

        # A list of PositionCollection in [7yr, 5yr, 3yr]
        self.tenor_positions = [
            PositionCollection(),
            PositionCollection(),
            PositionCollection(),
        ]

        conn = sqlite3.connect(const.DB.SQLITE_CONN)

        # Get CDB Yield Curve history from sqlite DB.
        cdb_yc_table = const.DB.TABLES.IndexEnhancement.CDB_YC
        self.cdb_yc = pd.read_sql(f"SELECT * FROM [{cdb_yc_table}]", conn)
        # Convert date strings to date objects and set as index
        self.cdb_yc["date"] = pd.to_datetime(self.cdb_yc["date"]).dt.date
        self.cdb_yc.set_index("date", inplace=True)

        # Get CDB bond information from sqlite DB.
        cdb_info_table = const.DB.TABLES.IndexEnhancement.CDB_INFO
        self.cdb_info = pd.read_sql(f"SELECT * FROM [{cdb_info_table}]", conn)
        # Convert date strings to date objects
        self.cdb_info["issue_date"] = pd.to_datetime(
            self.cdb_info["issue_date"]
        ).dt.date
        self.cdb_info["maturity_date"] = pd.to_datetime(
            self.cdb_info["maturity_date"]
        ).dt.date
        self.cdb_info["symbol"] = self.cdb_info["wind_code"].str.split(
            ".", expand=True
        )[0]

        conn.close()

        # Store daily average TTMs: list of (date, avg_ttm)
        self.daily_avg_ttms: list[tuple[datetime.date, float]] = []
        # Store coupon payments received: list of (date, coupon)
        self.daily_coupons: list[tuple[datetime.date, float]] = []
        # Store daily positions: list of (date, PositionCollection)
        self.daily_positions: list[tuple[datetime.date, PositionCollection]] = []
        # Bars to skip in on_init()
        self._bars_to_skip = 1
        # Counter to skip initial bars loaded in on_init()
        self._bars_loaded_count = 0
        # Previous date of bar in on_bars()
        self._prev_date: datetime.date = None

    def _add_daily_coupon(self, today: datetime.date) -> float:
        """
        Find the positions held at the closest date strictly before 'today',
        and calculate coupon payments received 'today'. Assumes self.daily_positions
        is sorted chronologically by date.
        """
        if not self.daily_positions:
            return 0.0

        # Extract dates for searching
        dates = [item[0] for item in self.daily_positions]

        # Find the insertion point for 'today'. This is the index of the first date >= today.
        insertion_point = bisect.bisect_left(dates, today)

        # If insertion_point is 0, 'today' is before or equal to the first recorded date.
        # No previous date exists.
        if insertion_point == 0:
            return 0.0

        # The index of the closest date *before* 'today' is insertion_point - 1.
        prev_date_index = insertion_point - 1
        prev_date, prev_positions = self.daily_positions[prev_date_index]
        # Skip checking coupon for the previous date.
        # Positions held at previous date does not get payments on previous date.
        prev_date = prev_date + datetime.timedelta(days=1)

        total_coupon = 0.0 
        for bond_pos in prev_positions.positions:
            symbol = bond_pos.symbol.split(".")[0]  # Remove exchange from symbol
            # Get bond information from sqlite DB.
            cdb_info = self.cdb_info[self.cdb_info["symbol"] == symbol]
            if cdb_info.empty:
                raise ValueError(f"No bond information found for {symbol}")
            coupon_rate = cdb_info.iloc[0]["coupon_rate"]
            coupon_freq = cdb_info.iloc[0]["coupon_freq"]
            issue_date = cdb_info.iloc[0]["issue_date"]
            maturity_date = cdb_info.iloc[0]["maturity_date"]

            # Face value is 100. cal_coupon() returns coupon per 1 face value for the period.
            paid_coupon = (
                bond_pos.position
                * 100
                * utils.cal_coupon(
                    prev_date,
                    today,
                    issue_date,
                    maturity_date,
                    coupon_rate / 100.0, # Ensure float division
                    coupon_freq,
                )
            )
            total_coupon += paid_coupon # Accumulate coupon for this bond

        if total_coupon > 0:
            self.daily_coupons.append((today, total_coupon))

        return total_coupon

    def _select_bonds_for_buy(
        self, bars: dict[str, BarData], min_vol: float = 1e9, mode: str = "max_ytm"
    ) -> list[BarData]:
        """
        Select bond bars for buying in [7yr, 5yr, 3yr]. Each tenor has only one bond bar.

        Args:
            bars(dict): dict of bars to be selected from
            min_vol(float): minimum volume required for a bond to be considered
            mode(str): "max_ytm", "max_vol", "match_ttm"

        Returns:
            list[BarData]: 3 bars in [7yr, 5yr, 3yr]
        """
        bond_bars = [[], [], []]  # Bond bars candidates for [7yr, 5yr, 3yr]
        for bar in bars.values():
            # Filter out bonds with low volume.
            if bar.volume < min_vol:
                continue
            # Assign bars into 3 lists according to their ttm
            if bar.extra["matu"] >= 6 and bar.extra["matu"] <= 8:
                bond_bars[0].append(bar)
            elif bar.extra["matu"] >= 4 and bar.extra["matu"] < 6:
                bond_bars[1].append(bar)
            elif bar.extra["matu"] >= 2 and bar.extra["matu"] < 4:
                bond_bars[2].append(bar)

        if mode == "max_ytm":
            return [
                (
                    max(bond_bars[0], key=lambda x: x.extra["ytm"])
                    if bond_bars[0]
                    else None
                ),
                (
                    max(bond_bars[1], key=lambda x: x.extra["ytm"])
                    if bond_bars[1]
                    else None
                ),
                (
                    max(bond_bars[2], key=lambda x: x.extra["ytm"])
                    if bond_bars[2]
                    else None
                ),
            ]
        elif mode == "max_vol":
            return [
                max(bond_bars[0], key=lambda x: x.volume) if bond_bars[0] else None,
                max(bond_bars[1], key=lambda x: x.volume) if bond_bars[1] else None,
                max(bond_bars[2], key=lambda x: x.volume) if bond_bars[2] else None,
            ]
        # Find the bond with the closest ttm to the target ttm.
        elif mode == "match_ttm":
            return [
                (
                    min(
                        bond_bars[0],
                        key=lambda x: abs(x.extra["matu"] - 7),
                    )
                    if bond_bars[0]
                    else None
                ),
                (
                    min(
                        bond_bars[1],
                        key=lambda x: abs(x.extra["matu"] - 5),
                    )
                    if bond_bars[1]
                    else None
                ),
                (
                    min(
                        bond_bars[2],
                        key=lambda x: abs(x.extra["matu"] - 3),
                    )
                    if bond_bars[2]
                    else None
                ),
            ]
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Choose from 'max_ytm', 'max_vol', 'match_ttm'."
            )

    def log_overall_avg_ttm(self) -> None:
        # Calculate overall time-weighted average TTM
        if not self.daily_avg_ttms:
            logger.info("No daily TTM data recorded to calculate overall average.")
            return

        total_weighted_ttm_sum = 0.0
        total_duration = 0

        # Iterate up to the second to last element
        for i in range(len(self.daily_avg_ttms) - 1):
            date_i, ttm_i = self.daily_avg_ttms[i]
            date_next, _ = self.daily_avg_ttms[i + 1]
            # Duration is the number of calendar days the position was held
            duration = (date_next - date_i).days
            if duration <= 0:  # Should not happen in a chronological backtest
                logger.warning(
                    f"Non-positive duration {duration} found between {date_i} and {date_next}. Skipping."
                )
                continue
            total_weighted_ttm_sum += ttm_i * duration
            total_duration += duration

        # Add the last day's contribution, assuming it's held for 1 day
        _, ttm_last = self.daily_avg_ttms[-1]
        duration_last = 1
        total_weighted_ttm_sum += ttm_last * duration_last
        total_duration += duration_last

        if total_duration > 0:
            overall_average_ttm = total_weighted_ttm_sum / total_duration
            logger.info(
                f"Overall Time-Weighted Average Portfolio TTM: {overall_average_ttm:.2f} years"
            )
        else:
            logger.info(
                "Could not calculate overall average TTM (total duration is zero)."
            )

    def _calculate_avg_ttm(self, bars: dict[str, BarData]) -> float:
        """Calculate the weighted average ttm of the current portfolio."""
        total_value = 0.0
        weighted_ttm_sum = 0.0

        for symbol_pos in self._get_positions().positions:
            symbol = symbol_pos.symbol
            if symbol in bars:
                bar = bars[symbol]
                position_size = symbol_pos.position
                # Use close price for value weighting
                value = position_size * bar.close_price
                total_value += value
                weighted_ttm_sum += value * bar.extra["matu"]

        if total_value > 0:
            return weighted_ttm_sum / total_value
        else:
            return 0.0  # Return 0 if portfolio is empty or has no ttm data

    def _get_positions(self) -> PositionCollection:
        positions = PositionCollection()
        for vt_symbol in self.vt_symbols:
            position = self.get_pos(vt_symbol)
            if position:
                positions.add_bond(vt_symbol, position)
        return positions

    def _log_key_dates_positions(
        self,
        today: datetime.date,
        target_positions: list[float],
        delta_sizes: list[float],
        avg_ttm: float,
    ) -> None:
        logger.info("----Daily Positions----")
        logger.info(f"today: {today}")
        positions = self._get_positions()
        for position in positions.positions:  # Iterate over the list attribute
            logger.info(f"position of {position.symbol}: {position.position}")
        logger.info(f"Capital: {self.strategy_engine.capital}")
        logger.info(
            f"current_positions: [{self.tenor_positions[0].total_position()}, {self.tenor_positions[1].total_position()}, {self.tenor_positions[2].total_position()}]"
        )
        logger.info(f"target_positions: {target_positions}")
        logger.info(f"delta_sizes: {delta_sizes}")
        logger.info(f"Average Portfolio TTM: {avg_ttm:.2f} years")
        logger.info("----End of Daily Positions----")

    def on_init(self) -> None:
        logger.info("Initializing backtest...")
        # The first a few bars are used for initialization and not for trading.
        self.load_bars(days=self._bars_to_skip, interval=Interval.DAILY)

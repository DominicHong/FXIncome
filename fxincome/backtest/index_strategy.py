import pandas as pd
import datetime
import sqlite3
import bisect
from enum import Enum
from typing import Optional

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


class ThresholdConfig:
    """Configuration for thresholds with built-in types and validation."""
    
    class Type(Enum):
        """Types of thresholds supported by the strategy."""
        AVG_PCTL = "avg_pctl"  # Percentile of normalization by 'average' method
        ZSCORE = "zscore"      # Z-score thresholds
        ORI_PCTL = "ori_pctl"  # Original percentile

        def __eq__(self, other):
            # Check if 'other' is an Enum instance first.
            if not isinstance(other, Enum):
                return NotImplemented
            
            # Compare based on the class name and the member's value.
            # This handles cases where enum instances might be from different
            # module loads but are conceptually the same.
            if self.__class__.__name__ == other.__class__.__name__:
                return self.value == other.value
            
            return NotImplemented

    def __init__(
        self,
        threshold_type: Type,
        low: float,
        high: float,
        extreme_low: Optional[float] = None,
        extreme_high: Optional[float] = None,
    ):
        """
        Initialize threshold configuration.

        Args:
            threshold_type (Type enum): Type of threshold 
            low: Low threshold value
            high: High threshold value
            extreme_low: Optional extreme low threshold value
            extreme_high: Optional extreme high threshold value
        """
        self.type = threshold_type
        self.low = low
        self.high = high
        self.extreme_low = extreme_low
        self.extreme_high = extreme_high

        self._validate()

    def _validate(self):
        """Validate threshold values."""
        if self.low >= self.high:
            raise ValueError(f"low threshold ({self.low}) must be less than high threshold ({self.high})")
        
        if self.extreme_low is not None and self.extreme_low >= self.low:
            raise ValueError(f"extreme_low threshold ({self.extreme_low}) must be less than low threshold ({self.low})")
        
        if self.extreme_high is not None and self.extreme_high <= self.high:
            raise ValueError(f"extreme_high threshold ({self.extreme_high}) must be greater than high threshold ({self.high})")

    @classmethod
    def avg_pctl(cls) -> 'ThresholdConfig':
        """Create a default average percentile threshold configuration."""
        return cls(
            threshold_type=cls.Type.AVG_PCTL,
            low=0.25,
            high=0.75,
            extreme_low=0.10,
            extreme_high=0.90
        )

    @classmethod
    def zscore(cls) -> 'ThresholdConfig':
        """Create a default z-score threshold configuration."""
        return cls(
            threshold_type=cls.Type.ZSCORE,
            low=-0.67,  # Approximately 25th percentile
            high=0.67,  # Approximately 75th percentile
            extreme_low=-1.28,  # Approximately 10th percentile
            extreme_high=1.28   # Approximately 90th percentile
        )

    @classmethod
    def ori_pctl(cls) -> 'ThresholdConfig':
        """Create a default original percentile threshold configuration."""
        return cls(
            threshold_type=cls.Type.ORI_PCTL,
            low=0.25,
            high=0.75,
            extreme_low=0.10,
            extreme_high=0.90
        )

    @classmethod
    def from_settings(cls, settings: dict) -> 'ThresholdConfig':
        """
        Create a threshold configuration from settings dictionary.
        
        Args:
            settings: Dictionary containing threshold settings
                     Must include 'threshold_type' and optionally threshold values
        """
        threshold_type = settings.get("threshold_type", cls.Type.AVG_PCTL)

        if threshold_type == cls.Type.AVG_PCTL:
            config = cls.avg_pctl()
        elif threshold_type == cls.Type.ZSCORE:
            config = cls.zscore()
        elif threshold_type == cls.Type.ORI_PCTL:
            config = cls.ori_pctl()
        else:
            raise ValueError(
                f"Invalid threshold_type : '{threshold_type}'. "
            )

        # Override default values if provided in settings
        if "low_threshold" in settings:
            config.low = settings["low_threshold"]
        if "high_threshold" in settings:
            config.high = settings["high_threshold"]
        if "extreme_low_threshold" in settings:
            config.extreme_low = settings["extreme_low_threshold"]
        if "extreme_high_threshold" in settings:
            config.extreme_high = settings["extreme_high_threshold"]

        # Revalidate after potential overrides
        config._validate()
        return config



class IndexStrategy(StrategyTemplate):

    TOTAL_SIZE = 6e6  # Bond Size. 100 face value per unit size. 6 million size -> 600 million face value.
    CASH_AVAILABLE = 630e6  # If use more than available cash, you need to borrow cash.
    MAX_CASH = 630e6  # Maximum cash for backtrader. Order will be rejected if cash is insufficient.
    TOTAL_POS = 6  # Total units of positions.

    # Fixed parameters to be shown and set in the UI
    parameters = ["TOTAL_SIZE", "CASH_AVAILABLE", "MAX_CASH", "TOTAL_POS"]

    # Variables to be shown in the UI
    variables = [
        "threshold_type",  # Type of thresholds to use (avg_pctl, zscore, ori_pctl)
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
        
        # Initialize threshold configuration from settings
        # example settings = {
        #     "threshold_type": ThresholdConfig.Type.AVG_PCTL,
        #     "low_threshold": 0.25,
        #     "high_threshold": 0.75,
        #     "extreme_low_threshold": 0.10,
        #     "extreme_high_threshold": 0.90
        # }
        self.threshold = ThresholdConfig.from_settings(setting)
        
        # Select bonds to buy. Mode: "max_ytm", "max_vol", "match_ttm"
        self.select_mode = setting.get("select_mode", "match_ttm")
        self.min_vol = setting.get("min_volume", 1e9)
        self.lookback_days = setting.get("lookback_days", 3 * 250)

        # A list of PositionCollection in [7yr, 5yr, 3yr]
        self.tenor_positions = [
            PositionCollection(),
            PositionCollection(),
            PositionCollection(),
        ]

        # Attributes that might be used by subclasses or moved methods
        self.expert_mode = False  # Default to normal mode
        self.key_dates = []  # Default to no key dates for base strategy

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
                    coupon_rate / 100.0,  # Ensure float division
                    coupon_freq,
                )
            )
            total_coupon += paid_coupon  # Accumulate coupon for this bond

        self.record_cashflow(total_coupon, "coupon")

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
            if bar.extra["ttm"] >= 6 and bar.extra["ttm"] <= 8:
                bond_bars[0].append(bar)
            elif bar.extra["ttm"] >= 4 and bar.extra["ttm"] < 6:
                bond_bars[1].append(bar)
            elif bar.extra["ttm"] >= 2 and bar.extra["ttm"] < 4:
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
                        key=lambda x: abs(x.extra["ttm"] - 7),
                    )
                    if bond_bars[0]
                    else None
                ),
                (
                    min(
                        bond_bars[1],
                        key=lambda x: abs(x.extra["ttm"] - 5),
                    )
                    if bond_bars[1]
                    else None
                ),
                (
                    min(
                        bond_bars[2],
                        key=lambda x: abs(x.extra["ttm"] - 3),
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
                weighted_ttm_sum += value * bar.extra["ttm"]

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

    def _transform_positions(self, positions: list[float]) -> list[float]:
        """
        Transform positions to ensure they are non-negative and sum to TOTAL_POS.
        """
        # Ensure all positions are non-negative
        positions = [max(0, p) for p in positions]

        # Transf positions so that total position is TOTAL_POS.
        # If total position exceeds TOTAL_POS(normally 6 units), scale proportionally
        total = sum(positions)
        if total > self.TOTAL_POS:
            positions = [p * self.TOTAL_POS / total for p in positions]

        # Round positions to 1 decimal place
        positions = [round(p, 1) for p in positions]

        # Ensure total equals exactly TOTAL_POS (6.0) with a single adjustment
        total = sum(positions)
        if total != self.TOTAL_POS:
            diff = self.TOTAL_POS - total
            if diff > 0:
                # Find the smallest position and add the remaining amount
                min_idx = positions.index(min(positions))
                positions[min_idx] = round(positions[min_idx] + diff, 1)
            else:
                # Find the largest position and subtract the excess
                max_idx = positions.index(max(positions))
                positions[max_idx] = round(
                    positions[max_idx] + diff, 1
                )  # diff is negative

        if sum(positions) != self.TOTAL_POS:
            raise ValueError(
                f"Total position sum is not equal to TOTAL_POS: {sum(positions)} != {self.TOTAL_POS}"
            )
        return positions


    def __calculate_position(
        self, spread_pctl: float, positions: list[float], x_idx: int, y_idx: int
    ) -> list[float]:
        """
        Calculate positions based on spread percentile for "normal mode".

        Args:
            spread_pctl (float): Percentile of the spread. Spread = longer bond yield - shorter bond yield.
                                longer bond position is at x_idx, shorter bond position is at y_idx.
            positions (list): Current positions in [7yr, 5yr, 3yr]
            x_idx (int): Position index of longer-term bond
            y_idx (int): Position index of shorter-term bond

        Returns:
            list (float): Calculated positions in [7yr, 5yr, 3yr]
        """
        if spread_pctl <= self.threshold.low:
            # Low spread, shift to shorter position.
            positions[y_idx] += 1
            positions[x_idx] -= 1
        elif spread_pctl >= self.threshold.high:
            # High spread, shift to longer position.
            positions[x_idx] += 1
            positions[y_idx] -= 1

        # Transform positions to ensure they are non-negative and sum to TOTAL_POS.
        positions = self._transform_positions(positions)
        return positions

    def _generate_target_positions(self, today: datetime.date) -> list[float]:
        """
        Calculate the final positions based on all three spreads for "normal mode".

        Args:
            today (datetime.date): Today's date

        Returns:
            list (float): Final positions [7yr, 5yr, 3yr] in real size
        """

        # Get 5yr-3yr, 7yr-5yr, 7yr-3yr average spread percentiles for today
        try:
            spread_53 = self.cdb_yc.loc[today, "pctl_avg_53"]
            spread_75 = self.cdb_yc.loc[today, "pctl_avg_75"]
            spread_73 = self.cdb_yc.loc[today, "pctl_avg_73"]
        except KeyError:
            raise ValueError(f"No spread data found for {today}")

        # Initial positions: [7yr, 5yr, 3yr] 2 units each
        positions = [self.TOTAL_POS / 3, self.TOTAL_POS / 3, self.TOTAL_POS / 3]

        # Apply adjustments in sequence
        # For 5yr-3yr spread: x_idx=1 (5yr), y_idx=2 (3yr)
        positions = self.__calculate_position(spread_53, positions, 1, 2)

        # For 7yr-5yr spread: x_idx=0 (7yr), y_idx=1 (5yr)
        positions = self.__calculate_position(spread_75, positions, 0, 1)

        # For 7yr-3yr spread: x_idx=0 (7yr), y_idx=2 (3yr)
        positions = self.__calculate_position(spread_73, positions, 0, 2)

        positions = [p * self.TOTAL_SIZE / self.TOTAL_POS for p in positions]

        return positions

    def on_bars(self, bars: dict[str, BarData]) -> None:
        # Increment the counter for bars received
        self._bars_loaded_count += 1

        # Skip the initial bars loaded during on_init
        if self._bars_loaded_count <= self._bars_to_skip:
            skipped_date = self.strategy_engine.datetime.date()
            logger.info(f"Skip initial bar for date: {skipped_date}")
            return

        # Check if the positions held are the same as the tenor positions.
        if not self._get_positions().same_as_tenor_positions(self.tenor_positions):
            raise ValueError("Positions held are not the same as the tenor positions.")

        today = self.strategy_engine.datetime.date()

        self.daily_positions.append((today, self._get_positions()))

        # Calculate coupon payments from previous date to today and add to self.daily_coupons.
        # You get payments only if you hold positions on the previous date.
        self._add_daily_coupon(today)

        # Calculate average ttm before adjustments for today
        avg_ttm = self._calculate_avg_ttm(bars)
        # Record the daily average TTM
        self.daily_avg_ttms.append((today, avg_ttm))

        # Get bonds with sufficient volume(self.min_vol) for trading
        sufficient_volume_bars = {
            symbol: bar for symbol, bar in bars.items() if bar.volume >= self.min_vol
        }
        # if no bond with sufficient volume, positions unchanged
        if not sufficient_volume_bars:
            logger.info(
                f"No bond with sufficient trading volume found at {today}. Positions unchanged."
            )
            return

        # Select bonds for 3 positions in [7yr, 5yr, 3yr]
        bond_bars_for_buy = self._select_bonds_for_buy(
            sufficient_volume_bars, min_vol=self.min_vol, mode=self.select_mode
        )

        # Calculate positions based on spread.
        # IndexExtremeStrategy will override _generate_target_positions for its specific logic (e.g. expert_mode).
        # The base IndexStrategy._generate_target_positions implements "normal mode".
        target_positions = self._generate_target_positions(today)

        # Calculate size changes for each position
        delta_sizes = [
            target_positions[i] - self.tenor_positions[i].total_position()
            for i in range(3)
        ]

        if (
            today in self.key_dates
        ):  # self.key_dates will be empty for IndexStrategy unless set otherwise
            self._log_key_dates_positions(today, target_positions, delta_sizes, avg_ttm)

        # Execute trades
        for i in range(3):
            # Buy
            if delta_sizes[i] > 0:
                # No bond bar candidate for buying, skip
                if not bond_bars_for_buy[i]:
                    continue

                # Buy the selected bond
                symbol = (
                    bond_bars_for_buy[i].symbol
                    + "."
                    + bond_bars_for_buy[i].exchange.value
                )

                # Buy at 5% above the close price. Normally it will be filled on T+1's open.
                buy_price = bond_bars_for_buy[i].close_price * 1.05

                self.buy(
                    vt_symbol=symbol,
                    price=buy_price,
                    volume=delta_sizes[i],
                )
                self.tenor_positions[i].add_bond(
                    symbol=symbol,
                    position=delta_sizes[i],
                )
                if today in self.key_dates:
                    logger.info(
                        f"Tenor[{i}] Buy {symbol} at {buy_price:} with volume {delta_sizes[i]:.0f}"
                    )

            # Sell randomlly from each tenor until the target position is reached.
            elif delta_sizes[i] < 0:
                remaining_to_sell = abs(delta_sizes[i])
                # Get all positions for this tenor
                tenor_positions = self.tenor_positions[i]

                # Try to sell from each position until we've sold enough
                for pos in tenor_positions.positions[
                    :
                ]:  # Make a copy to safely modify while iterating

                    if remaining_to_sell <= 0:
                        break

                    bond_symbol = pos.symbol
                    # The bond should have enough trading volume for selling.
                    bond_bar_for_sale = sufficient_volume_bars.get(bond_symbol, None)
                    if not bond_bar_for_sale:
                        logger.warning(
                            f"Bond {bond_symbol} has insufficient trading volume on {today}. Skipping sell for this bond."
                        )
                        continue

                    # Calculate how much we can sell from this position
                    amount_to_sell = min(remaining_to_sell, pos.position)

                    # Execute the sell order
                    # Sell at 5% below the close price. Normally it will be filled on T+1's open.
                    sell_price = bond_bar_for_sale.close_price * 0.95
                    self.sell(
                        vt_symbol=bond_symbol,
                        price=sell_price,
                        volume=amount_to_sell,
                    )

                    if today in self.key_dates:
                        logger.info(
                            f"Tenor[{i}] Sell {bond_symbol} at {sell_price} with volume {amount_to_sell:.0f}"
                        )

                    # Update the position
                    tenor_positions.substract_bond(bond_symbol, amount_to_sell)

                    remaining_to_sell -= amount_to_sell

                    # If position is now 0, remove it from the list
                    if pos.position == 0:
                        tenor_positions.positions.remove(pos)

        self._prev_date = today

    def on_init(self) -> None:
        logger.info("Initializing backtest...")
        # The first a few bars are used for initialization and not for trading.
        self.load_bars(days=self._bars_to_skip, interval=Interval.DAILY)

    def update_trade(self, trade: TradeData) -> None:
        """
        Overriding update_trade to log trade details
        """
        super().update_trade(trade)
        if (
            trade.datetime.date() in self.key_dates
        ):  # self.key_dates will be empty for IndexStrategy unless set otherwise
            logger.info(
                f"Trade {trade.direction}: {trade.symbol}, datetime: {trade.datetime.date()}, price: {trade.price}, volume: {trade.volume:.0f}"
            )

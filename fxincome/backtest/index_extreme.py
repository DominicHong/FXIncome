from datetime import datetime
from vnpy_portfoliostrategy import StrategyEngine
from vnpy.trader.object import BarData, TradeData
from vnpy.trader.database import DB_TZ

from fxincome.backtest.index_strategy import IndexStrategy
from fxincome import const, logger


class IndexExtremeStrategy(IndexStrategy):
    """
    An index enhancement strategy that implements possible extreme positions.
    This strategy supports both "extreme percentiles mode" and "expert mode".
    Positions are adjusted twice:
    1. based on spread percentiles, then
    2. based on either extreme percentiles or expert signals.
        2.1 In "extreme percentiles mode", positions are adjusted based on another set of spread percentiles.
        2.2 In "expert mode", positions are adjusted based on expert rate signals.
    """

    def __init__(
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict,
    ):
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)
        self.expert_mode = setting.get("expert_mode", False)


    def __calculate_position(
        self,
        spread_pctl: float,
        positions: list[float],
        x_idx: int,
        y_idx: int,
        expert_signal: int | None = None,
    ) -> list[float]:
        """
        Calculate positions based on spread percentile and optional expert signal.

        Args:
            spread_pctl (float): Percentile of the spread. Spread = longer bond yield - shorter bond yield.
                                longer bond position is at x_idx, shorter bond position is at y_idx.
            positions (list): Current positions in [7yr, 5yr, 3yr]
            x_idx (int): Position index of longer-term bond
            y_idx (int): Position index of shorter-term bond
            expert_signal (int, optional): Expert signal, 0 for rates down, 1 for rates up

        Returns:
            list (float): Calculated positions in [7yr, 5yr, 3yr]
        """
        # Check expert_signal is valid
        if self.expert_mode:
            if expert_signal is None:
                raise ValueError(
                    "Expert signal is required when expert mode is enabled"
                )
            elif expert_signal not in [0, 1]:
                raise ValueError("Expert signal must be 0 or 1")

        # If using expert mode, use expert signal to adjust positions
        if self.expert_mode:
            if spread_pctl <= self.threshold.low and expert_signal == 1:
                # Low spread + Rates up expectation, shift to extremely shorter position.
                positions[y_idx] += 2
                positions[x_idx] -= 2
            elif spread_pctl <= self.threshold.low and expert_signal == 0:
                # Low spread + Rates down expectation, shift to shorter position.
                positions[y_idx] += 1
                positions[x_idx] -= 1
            elif spread_pctl >= self.threshold.high and expert_signal == 0:
                # High spread + Rates down expectation, shift to extremely longer position.
                positions[x_idx] += 2
                positions[y_idx] -= 2
            elif spread_pctl >= self.threshold.high and expert_signal == 1:
                # High spread + Rates up expectation, shift to longer position.
                positions[x_idx] += 1
                positions[y_idx] -= 1
        # If not using expert mode, use four-tier percentile thresholds
        else:
            if self.threshold.extreme_high is None or self.threshold.extreme_low is None:
                raise ValueError("Extreme thresholds are not set")
            if spread_pctl <= self.threshold.extreme_low:
                positions[y_idx] += 2
                positions[x_idx] -= 2
            elif self.threshold.extreme_low < spread_pctl <= self.threshold.low:
                positions[y_idx] += 1
                positions[x_idx] -= 1
            elif self.threshold.high <= spread_pctl < self.threshold.extreme_high:
                positions[x_idx] += 1
                positions[y_idx] -= 1
            elif self.threshold.extreme_high <= spread_pctl:
                positions[x_idx] += 2
                positions[y_idx] -= 2

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
        current_expert_signal = None  # Default to None
        if self.expert_mode:
            # Placeholder: Logic to determine expert_signal for the current date based on self.strategy_engine.datetime.date()
            # For example: current_expert_signal = self._get_signal_for_date(self.strategy_engine.datetime.date())
            # If expert_mode is True, and no signal is found (current_expert_signal remains None),
            # _calculate_position will raise a ValueError as per its existing logic.
            # This maintains the original behavior where a signal is strictly required in expert mode.
            pass  # This pass means current_expert_signal remains None if expert_mode is True and no explicit signal logic is added here.

        
        # Get 5yr-3yr, 7yr-5yr, 7yr-3yr average spread percentiles for today
        try:
            spread_53 = self.cdb_yc.loc[today, "pctl_avg_53"]
            spread_75 = self.cdb_yc.loc[today, "pctl_avg_75"]
            spread_73 = self.cdb_yc.loc[today, "pctl_avg_73"]
        except KeyError:
            raise ValueError(f"No spread data found for {today}")
        
        # Initial positions: [7yr, 5yr, 3yr] 2 units each
        positions = [self.TOTAL_POS / 3, self.TOTAL_POS / 3, self.TOTAL_POS / 3]

        # Apply adjustments in sequence, passing the (potentially None) current_expert_signal
        # For 5yr-3yr spread: x_idx=1 (5yr), y_idx=2 (3yr)
        positions = self.__calculate_position(
            spread_53, positions, 1, 2, current_expert_signal
        )

        # For 7yr-5yr spread: x_idx=0 (7yr), y_idx=1 (5yr)
        positions = self.__calculate_position(
            spread_75, positions, 0, 1, current_expert_signal
        )

        # For 7yr-3yr spread: x_idx=0 (7yr), y_idx=2 (3yr)
        positions = self.__calculate_position(
            spread_73, positions, 0, 2, current_expert_signal
        )

        positions = [p * self.TOTAL_SIZE / self.TOTAL_POS for p in positions]

        return positions

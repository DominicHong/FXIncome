from datetime import datetime
from vnpy_portfoliostrategy import StrategyEngine
from vnpy.trader.object import BarData, TradeData
from vnpy.trader.database import DB_TZ

from fxincome.backtest.index_strategy import IndexStrategy, SymbolPosition, PositionCollection
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
        # Additional percentile thresholds for more granular control
        self.extreme_low = setting.get("extreme_low_percentile", 0.10)
        self.extreme_high = setting.get("extreme_high_percentile", 0.90)

        # Key dates for detailed logging
        self.key_dates = [
            datetime(2024, 1, 2).date(),
            datetime(2024, 1, 3).date(),
            datetime(2024, 1, 4).date(),
            datetime(2024, 6, 11).date(),
        ]


    def _calculate_position(
        self, spread_pctl, positions, x_idx, y_idx, expert_signal=None
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
            if spread_pctl <= self.low_pctl and expert_signal == 1:
                # Low spread + Rates up expectation, shift to extremely shorter position.
                positions[y_idx] += 2
                positions[x_idx] -= 2
            elif spread_pctl <= self.low_pctl and expert_signal == 0:
                # Low spread + Rates down expectation, shift to shorter position.
                positions[y_idx] += 1
                positions[x_idx] -= 1
            elif spread_pctl >= self.high_pctl and expert_signal == 0:
                # High spread + Rates down expectation, shift to extremely longer position.
                positions[x_idx] += 2
                positions[y_idx] -= 2
            elif spread_pctl >= self.high_pctl and expert_signal == 1:
                # High spread + Rates up expectation, shift to longer position.
                positions[x_idx] += 1
                positions[y_idx] -= 1
        # If not using expert mode, use four-tier percentile thresholds
        else:
            if spread_pctl <= self.extreme_low:
                positions[y_idx] += 2
                positions[x_idx] -= 2
            elif self.extreme_low < spread_pctl <= self.low_pctl:
                positions[y_idx] += 1
                positions[x_idx] -= 1
            elif self.high_pctl <= spread_pctl < self.extreme_high:
                positions[x_idx] += 1
                positions[y_idx] -= 1
            elif self.extreme_high <= spread_pctl:
                positions[x_idx] += 2
                positions[y_idx] -= 2

        # Ensure all positions are non-negative
        positions = [max(0, p) for p in positions]

        # Adjust positions so that total position is TOTAL_POS.
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

    def _generate_target_positions(
        self, spread_53, spread_75, spread_73, expert_signal=None
    ) -> list[float]:
        """
        Calculate the final positions based on all three spreads.

        Args:
            spread_53 (float): 5yr-3yr spread percentile
            spread_75 (float): 7yr-5yr spread percentile
            spread_73 (float): 7yr-3yr spread percentile
            expert_signal (int, optional): Expert signal, 0 for rates down, 1 for rates up

        Returns:
            list (float): Final positions [7yr, 5yr, 3yr] in real size
        """
        # Initial positions: [7yr, 5yr, 3yr] 2 units each
        positions = [self.TOTAL_POS / 3, self.TOTAL_POS / 3, self.TOTAL_POS / 3]

        # Apply adjustments in sequence
        # For 5yr-3yr spread: x_idx=1 (5yr), y_idx=2 (3yr)
        positions = self._calculate_position(spread_53, positions, 1, 2, expert_signal)

        # For 7yr-5yr spread: x_idx=0 (7yr), y_idx=1 (5yr)
        positions = self._calculate_position(spread_75, positions, 0, 1, expert_signal)

        # For 7yr-3yr spread: x_idx=0 (7yr), y_idx=2 (3yr)
        positions = self._calculate_position(spread_73, positions, 0, 2, expert_signal)

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

        # Get 'pctl_avg_53', 'pctl_avg_75', 'pctl_avg_73' for today
        try:
            pctl_avg_53 = self.cdb_yc.loc[today, "pctl_avg_53"]
            pctl_avg_75 = self.cdb_yc.loc[today, "pctl_avg_75"]
            pctl_avg_73 = self.cdb_yc.loc[today, "pctl_avg_73"]
        except KeyError:
            logger.info(f"No spread data found for {today}. Skipping.")
            return

        # Calculate positions based on average spread percentiles
        if self.expert_mode:
            pass
        else:
            target_positions = self._generate_target_positions(
                pctl_avg_53, pctl_avg_75, pctl_avg_73, expert_signal=None
            )

        # Calculate size changes for each position
        delta_sizes = [
            target_positions[i] - self.tenor_positions[i].total_position()
            for i in range(3)
        ]

        if today in self.key_dates:
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

            # Sell randomlly from each tenor until the target is reached.
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
                    bond_bar = sufficient_volume_bars.get(bond_symbol, None)
                    # Skip if the bond doesn't have sufficient volume
                    if not bond_bar:
                        continue

                    # Calculate how much we can sell from this position
                    amount_to_sell = min(remaining_to_sell, pos.position)

                    # Execute the sell order
                    # Sell at 5% below the close price. Normally it will be filled on T+1's open.
                    sell_price = bond_bar.close_price * 0.95
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

    def update_trade(self, trade: TradeData) -> None:
        """
        Overriding update_trade to log trade details
        """
        super().update_trade(trade)
        if trade.datetime.date() in self.key_dates:
            logger.info(
                f"Trade {trade.direction}: {trade.symbol}, datetime: {trade.datetime.date()}, price: {trade.price}, volume: {trade.volume:.0f}"
        )

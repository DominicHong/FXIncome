from vnpy_portfoliostrategy import StrategyEngine
from vnpy.trader.object import BarData
from fxincome.backtest.index_strategy import IndexStrategy


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
        self.current_positions = [0, 0, 0]  # Bond positions in [7yr, 5yr, 3yr]

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
            list (float): Final positions [7yr, 5yr, 3yr]
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

        return positions


    def on_init(self) -> None:
        self.write_log("Initializing backtest...")
        self.load_bars(1)

    def on_start(self) -> None:
        self.write_log("Starting backtest...")

    def on_stop(self) -> None:
        self.write_log("Stopping backtest...")

    def on_bars(self, bars: dict[str, BarData]) -> None:
        # Cancel all pending orders
        self.cancel_all()
        bar = bars.get("010214.CFETS", None)
        if bar:
            print(
                f"Symbol: {bar.symbol}, Date: {bar.datetime}, YTM: {bar.extra['ytm']}, Matu: {bar.extra['matu']}, Out Bal: {bar.extra['out_bal']}"
            )
        bar = bars.get("010216.CFETS", None)
        if bar:
            print(
                f"Symbol: {bar.symbol}, Date: {bar.datetime}, YTM: {bar.extra['ytm']}, Matu: {bar.extra['matu']}, Out Bal: {bar.extra['out_bal']}"
            )


    def prenext(self):
        # Record positions in result dataframe
        sum_position = 0
        for code in self.p.code_list:
            if len(self.getdatabyname(code)) != 0:
                self.result.loc[
                    self.result["DATE"] == self.data.datetime.date(0), code
                ] = self.getposition(self.getdatabyname(code)).size
                sum_position += self.getposition(self.getdatabyname(code)).size
        self.result.loc[
            self.result["DATE"] == self.data.datetime.date(0), "sum_position"
        ] = sum_position

        # Last day processing
        if self.data.datetime.date(0) == self.last_day:
            self.last_day_process()

        # Process coupon payments
        for code in self.p.code_list:
            if len(self.getdatabyname(code)) == 0:
                continue
            if self.getdatabyname(code).datetime.date(0) == self.last_day:
                continue
            if self.getposition(self.getdatabyname(code)).size > 0:
                # Add coupon payment (if any) to broker
                self.add_coupon(code)

        # Get current spread percentiles
        spread_57 = self.yield_curve_data.loc[
            self.yield_curve_data["DATE"] == self.data.datetime.date(0),
            "7年-5年国开/均值",
        ].values[0]

        spread_53 = self.yield_curve_data.loc[
            self.yield_curve_data["DATE"] == self.data.datetime.date(0),
            "5年-3年国开/均值",
        ].values[0]

        spread_73 = self.yield_curve_data.loc[
            self.yield_curve_data["DATE"] == self.data.datetime.date(0),
            "7年-3年国开/均值",
        ].values[0]

        # Calculate target positions
        target_positions = self._generate_target_positions(
            spread_53, spread_57, spread_73
        )

        # Scale positions to match SIZE
        target_positions = [pos * self.SIZE / 6 for pos in target_positions]

        # Calculate position adjustments needed
        delta_size = [target_positions[i] - self.init_position[i] for i in range(3)]

        # Skip if no changes needed
        if delta_size == [0, 0, 0]:
            self.log("No position changes needed")
            self.record()
            return
        else:
            self.numbers_tradays += 1

        # Get current yield data for bond selection
        yield_3 = self.yield_curve_data.loc[
            self.yield_curve_data["DATE"] == self.data.datetime.date(0), "3年国开"
        ].values[0]
        yield_5 = self.yield_curve_data.loc[
            self.yield_curve_data["DATE"] == self.data.datetime.date(0), "5年国开"
        ].values[0]
        yield_7 = self.yield_curve_data.loc[
            self.yield_curve_data["DATE"] == self.data.datetime.date(0), "7年国开"
        ].values[0]

        # Select bonds by maturity and yield
        years_3_bond = [
            code
            for code in self.p.code_list
            if len(self.getdatabyname(code)) != 0
            and abs(self.getdatabyname(code).ytm[0] - yield_3) < 0.05
            and self.getdatabyname(code).matu[0] > 2
            and self.getdatabyname(code).matu[0] < 3.5
            and self.getdatabyname(code).volume[0] > self.p.min_volume
        ]

        years_5_bond = [
            code
            for code in self.p.code_list
            if len(self.getdatabyname(code)) != 0
            and abs(self.getdatabyname(code).ytm[0] - yield_5) < 0.05
            and self.getdatabyname(code).matu[0] > 4
            and self.getdatabyname(code).matu[0] < 5.5
            and self.getdatabyname(code).volume[0] > self.p.min_volume
        ]

        years_7_bond = [
            code
            for code in self.p.code_list
            if len(self.getdatabyname(code)) != 0
            and abs(self.getdatabyname(code).ytm[0] - yield_7) < 0.05
            and self.getdatabyname(code).matu[0] > 6
            and self.getdatabyname(code).matu[0] < 7.5
            and self.getdatabyname(code).volume[0] > self.p.min_volume
        ]

        # Check if we can execute the needed trades
        feasibility_judge_buy = [0, 0, 0]
        feasibility_judge_sell = [3, 5, 7]
        sell_list = []

        # Check 3-year bond trading feasibility
        if delta_size[0] > 0:
            feasibility_judge_sell[0] = 0
            if len(years_3_bond) == 0:
                feasibility_judge_buy[0] = 3
        elif delta_size[0] < 0:
            for code in self.code_list_3:
                if self.getdatabyname(code).volume[0] >= self.p.min_volume:
                    sell_list.append(code)
                    feasibility_judge_sell[0] = 0
                    break
        else:
            feasibility_judge_sell[0] = 0

        # Check 5-year bond trading feasibility
        if delta_size[1] > 0:
            feasibility_judge_sell[1] = 0
            if len(years_5_bond) == 0:
                feasibility_judge_buy[1] = 5
        elif delta_size[1] < 0:
            for code in self.code_list_5:
                if self.getdatabyname(code).volume[0] >= self.p.min_volume:
                    sell_list.append(code)
                    feasibility_judge_sell[1] = 0
                    break
        else:
            feasibility_judge_sell[1] = 0

        # Check 7-year bond trading feasibility
        if delta_size[2] > 0:
            feasibility_judge_sell[2] = 0
            if len(years_7_bond) == 0:
                feasibility_judge_buy[2] = 7
        elif delta_size[2] < 0:
            for code in self.code_list_7:
                if self.getdatabyname(code).volume[0] >= self.p.min_volume:
                    sell_list.append(code)
                    feasibility_judge_sell[2] = 0
                    break
        else:
            feasibility_judge_sell[2] = 0

        # Skip if trades not feasible
        if feasibility_judge_sell != [0, 0, 0] or feasibility_judge_buy != [0, 0, 0]:
            self.log(
                f"Trading not feasible: buy_status={feasibility_judge_buy}, sell_status={feasibility_judge_sell}"
            )
            self.record()
            self.numbers_tradays -= 1
            return

        # Execute trades using maximum yield strategy
        self.strategy_max_yield(
            delta_size,
            target_positions,
            sell_list,
            years_3_bond,
            years_5_bond,
            years_7_bond,
        )

        # Update current positions after trading
        self.init_position = target_positions

        # Record results
        self.record()

    def sell_bond(self, amount_to_sell, sell_list, code_list):
        """
        Sell bonds from a specific maturity bucket.

        Args:
            amount_to_sell (float): Amount to sell
            sell_list (list): List of bonds to consider selling
            code_list (list): List of codes for this maturity bucket

        Returns:
            list: Updated code_list after selling
        """
        code_all = code_list.copy()
        if amount_to_sell < self.getposition(self.getdatabyname(sell_list[0])).size:
            self.sell(data=self.getdatabyname(sell_list[0]), size=amount_to_sell)
        elif amount_to_sell == self.getposition(self.getdatabyname(sell_list[0])).size:
            self.sell(data=self.getdatabyname(sell_list[0]), size=amount_to_sell)
            code_list.remove(sell_list[0])
        else:
            for bond in code_all:
                bond_size = self.getposition(self.getdatabyname(bond)).size
                if bond_size <= amount_to_sell:
                    self.sell(data=self.getdatabyname(bond), size=bond_size)
                    amount_to_sell -= bond_size
                    code_list.remove(bond)
                else:
                    self.sell(data=self.getdatabyname(bond), size=amount_to_sell)
                    amount_to_sell = 0
                if amount_to_sell == 0:
                    break
        sell_list.remove(sell_list[0])
        return code_list, sell_list

    def strategy_max_yield(
        self, delta_size, each_size, sell_list, years_3_bond, years_5_bond, years_7_bond
    ):
        """
        Execute trades using maximum yield strategy.

        Args:
            delta_size (list): Size changes needed
            each_size (list): Target size for each maturity
            sell_list (list): Bonds to consider selling
            years_3_bond (list): 3-year bonds to consider buying
            years_5_bond (list): 5-year bonds to consider buying
            years_7_bond (list): 7-year bonds to consider buying
        """
        if delta_size[0] > 0:
            # Buy 3-year bond with highest yield
            max_yield_3_bond = max(
                years_3_bond, key=lambda bond: self.getdatabyname(bond).ytm[0]
            )
            self.buy(data=self.getdatabyname(max_yield_3_bond), size=delta_size[0])
            self.init_position[0] = each_size[0]
            if max_yield_3_bond not in self.code_list_3:
                self.code_list_3.append(max_yield_3_bond)
        elif delta_size[0] < 0:
            self.code_list_3, sell_list = self.sell_bond(
                abs(delta_size[0]), sell_list, self.code_list_3
            )
            self.init_position[0] = each_size[0]
        if delta_size[1] > 0:
            # Buy 5-year bond with highest yield
            max_yield_5_bond = max(
                years_5_bond, key=lambda bond: self.getdatabyname(bond).ytm[0]
            )
            self.buy(data=self.getdatabyname(max_yield_5_bond), size=delta_size[1])
            self.init_position[1] = each_size[1]
            if max_yield_5_bond not in self.code_list_5:
                self.code_list_5.append(max_yield_5_bond)
        elif delta_size[1] < 0:
            self.code_list_5, sell_list = self.sell_bond(
                abs(delta_size[1]), sell_list, self.code_list_5
            )
            self.init_position[1] = each_size[1]
        if delta_size[2] > 0:
            # Buy 7-year bond with highest yield
            max_yield_7_bond = max(
                years_7_bond, key=lambda bond: self.getdatabyname(bond).ytm[0]
            )
            self.buy(data=self.getdatabyname(max_yield_7_bond), size=delta_size[2])
            self.init_position[2] = each_size[2]
            if max_yield_7_bond not in self.code_list_7:
                self.code_list_7.append(max_yield_7_bond)
        elif delta_size[2] < 0:
            self.code_list_7, sell_list = self.sell_bond(
                abs(delta_size[2]), sell_list, self.code_list_7
            )
            self.init_position[2] = each_size[2]

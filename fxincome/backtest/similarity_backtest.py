# -*- coding: utf-8; py-indent-offset:4 -*-
import os
import math
import datetime
import backtrader as bt
import pandas as pd
from backtrader.order import OrderBase
from analyzers.kelly import Kelly
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)
from dataclasses import dataclass
from fxincome import logger, handler, const
from sqlalchemy import create_engine


class ETFData(bt.feeds.PandasData):
    lines = ("turnover", "gc001")

    # 左边为lines里的名字，右边为dataframe column的名字
    params = (
        ("datetime", "date"),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),  # 成交额，单位是元
        ("openinterest", -1),
        ("turnover", "turnover"),  # 换手率，单位是%
        ("gc001", "GC001_close"),  # GC001 回购利率收盘价
    )


@dataclass
class Trader:
    judgement_day: datetime.date
    cash: float
    bond_commission_rate: float
    repo_commission_rate: float
    position: int = 0
    buy_all_confidence: float = -0.005  # unit %. -0.005 -> -0.005%. prediction weight < this confidence, buy all
    sell_all_confidence: float = 0.005  # unit %. 0.005 -> 0.005%. prediction weight > this confidence, sell all

    def buy(self, price: float, sizer: str, pred_w: float) -> int:
        """
        Calculate the size to buy with all cash, accounting for commission, and update cash and position.
        Args:
            price(float): The price to buy
            sizer(str): "all" or "weighted". "all" means all in. "weighted" means buy based on prediction weights.
            pred_w(float): The prediction weight.
        Returns:
            size(int): The size to buy, greater than or equal to 0
        """
        max_size = int(self.cash / (price * (1 + self.bond_commission_rate)))
        if sizer == "all":
            size = max_size
        elif sizer == "weighted":
            if self.buy_all_confidence <= pred_w <= 0:
                size = int(0.5 * max_size)
            elif pred_w < self.buy_all_confidence:
                size = max_size
            else:
                size = 0
        else:
            raise ValueError(f"Invalid sizer: {sizer}")

        total_cost = size * price * (1 + self.bond_commission_rate)
        self.cash = self.cash - total_cost
        self.position = self.position + size
        return size

    def sell(self, price: float, sizer: str, pred_w: float) -> int:
        """
        Calculate the cash received from selling all positions, accounting for commission, and update cash and position.
        Args:
            price(float): The price to sell
            sizer(str): "all" or "weighted". "all" means all out. "weighted" means sell based on prediction weights.
            pred_w(float): The prediction weight.
        Returns:
            size(int): The size to sell, greater than or equal to 0
        """
        max_size = self.position
        if sizer == "all":
            size = max_size
        elif sizer == "weighted":
            if 0 < pred_w <= self.sell_all_confidence:
                size = int(0.5 * max_size)
            elif pred_w > self.sell_all_confidence:
                size = max_size
            else:
                size = 0
        else:
            raise ValueError(f"Invalid sizer: {sizer}")

        cash_received = size * price * (1 - self.bond_commission_rate)
        self.cash = self.cash + cash_received
        self.position = self.position - size
        return size

    def reverse_repo(
        self, rate: float, start_date: datetime.date, end_date: datetime.date
    ) -> float:
        """
        Use all remaining cash to do reverse repo. Calculate the interest received from reverse repo, and update cash.
        Args:
            rate(float): The annualized interest rate. 0.01 for 1%.
            start_date(datetime.date): The start date of reverse repo
            end_date(datetime.date): The end date of reverse repo
        Returns:
            interest(float): The interest received(after paying commission) from reverse repo.
        """
        days = (end_date - start_date).days
        interest = self.cash * rate * days / 365
        commission = self.cash * self.repo_commission_rate
        self.cash = self.cash + interest - commission
        return interest - commission


class NTraderStrategy(bt.Strategy):
    """
    Cash are divided into n parts, and each part is used to trade on a different day based on predictions.
    Normally num_traders <= pred_days.
    Parmas can be accessed by self.p.xxx
        num_traders: Number of traders
        pred_days: YTM direction = ytm(t + pred_days) - ytm(t)
        sizer: "all" or "weighted".
               "all" means all in or all out. "weighted" means buy/sell based on prediction weights.
        repo_commission: Commission rate for reverse repo. 0.01 for 1%.
        pred_df: DataFrame with columns ["date", "pred_5", "pred_10", ..., "pred_{n}"]. n = pred_days
    """

    params = dict(
        num_traders=1,
        pred_days=10,
        sizer="all",  # All in or all out
        repo_commission=0.001 / 100,
        pred_df=None,
    )

    etf_name = "511260.SH"  # 国泰上证10年期国债ETF

    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        dt = dt or self.etf_data.datetime.date(0)
        print(f"{dt:%Y%m%d} - {txt}")

    def total_value(self):
        cash = sum([trader.cash for trader in self.traders])
        position = sum([trader.position for trader in self.traders])
        return cash + position * self.etf_data.close[0]

    def get_traders_cash(self):
        return sum([trader.cash for trader in self.traders])

    def get_traders_position(self):
        return sum([trader.position for trader in self.traders])

    def __init__(self):
        self.etf_data = self.getdatabyname(self.etf_name)

        # None means no pending order
        self.order = None

        total_days = self.etf_data.buflen() - 1

        self.bond_commission_rate = self.broker.getcommissioninfo(
            self.etf_data
        ).p.commission
        self.traders = [
            Trader(
                # The first n judgement days are the first n days of backtest data.
                judgement_day=self.etf_data.datetime.date(-total_days + n),
                cash=self.broker.get_cash() / self.p.num_traders,
                position=0,
                bond_commission_rate=self.bond_commission_rate,
                repo_commission_rate=self.p.repo_commission,
            )
            for n in range(self.p.num_traders)
        ]

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"Order {order.ref}, BUY, {order.executed.price:.2f}, {order.executed.size:.2f}, Cash after: {self.broker.get_cash():.2f}"
                )
            elif order.issell():
                self.log(
                    f"Order {order.ref}, SELL, {order.executed.price:.2f}, {order.executed.size:.2f}, Cash after: {self.broker.get_cash():.2f}"
                )
        elif order.status in [
            order.Canceled,
            order.Margin,
            order.Rejected,
            order.Expired,
        ]:
            self.log(
                f"Order {order.ref} NOT COMPLETED: {OrderBase.Status[order.status]}. {'BUY' if order.isbuy() else 'SELL'}, Size: {order.size:.2f}"
            )

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(
                f"Trade Profit, Gross:{trade.pnl:.4f}, Net:{trade.pnlcomm:.4f}, Commission:{trade.commission:.4f}"
            )

    def next(self):
        today = self.etf_data.datetime.date(0)
        # Do nothing on the final day.
        if len(self.etf_data) == self.etf_data.buflen():
            return

        for trader in self.traders:
            interest = trader.reverse_repo(
                rate=self.etf_data.gc001[0] / 100,
                start_date=today,
                end_date=self.etf_data.datetime.date(1),
            )
            self.broker.add_cash(interest)
            # Only on the judgement_day we decide whether to trade.
            if trader.judgement_day != today:
                continue
            # Update the judgement day to pred_days later
            if len(self.etf_data) + self.p.pred_days <= self.etf_data.buflen():
                trader.judgement_day = self.etf_data.datetime.date(self.p.pred_days)
            else:
                continue

            # Trade based on the predictions
            preds = self.p.pred_df[self.p.pred_df["date"].dt.date == today]

            if len(preds) == 0:  # Do nothing if no prediction
                logger.info(f"No prediction")
                continue
            else:
                # prediction of change of ytm after next pred_days trade days.
                pred_weight = preds[f"pred_weight_{self.p.pred_days}"].iat[0]
            if pred_weight <= 0:  # ytm down, buy
                buy_size = trader.buy(
                    price=self.etf_data.open[1], sizer=self.p.sizer, pred_w=pred_weight
                )
                self.order = self.buy(
                    data=self.etf_data,
                    size=buy_size,  # buy at t+1's open price
                    price=None,  # buy at t+1's open price
                    exectype=bt.Order.Market,  # buy at t+1's open price
                    valid=None,
                )
            elif pred_weight > 0:  # ytm up, sell
                sell_size = trader.sell(
                    price=self.etf_data.open[1], sizer=self.p.sizer, pred_w=pred_weight
                )
                self.order = self.sell(
                    data=self.etf_data,
                    size=sell_size,  # sell at t+1's open price
                    price=None,  # sell at t+1's open price
                    exectype=bt.Order.Market,  # sell at t+1's open price
                    valid=None,
                )
            else:
                logger.info(f"The prediction weight is NOT a number")
                continue


class TradeEverydayStrategy(bt.Strategy):
    """
    Trade every day based on rule of taking profit(TP) and stopping loss(SL), and then based on the predictions.
    If holding position, check position's pnl every day. Sell all if pnl(%)  > tp_pct or pnl(%) < sl_pct.
    If condition above is not satisfied, check the prediction:
        Buy all if predicted ytm is down. Sell all if predicted ytm is up.
    Parmas can be accessed by self.p.xxx
        pred_days: YTM direction = ytm(t + pred_days) - ytm(t)
        tp_pct: Take profit percentage = (current price - position's average cost) / position's average cost
        sl_pct: Stop loss percentage = (current price - position's average cost) / position's average cost
        repo_commission: Commission rate for reverse repo. 0.01 for 1%.
        pred_df: DataFrame with columns ["date", "pred_5", "pred_10", ..., "pred_{n}"]. n = pred_days
    """

    params = dict(
        pred_days=10,
        tp_pct=0.0015,  # ytm change of 1bp = value change of 8.8bp for a 10-year bond
        sl_pct=-0.0008,  # ytm change of 1bp = value change of 8.8bp for a 10-year bond
        repo_commission=0.001 / 100,  # Commission rate for reverse repo. 0.01 for 1%
        pred_df=None,
    )

    etf_name = "511260.SH"  # 国泰上证10年期国债ETF

    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        dt = dt or self.etf_data.datetime.date(0)
        print(f"{dt:%Y%m%d} - {txt}")

    def __init__(self):
        self.etf_data = self.getdatabyname(self.etf_name)

        # None means no pending order
        self.order = None

        self.total_days = self.etf_data.buflen() - 1

        self.bond_commission_rate = self.broker.getcommissioninfo(self.etf_data).p.commission

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"Order {order.ref}, BUY, {order.executed.price:.2f}, {order.executed.size:.2f}, Cash after: {self.broker.get_cash():.2f}"
                )
            elif order.issell():
                self.log(
                    f"Order {order.ref}, SELL, {order.executed.price:.2f}, {order.executed.size:.2f}, Cash after: {self.broker.get_cash():.2f}"
                )
        elif order.status in [
            order.Canceled,
            order.Margin,
            order.Rejected,
            order.Expired,
        ]:
            self.log(
                f"Order {order.ref} NOT COMPLETED: {OrderBase.Status[order.status]}. {'BUY' if order.isbuy() else 'SELL'}, Size: {order.size:.2f}"
            )

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(
                f"Trade Profit, Gross:{trade.pnl:.4f}, Net:{trade.pnlcomm:.4f}, Commission:{trade.commission:.4f}"
            )

    def next(self):
        today = self.etf_data.datetime.date(0)
        # Do nothing on the final day.
        if len(self.etf_data) == self.etf_data.buflen():
            return

        rate = self.etf_data.gc001[0] / 100
        days = (self.etf_data.datetime.date(1) - today).days
        cash = self.broker.get_cash()
        interest = cash * rate * days / 365
        commission = cash * self.p.repo_commission
        self.broker.add_cash(interest - commission)

        # Check position's pnl every day and decide wether to take profit or stop loss.
        # Sell all if pnl(%)  > tp_pct or pnl(%) < sl_pct.
        position = self.broker.getposition(self.etf_data)
        if position.size > 0:
            pnl_pct = (self.etf_data.close[0] - position.price) / position.price
            if pnl_pct > self.p.tp_pct or pnl_pct < self.p.sl_pct:
                self.order = self.sell(
                    data=self.etf_data,
                    size=position.size,
                    price=None,  # sell at t+1's open price
                    exectype=bt.Order.Market,  # sell at t+1's open price
                    valid=None,
                )
                return
        # Trade based on the predictions
        preds = self.p.pred_df[self.p.pred_df["date"].dt.date == today]

        if len(preds) == 0:  # Do nothing if no prediction
            logger.info(f"No prediction")
            return
        else:
            # prediction of change of ytm after next pred_days trade days.
            pred_weight = preds[f"pred_weight_{self.p.pred_days}"].iat[0]
        if pred_weight <= 0:  # ytm down, buy all
            buy_size = int(
                self.broker.get_cash()
                / (self.etf_data.open[1] * (1 + self.bond_commission_rate))
            )
            self.order = self.buy(
                data=self.etf_data,
                size=buy_size,  # buy at t+1's open price
                price=None,  # buy at t+1's open price
                exectype=bt.Order.Market,  # buy at t+1's open price
                valid=None,
            )
        elif pred_weight > 0:  # ytm up, sell all
            sell_size = position.size
            self.order = self.sell(
                data=self.etf_data,
                size=sell_size,  # sell at t+1's open price
                price=None,  # sell at t+1's open price
                exectype=bt.Order.Market,  # sell at t+1's open price
                valid=None,
            )
        else:
            logger.info(f"{today} - prediction weight is NOT a number")
            return


def run_backtest(
    strat,  # NTraderStrategy or TradeEverydayStrategy
    start_date: datetime.date,
    end_date: datetime.date,
    num_traders: int,
    pred_days: int,
    sizer: str = "all",
    tp_pct: float = 0.0015,  # ytm change of 1bp = value change of 8.8bp for a 10-year bond
    sl_pct: float = -0.0008,  # ytm change of 1bp = value change of 8.8bp for a 10-year bond
    repo_commission: float = 0.001 / 100,  # Commission rate for reverse repo. 0.01 -> 1%
    bond_commission: float = 0.0002,  # commission for bond trade. 0.0002 -> 0.02%
):
    bond_pred, etf_price = read_predictions_prices(start_date, end_date)

    # Create a cerebro entity
    cerebro = bt.Cerebro(tradehistory=True)

    # Add data to Cerebro
    data1 = ETFData(dataname=etf_price, nocase=True)
    cerebro.adddata(data1, name=NTraderStrategy.etf_name)

    cerebro.broker.set_checksubmit(False)
    cerebro.broker.set_fundmode(False, 1)
    cerebro.broker.set_cash(10000)
    cerebro.broker.setcommission(commission=bond_commission)

    # Add Strategies
    if strat is NTraderStrategy:
        cerebro.addstrategy(
            strategy=strat,
            num_traders=num_traders,
            pred_days=pred_days,
            sizer=sizer,
            repo_commission=repo_commission,
            pred_df=bond_pred,
        )
    if strat is TradeEverydayStrategy:
        cerebro.addstrategy(
            strategy=TradeEverydayStrategy,
            pred_days=pred_days,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            repo_commission=repo_commission,
            pred_df=bond_pred,
        )

    #  Add analyzers
    cerebro.addanalyzer(bt.analyzers.TimeReturn, fund=True, _name="TimeReturn")
    cerebro.addanalyzer(Kelly, _name="Kelly")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        timeframe=bt.TimeFrame.Weeks,
        riskfreerate=0.02,  # annual rate
        fund=False,
        annualize=True,
        _name="SharpRatio",
    )
    cerebro.addanalyzer(bt.analyzers.SQN, _name="SQN")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="DrawDown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="TradeReport")
    #  Add observers
    cerebro.addobserver(bt.observers.FundValue)
    cerebro.addobserver(bt.observers.FundShares)
    # Run over everything
    strategies = cerebro.run()
    print_backtest_result(cerebro, data1, strategies[0])


def print_backtest_result(cerebro, data1, strategy):
    # Plot the result
    # cerebro.plot(style="bar")
    broker = cerebro.broker
    print(strategy)
    if isinstance(strategy, NTraderStrategy):
        print(f"Broker Cash: {broker.get_cash():.2f}")
        print(f"Tradres Cash: {strategy.get_traders_cash():.2f}")
        print(cerebro.broker.getposition(data1))
        print(f"Traders Position: {strategy.get_traders_position():.2f}")
        print(f"Traders Total Value: {strategy.total_value():.2f}")
        print(f"Broker Value: {broker.get_value():.2f}")
        print(f"Broker Fund Value: {broker.get_fundvalue():.6f}")
        print(f"Broker Fund Shares: {broker.get_fundshares():.2f}")
    else:
        print(f"Broker Cash: {broker.get_cash():.2f}")
        print(cerebro.broker.getposition(data1))
        print(f"Broker Value: {broker.get_value():.2f}")
        print(f"Broker Fund Value: {broker.get_fundvalue():.6f}")
        print(f"Broker Fund Shares: {broker.get_fundshares():.2f}")

    sharp_ratio = strategy.analyzers.SharpRatio
    sqn = strategy.analyzers.SQN
    draw_down = strategy.analyzers.DrawDown
    kelly = strategy.analyzers.Kelly
    trade_report = strategy.analyzers.TradeReport
    sharp_ratio.print()
    # sqn.print()
    # draw_down.print()
    # kelly.print()
    # trade_report.print()


def read_predictions_prices(
    start_date: datetime.date, end_date: datetime.date
) -> tuple:
    # predictions of Ytm direction
    bond_pred = pd.read_csv(
        os.path.join(const.PATH.STRATEGY_POOL, const.HistorySimilarity.PREDICT_FILE),
        parse_dates=["date"],
    )
    bond_pred["date"] = bond_pred["date"].dt.date
    # etf_price = pd.read_csv(
    #     os.path.join(const.PATH.STRATEGY_POOL, "511260.sh.csv"),
    #     parse_dates=["date"],
    # )
    connection_string = (f'mysql+mysqlconnector://{const.MYSQL_CONFIG.USER}:{const.MYSQL_CONFIG.PASSWORD}'
                         f'@{const.MYSQL_CONFIG.HOST}/{const.MYSQL_CONFIG.DATABASE}')
    engine=create_engine(connection_string)
    etf_query = f"SELECT * FROM {const.PATH.STRATEGY_POOL.replace(const.PATH.MAIN, '').replace('/', '')}_511260sh"
    etf_price = pd.read_sql(etf_query, engine)
    engine.dispose()


    etf_price["date"] = etf_price["date"].dt.date
    # Filter prices between start_date and end_date
    etf_price = etf_price[
        (etf_price["date"] >= start_date) & (etf_price["date"] <= end_date)
    ]
    numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
    for col in numeric_cols:
        etf_price[col] = pd.to_numeric(etf_price[col])

    # Backtrader's datafeed requires datetime type
    bond_pred["date"] = pd.to_datetime(bond_pred["date"])
    etf_price["date"] = pd.to_datetime(etf_price["date"])

    return bond_pred, etf_price


def analyze_prediction(
    start_date: datetime.date, end_date: datetime.date, pred_days: int
):
    bond_pred, etf_price = read_predictions_prices(start_date, end_date)

    tbond_df = pd.read_csv(
        os.path.join(const.PATH.STRATEGY_POOL, "history_processed.csv"),
        parse_dates=["date"],
    )
    tbond_df = tbond_df[["date", "t_10y"]]
    tbond_df["date"] = tbond_df["date"].dt.date
    etf_price = pd.merge(etf_price, tbond_df, on="date", how="left")
    etf_price[f"y_fwd_{pred_days}"] = (
        etf_price["t_10y"].shift(-pred_days) - etf_price["t_10y"]
    )
    etf_price[f"y_actual_{pred_days}"] = etf_price[f"y_fwd_{pred_days}"].apply(
        lambda x: 1 if x > 0 else 0
    )

    bond_pred[f"actual_{pred_days}"] = bond_pred[f"yield_chg_fwd_{pred_days}"].apply(
        lambda x: 1 if x > 0 else 0
    )

    etf_price[f"price_chg_{pred_days}"] = (
        etf_price["close"].shift(-pred_days) - etf_price["close"]
    )
    etf_price[f"actual_price_{pred_days}"] = etf_price[f"price_chg_{pred_days}"].apply(
        lambda x: 1 if x > 0 else 0
    )

    merged_df = pd.merge(etf_price, bond_pred, on="date", how="left")
    merged_df = merged_df.dropna()

    bond_pred_values = merged_df[f"pred_{pred_days}"]
    bond_actual_values = merged_df[f"actual_{pred_days}"]
    etf_actual_values = merged_df[f"actual_price_{pred_days}"]
    y_actual_values = merged_df[f"y_actual_{pred_days}"]

    bond_pred_accuracy = accuracy_score(bond_actual_values, bond_pred_values)
    bond_pred_etf_accuracy = accuracy_score(etf_actual_values, 1 - bond_pred_values)
    bond_actual_etf_accuracy = accuracy_score(etf_actual_values, 1 - bond_actual_values)
    yield_actual_etf_accuracy = accuracy_score(etf_actual_values, 1 - y_actual_values)

    logger.info(f"bond_pred_accuracy: {bond_pred_accuracy:.4f}")
    logger.info(f"bond_pred_etf_accuracy: {bond_pred_etf_accuracy:.4f}")
    logger.info(f"bond_actual_etf_accuracy: {bond_actual_etf_accuracy:.4f}")
    logger.info(f"yield_actual_etf_accuracy: {yield_actual_etf_accuracy:.4f}")
    print(confusion_matrix(etf_actual_values, 1 - bond_actual_values))
    print(confusion_matrix(etf_actual_values, 1 - y_actual_values))

    merged_df.to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, "bond_etf_predictions.csv"), index=False
    )


def main():
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 10, 18)
    run_backtest(
        strat=TradeEverydayStrategy,
        start_date=start_date,
        end_date=end_date,
        num_traders=1,
        pred_days=5,
        sizer="all",
        tp_pct=0.0015,
        sl_pct=-0.0008,
        repo_commission=0.001 / 100,
        bond_commission=0.0002,
    )
    # analyze_prediction(start_date, end_date, pred_days=5)


if __name__ == "__main__":
    main()

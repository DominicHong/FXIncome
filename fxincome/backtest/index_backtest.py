from datetime import datetime
from numpy import nan
from vnpy_portfoliostrategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ, get_database

from index_strategy import IndexStrategy, ThresholdConfig
from index_extreme import IndexExtremeStrategy
from fxincome import logger, const

import pandas as pd
import os


def run_backtesting(
    strategy_class,
    strat_setting,
    interval,
    start,
    end,
    capital,
    risk_free,
    annual_days,
    vt_symbols: list[str] | None = None,
    rates: dict | None = None,
    slippages: dict | None = None,
    sizes: dict | None = None,
    priceticks: dict | None = None,
):

    engine = BacktestingEngine()

    # Discover symbols from database if not provided
    if vt_symbols is None:
        logger.info("vt_symbols not provided, querying database...")
        database = get_database()
        overview = database.get_bar_overview()

        # Filter for daily CFETS data and create vt_symbols list
        discovered_symbols = [
            f"{o.symbol}.{o.exchange.value}"
            for o in overview
            if o.exchange == Exchange.CFETS and o.interval == Interval.DAILY
        ]

        if not discovered_symbols:
            raise ValueError(
                "No daily CFETS symbols found in database. Run index_process_data.load_cdb_ohlc first."
            )

        vt_symbols = discovered_symbols

        # Generate default parameters if not provided
        # Total turnover = volume * size * price.
        sizes = {symbol: 1 for symbol in vt_symbols} if sizes is None else sizes
        # Rates are commission rates in percentage.
        rates = {symbol: 1 / 1e5 for symbol in vt_symbols} if rates is None else rates
        # Price ticks are the minimum price change in dollars for the asset.
        priceticks = (
            {symbol: 1 / 10000 for symbol in vt_symbols}
            if priceticks is None
            else priceticks
        )
        # If the market price is 102, the traded price for buy is 102 + slippage.
        slippages = (
            {symbol: 0.01 for symbol in vt_symbols} if slippages is None else slippages
        )

    # Ensure parameters are provided if symbols were passed directly
    elif not all([rates, slippages, sizes, priceticks]):
        raise ValueError(
            "rates, slippages, sizes, and priceticks must be provided if vt_symbols is specified."
        )

    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=interval,
        start=start,
        end=end,
        rates=rates,
        slippages=slippages,
        sizes=sizes,
        priceticks=priceticks,
        capital=capital,
        risk_free=risk_free,
        annual_days=annual_days,
    )
    engine.add_strategy(strategy_class, strat_setting)
    engine.load_data()
    engine.run_backtesting()
    engine.strategy.on_stop()
    df = engine.calculate_result()
    return engine, df


def run_multiple_backtesting(start_year: int, end_year: int):
    # Define columns to match the uploaded image
    columns = [
        "Period", "Spread_Type", "Decision_Mode", "Total_Return", "Annual_Return", "Average_TTM", "Total_Turnover(0.1B)", "Daily_Turnover(0.1B)",
        "Sharpe", "Max_Drawdown", "Total_Trade_Count", "Total_Days", "Profit:Loss_Days_Ratio"
    ]
    results = []

    for year in range(start_year, end_year + 1):
        start_date = datetime(year, 1, 1, tzinfo=DB_TZ)
        end_date = datetime(year, 12, 31, tzinfo=DB_TZ)
        threshold_types = [ThresholdConfig.Type.AVG_PCTL, ThresholdConfig.Type.ZSCORE]
        for tp in threshold_types:
            if tp == ThresholdConfig.Type.AVG_PCTL:
                low_threshold = 0.25
                high_threshold = 0.75
                extreme_low_threshold = 0.10
                extreme_high_threshold = 0.90
                spread_type_str = "AVG_PCTL"
            else:
                low_threshold = -0.67
                high_threshold = 0.67
                extreme_low_threshold = -1.28
                extreme_high_threshold = 1.28
                spread_type_str = "ZSCORE"
            strat_setting = {
                "select_mode": "max_ytm",
                "threshold_type": tp,  
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "expert_mode": False,
                "extreme_low_threshold": extreme_low_threshold,
                "extreme_high_threshold": extreme_high_threshold,
            }
            # Normal mode
            engine, df = run_backtesting(
                strategy_class=IndexStrategy,
                strat_setting=strat_setting,
                interval=Interval.DAILY,
                start=start_date,
                end=end_date,
                capital=IndexStrategy.CASH_AVAILABLE,
                risk_free=0.02,
                annual_days=240,
            )
            stats = engine.calculate_statistics(df, output=False)
            # Calculate average TTM if available
            avg_ttm = engine.strategy.cal_overall_avg_ttm()
            # Profit:Loss days ratio
            profit_days = stats.get("profit_days", nan)
            loss_days = stats.get("loss_days", nan)
            total_days = stats.get("total_days", nan)
            profit_loss_ratio = f"{profit_days}:{loss_days}" if loss_days > 0 else f"{profit_days}:0"
            # Append row
            results.append([
                str(year),
                spread_type_str,
                "Normal",
                f"{stats.get('total_return', nan):.2f}%",
                f"{stats.get('annual_return', nan):.2f}%",
                f"{avg_ttm:.2f}",
                f"{stats.get('total_turnover', nan)/1e8:.2f}",
                f"{stats.get('daily_turnover', nan)/1e8:.2f}",
                f"{stats.get('sharpe_ratio', nan):.2f}",
                f"{stats.get('max_ddpercent', nan):.2f}%",
                int(stats.get('total_trade_count', nan)),
                int(total_days),
                profit_loss_ratio
            ])

            # Extreme mode
            engine, df = run_backtesting(
                strategy_class=IndexExtremeStrategy,
                strat_setting=strat_setting,
                interval=Interval.DAILY,
                start=start_date,
                end=end_date,
                capital=IndexStrategy.CASH_AVAILABLE,
                risk_free=0.02,
                annual_days=240,
            )
            stats = engine.calculate_statistics(df, output=False)
            avg_ttm = engine.strategy.cal_overall_avg_ttm()
            profit_days = stats.get("profit_days", nan)
            loss_days = stats.get("loss_days", nan)
            total_days = stats.get("total_days", nan)
            profit_loss_ratio = f"{profit_days}:{loss_days}" if loss_days > 0 else f"{profit_days}:0"
            results.append([
                str(year),
                spread_type_str,
                "Extreme",
                f"{stats.get('total_return', nan):.2f}%",
                f"{stats.get('annual_return', nan):.2f}%",
                f"{avg_ttm:.2f}",
                f"{stats.get('total_turnover', nan)/1e8:.2f}",
                f"{stats.get('daily_turnover', nan)/1e8:.2f}",
                f"{stats.get('sharpe_ratio', nan):.2f}",
                f"{stats.get('max_ddpercent', nan):.2f}%",
                int(stats.get('total_trade_count', nan)),
                int(total_days),
                profit_loss_ratio
            ])

    stats_df = pd.DataFrame(results, columns=columns)
    # Save to CSV
    stats_df.to_csv(os.path.join(const.PATH.STRATEGY_POOL, "index_backtest_stats.csv"), index=False)


def main():
    strat_setting = {
        "select_mode": "max_ytm",
        "threshold_type": ThresholdConfig.Type.ZSCORE,  
        "low_threshold": -0.67,
        "high_threshold": 0.67,
        "expert_mode": False,
        "extreme_low_threshold": -1.28,
        "extreme_high_threshold": 1.28,
    }

    engine, df = run_backtesting(
        strategy_class=IndexExtremeStrategy,
        strat_setting=strat_setting,
        interval=Interval.DAILY,
        start=datetime(2020, 1, 1, tzinfo=DB_TZ),
        end=datetime(2020, 12, 31, tzinfo=DB_TZ),
        capital=IndexStrategy.CASH_AVAILABLE,
        risk_free=0.02,
        annual_days=240,
    )

    df.to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, "index_backtest_result.csv"), index=True
    )
    engine.calculate_statistics(df)
    engine.show_chart(df)

    # Process and save trades
    trade_data = []
    for trade_id, trade in engine.trades.items():
        trade_data.append({
            "Trade ID": trade_id.split(".")[1],  
            "Time": trade.datetime.strftime("%Y-%m-%d"),
            "Symbol": trade.vt_symbol,
            "Direction": trade.direction.value,
            "Price": trade.price,
            "Volume": trade.volume
        })

    trades_df = pd.DataFrame(trade_data)
    trades_df.set_index("Trade ID", inplace=True)
    trades_df.to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, "index_backtest_trades.csv"), index=True
    )


if __name__ == "__main__":
    # main()
    run_multiple_backtesting(2017, 2023)
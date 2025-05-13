from vnpy_portfoliostrategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ, get_database
from datetime import datetime
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


def main():
    strat_setting = {
        "select_mode": "max_ytm",
        "threshold_type": ThresholdConfig.Type.AVG_PCTL,  # Use average percentile thresholds
        "low_threshold": 0.25,
        "high_threshold": 0.75,
        "expert_mode": False,
        "extreme_low_threshold": -0.10,
        "extreme_high_threshold": 1.90,
    }

    engine, df = run_backtesting(
        strategy_class=IndexStrategy,
        strat_setting=strat_setting,
        interval=Interval.DAILY,
        start=datetime(2024, 7, 2, tzinfo=DB_TZ),
        end=datetime(2024, 12, 31, tzinfo=DB_TZ),
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
    main()

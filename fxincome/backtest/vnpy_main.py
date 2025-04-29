from vnpy_portfoliostrategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ, get_database
from datetime import datetime
from index_extreme import IndexExtremeStrategy
from fxincome import logger


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
             raise ValueError("No daily CFETS symbols found in database. Run index_process_data.load_cdb_ohlc first.")
        
        vt_symbols = discovered_symbols

        # Generate default parameters if not provided
        # Total turnover = volume * size * price.
        sizes = {symbol: 1 for symbol in vt_symbols} if sizes is None else sizes
        # Rates are commission rates in percentage.
        rates = {symbol: 1 / 1e5 for symbol in vt_symbols} if rates is None else rates
        # Price ticks are the minimum price change in dollars for the asset.
        priceticks = {symbol: 1 / 10000 for symbol in vt_symbols} if priceticks is None else priceticks
        # If the market price is 102, the traded price for buy is 102 + slippage.
        slippages = {symbol: 0.01 for symbol in vt_symbols} if slippages is None else slippages
        

    # Ensure parameters are provided if symbols were passed directly
    elif not all([rates, slippages, sizes, priceticks]):
         raise ValueError("rates, slippages, sizes, and priceticks must be provided if vt_symbols is specified.")


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
    engine.strategy.log_overall_avg_ttm()
    df = engine.calculate_result()
    return engine, df


def main():
    strat_setting = {
        "expert_mode": False,
        "extreme_low_percentile": -0.10,  # Impossible to reach. Strategy downgrades to normal mode.
        "extreme_high_percentile": 1.10,  # Impossible to reach. Strategy downgrades to normal mode.
    }

    engine, df = run_backtesting(
        strategy_class=IndexExtremeStrategy,
        strat_setting=strat_setting,
        interval=Interval.DAILY,
        start=datetime(2023, 12, 29, tzinfo=DB_TZ),
        end=datetime(2024, 12, 31, tzinfo=DB_TZ),
        capital=IndexExtremeStrategy.CASH_AVAILABLE,
        risk_free=0.02,
        annual_days=240,
    )

    df.to_csv("d:/backtest_result.csv", index=True)
    engine.calculate_statistics(df)
    engine.show_chart(df)

if __name__ == "__main__":
    main()

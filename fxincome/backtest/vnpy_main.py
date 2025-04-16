from vnpy_portfoliostrategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ, get_database
from datetime import datetime
from index_extreme import IndexExtremeStrategy
from fxincome import logger


def run_backtesting(
    strategy_class,
    setting,
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
        rates = {symbol: 2 / 10000 for symbol in vt_symbols} if rates is None else rates
        slippages = {symbol: 0.01 for symbol in vt_symbols} if slippages is None else slippages
        sizes = {symbol: 100 for symbol in vt_symbols} if sizes is None else sizes
        priceticks = {symbol: 0.0001 for symbol in vt_symbols} if priceticks is None else priceticks

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
    engine.add_strategy(strategy_class, setting)
    engine.load_data()
    engine.run_backtesting()
    df = engine.calculate_result()
    return df


def show_portafolio(df):
    engine = BacktestingEngine()
    engine.calculate_statistics(df)
    fig = engine.show_chart(df)
    fig.show()


def main():
    # Removed symbol loading and dynamic dict creation here

    df = run_backtesting(
        strategy_class=IndexExtremeStrategy,
        setting={},
        interval=Interval.DAILY,
        start=datetime(2017, 1, 2, tzinfo=DB_TZ),
        end=datetime(2024, 4, 30, tzinfo=DB_TZ),
        capital=1_000_000,
        risk_free=0.02,
        annual_days=240,
    )

    # show_portafolio(df)


if __name__ == "__main__":
    main()

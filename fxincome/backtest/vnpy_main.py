from vnpy_portfoliostrategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ
from datetime import datetime
from index_extreme import IndexExtremeStrategy



def run_backtesting(
    strategy_class,
    setting,
    vt_symbols,
    interval,
    start,
    end,
    rates,
    slippages,
    sizes,
    priceticks,
    capital,
    risk_free,
    annual_days,
):

    engine = BacktestingEngine()
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
    df = run_backtesting(
        strategy_class=IndexExtremeStrategy,
        setting={},
        vt_symbols=["010214.CFETS", "010216.CFETS"],
        interval=Interval.DAILY,
        start=datetime(2014, 1, 2, tzinfo=DB_TZ),
        end=datetime(2024, 4, 30, tzinfo=DB_TZ),
        rates={"010214.CFETS": 2 / 10000, "010216.CFETS": 2 / 10000},
        slippages={"010214.CFETS": 0.01, "010216.CFETS": 0.01},
        sizes={"010214.CFETS": 100, "010216.CFETS": 100},
        priceticks={"010214.CFETS": 0.0001, "010216.CFETS": 0.0001},
        capital=1_000_000,
        risk_free=0.02,
        annual_days=240,
    )

    # show_portafolio(df)


if __name__ == "__main__":
    main()

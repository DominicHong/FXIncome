import pandas as pd
import sqlite3
from WindPy import w
from datetime import datetime
from datetime import timedelta
from fxincome import const, logger


def update_strat_pool_hist_simi_featrues(conn):
    begin = pd.read_sql(
        "SELECT max(date) FROM strat_pool_hist_simi_features", conn
    ).iat[0, 0]
    end = datetime.today() + timedelta(days=-1)
    wind_data = w.edb("S0059744,S0059749,G0000886,G0000891,000300.SH", begin, end)
    if wind_data.ErrorCode != 0:
        logger.info(
            f"History Similarity Features Table Wind API Error code: {wind_data.ErrorCode}"
        )
        return
    if len(wind_data.Times) <= 1:
        logger.info("No new data to update for features.")
        return

    wind_list = [wind_data.Times] + wind_data.Data
    df = pd.DataFrame(
        wind_list,
        index=["date", "t_1y", "t_10y", "t_us_1y", "t_us_10y", "hs300"],
    )
    df = df.T
    df[1:].to_sql(
        "strat_pool_hist_simi_features", conn, if_exists="append", index=False
    )


def update_strat_pool_hist_simi_backtest(asset_code: str, conn):
    db_query = (
        f"SELECT * FROM strat_pool_hist_simi_backtest WHERE asset_code='{asset_code}'"
    )
    df = pd.read_sql(db_query, conn)
    df_of_code = df[df["asset_code"] == asset_code]
    latest_date = df_of_code["date"].max()

    end_date = datetime.today() + timedelta(-1)

    wind_data1 = w.wsd(
        codes=asset_code,
        fields="open,high,low,close,volume,turn",
        beginTime=latest_date,
        endTime=end_date,
    )
    if wind_data1.ErrorCode != 0:
        logger.info(f"{asset_code} Wind API Error code: {wind_data1.ErrorCode}")
        return
    if len(wind_data1.Times) <= 1:
        logger.info("No new data to update for backtest.")
        return
    wind_list = [wind_data1.Times] + [[asset_code] * len(wind_data1.Times)] + wind_data1.Data
    df1 = pd.DataFrame(
        wind_list,
        index=[
            "date",
            "asset_code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ],
    )
    df1 = df1.T
    # Get prices of 204001.SH
    wind_data2 = w.wsd(
        codes="204001.SH", fields="vwap,close", beginTime=latest_date, endTime=end_date
    )
    if wind_data2.ErrorCode != 0:
        logger.info(f"204001 Wind API Error code: {wind_data2.ErrorCode}")
        return
    if len(wind_data2.Times) <= 1:
        logger.info("No new data to update.")
        return

    wind_list = [wind_data2.Times] + wind_data2.Data
    df2 = pd.DataFrame(
        wind_list,
        index=["date", "GC001_avg", "GC001_close"],
    )
    df2 = df2.T
    df = pd.merge(df1, df2, on="date")
    df[1:].to_sql(
        "strat_pool_hist_simi_backtest", conn, if_exists="append", index=False
    )


def main():
    conn = sqlite3.connect(const.DATABASE_CONFIG.SQLITE_DB_CONN)
    w.start()

    update_strat_pool_hist_simi_featrues(conn)
    asset_code = "511260.SH"
    update_strat_pool_hist_simi_backtest(asset_code, conn)

    w.close()



    conn.close()

if __name__ == "__main__":
    main()

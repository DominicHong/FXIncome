# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import scipy
import numpy as np
import pandas as pd
import os
import glob

from datetime import datetime
from fxincome import const, logger
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData



def process_data(lookback_days: int = 3 * 250):

    conn = sqlite3.connect(const.DB.SQLITE_CONN)

    # Load bond information to databse
    bond_info = pd.read_csv(const.INDEX_ENHANCEMENT.CDB_INFO, encoding="gbk")
    bond_info.to_sql(
        const.DB.TABLES.IndexEnhancement.CDB_INFO,
        conn,
        if_exists="replace",
        index=False,
    )

    # Load yield spread to database
    cdb_yc = pd.read_csv(const.INDEX_ENHANCEMENT.CDB_YC)

    cdb_yc["spread_53"] = cdb_yc.y5 - cdb_yc.y3
    cdb_yc["spread_75"] = cdb_yc.y7 - cdb_yc.y5
    cdb_yc["spread_73"] = cdb_yc.y7 - cdb_yc.y3

    # Average Normalization
    cdb_yc["avg_53"] = cdb_yc.spread_53 / (cdb_yc.y5 + cdb_yc.y3)
    cdb_yc["avg_75"] = cdb_yc.spread_75 / (cdb_yc.y7 + cdb_yc.y5)
    cdb_yc["avg_73"] = cdb_yc.spread_73 / (cdb_yc.y7 + cdb_yc.y3)

    # Z-Score Normalization using rolling window
    def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score for a series using specified window."""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std

    # Apply rolling z-score normalization
    cdb_yc["zscore_53"] = rolling_zscore(cdb_yc["spread_53"], lookback_days)
    cdb_yc["zscore_75"] = rolling_zscore(cdb_yc["spread_75"], lookback_days)
    cdb_yc["zscore_73"] = rolling_zscore(cdb_yc["spread_73"], lookback_days)

    # Calculate percentile ranks for original spread. "weak" means the ratio of samples <= given value
    cdb_yc["pctl_spread_53"] = cdb_yc.spread_53.rolling(window=lookback_days).apply(
        lambda x: scipy.stats.percentileofscore(x, x.iloc[-1], kind='weak') / 100
    )
    cdb_yc["pctl_spread_75"] = cdb_yc.spread_75.rolling(window=lookback_days).apply(
        lambda x: scipy.stats.percentileofscore(x, x.iloc[-1], kind='weak') / 100
    )
    cdb_yc["pctl_spread_73"] = cdb_yc.spread_73.rolling(window=lookback_days).apply(
        lambda x: scipy.stats.percentileofscore(x, x.iloc[-1], kind='weak') / 100
    )

    # Calculate percentile ranks for average spread. "weak" means the ratio of samples <= given value
    cdb_yc["pctl_avg_53"] = cdb_yc.avg_53.rolling(window=lookback_days).apply(
        lambda x: scipy.stats.percentileofscore(x, x.iloc[-1], kind='weak') / 100
    )
    cdb_yc["pctl_avg_75"] = cdb_yc.avg_75.rolling(window=lookback_days).apply(
        lambda x: scipy.stats.percentileofscore(x, x.iloc[-1], kind='weak') / 100
    )
    cdb_yc["pctl_avg_73"] = cdb_yc.avg_73.rolling(window=lookback_days).apply(
        lambda x: scipy.stats.percentileofscore(x, x.iloc[-1], kind='weak') / 100
    )

    # Round normalization and percentile columns to 5 decimal places
    avg_cols = ['avg_53', 'avg_75', 'avg_73']
    zscore_cols = ['zscore_53', 'zscore_75', 'zscore_73']
    pctl_cols = ['pctl_spread_53', 'pctl_spread_75', 'pctl_spread_73',
                 'pctl_avg_53', 'pctl_avg_75', 'pctl_avg_73']
    
    cdb_yc[zscore_cols + pctl_cols + avg_cols] = cdb_yc[zscore_cols + pctl_cols + avg_cols].round(5)

    cdb_yc.to_sql(
        const.DB.TABLES.IndexEnhancement.CDB_YC,
        conn,
        if_exists="replace",
        index=False,
    )

    conn.close()
    return bond_info


def load_cdb_ohlc():
    # Define the directory path
    directory = r"D:\ProjectRicequant\fxincome\strategies_pool\index_enhancement\cdb_ohlc"
    
    # Get all CSV files that don't start with 'cashflow'
    csv_files = [f for f in glob.glob(os.path.join(directory, "*.csv")) 
                 if not os.path.basename(f).startswith("cash_flow")]
    
    if not csv_files:
        raise ValueError(f"No valid CSV files found in {directory}")
    
    # Initialize an empty list to store DataFrames
    dfs = []
    
    # Read each CSV file and add filename as sec_code
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                df['sec_code'] = os.path.splitext(os.path.basename(file))[0]
                # Drop any completely empty columns
                df = df.dropna(axis=1, how='all')
                dfs.append(df)
            else:
                logger.warning(f"Empty CSV file found: {file}")
        except Exception as e:
            logger.error(f"Error reading file {file}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("No valid data found in any of the CSV files")
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Clean up the final DataFrame
    combined_df = combined_df.dropna(axis=1, how='all')  # Remove any empty columns
    combined_df = combined_df.drop(columns=["CODE"])
    combined_df.columns = combined_df.columns.str.lower()
    print(f"Successfully loaded {len(csv_files)} files. Shape of combined DataFrame: {combined_df.shape}")
    
    bars = []
    for _, row in combined_df.iterrows():
        dt = datetime.strptime(row["date"], "%Y-%m-%d")

        bar = BarData(
            symbol=row["sec_code"],
            exchange=Exchange.CFETS,
            datetime=dt,
            interval=Interval.DAILY,
            open_price=row["open"],
            high_price=row["high"],
            low_price=row["low"],
            close_price=row["close"],
            volume=row["vol"],
            gateway_name="CSV",
        )
        bar.extra = {
            "ytm": row["ytm"],
            "matu": row["matu"],
            "out_bal": row["out_bal"],
        }
        
        bars.append(bar)

    db = get_database()
    db.save_bar_data(bars)
    print(f"Successfully saved {len(bars)} bars to database")

    return combined_df


if __name__ == "__main__":
    # process_data(lookback_days=3 * 250)
    load_cdb_ohlc()

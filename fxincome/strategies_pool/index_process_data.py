# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import scipy
import datetime
from pathlib import Path
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
import numpy as np
from fxincome import const, logger


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

    # Calculate percentile ranks for averagespread. "weak" means the ratio of samples <= given value
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


if __name__ == "__main__":
    process_data(lookback_days=3 * 250)

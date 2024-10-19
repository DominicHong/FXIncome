# -*- coding: utf-8 -*-
import pandas as pd
import os
import scipy
import datetime
from pandas import DataFrame
from fxincome import const, logger
from sqlalchemy import create_engine


def feature_engineering(
    df: DataFrame,
    yield_pctl_window=5 * 250,
    yield_chg_window_long=20,
    yield_chg_window_short=10,
    yield_chg_pctl_window=5 * 250,
    stock_return_window=10,
    stock_return_pctl_window=5 * 250,
    hs300_pctl_window=5 * 250,
):
    """
    Dates' types are dt.date.
    """
    df = df.copy()
    df["date"] = df["date"].dt.date
    # Only bond trade days remain.
    df = df.dropna(subset=["t_10y", "t_1y"])

    # For missing values of us treasury bonds and hs300, fill them with the previous value.
    df[["t_us_10y", "t_us_1y", "hs300"]] = df[["t_us_10y", "t_us_1y", "hs300"]].ffill()
    df["t_us_cn_10y_spread"] = df["t_us_10y"] - df["t_10y"]

    # 10-year Chinese Treasury bond yield change are calculated by short and long term.
    # 10y_yield_change_long = t_10y(t) / t_10y(t - yield_chg_window_long)
    # 10y_yield_change_short = t_10y(t) / t_10y(t - yield_chg_window_short)
    # 1y_yield_change_short = t_1y(t) / t_1y(t - yield_chg_window_short)
    df["t_10y_yield_chg_long"] = df["t_10y"] - df["t_10y"].shift(yield_chg_window_long)
    df["t_10y_yield_chg_short"] = df["t_10y"] - df["t_10y"].shift(
        yield_chg_window_short
    )
    df["t_1y_yield_chg_short"] = df["t_1y"] - df["t_1y"].shift(yield_chg_window_short)

    # stock_return = hs300(t) / hs300(t - stock_return_window) - 1
    df["stock_return"] = df["hs300"] / df["hs300"].shift(stock_return_window) - 1
    df["stock_return_pctl"] = (
        df["stock_return"]
        .rolling(stock_return_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df["hs300_pctl"] = (
        df["hs300"]
        .rolling(hs300_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )

    # Calculate the percentiles of 10-year Chinese Treasury bond yield, 1-year Chinese Treasury bond yield, 10-year US
    # Treasury and Chinese Treasury spread for the past "yield percentile window".
    df["t_10y_pctl"] = (
        df["t_10y"]
        .rolling(yield_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df["t_1y_pctl"] = (
        df["t_1y"]
        .rolling(yield_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df["t_us_cn_10y_spread_pctl"] = (
        df["t_us_cn_10y_spread"]
        .rolling(yield_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )

    # Calculate the percentiles of 10y_yield_change and 1y_yield_change for the past "yield change percentile window".
    df["t_10y_yield_chg_long_pctl"] = (
        df["t_10y_yield_chg_long"]
        .rolling(yield_chg_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df["t_10y_yield_chg_short_pctl"] = (
        df["t_10y_yield_chg_short"]
        .rolling(yield_chg_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df["t_1y_yield_chg_short_pctl"] = (
        df["t_1y_yield_chg_short"]
        .rolling(yield_chg_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )

    df["avg_chg_5"] = (df["t_10y"] - df["t_10y"].rolling(5).mean()) / df[
        "t_10y"
    ].rolling(5).mean()
    df["avg_chg_10"] = (df["t_10y"] - df["t_10y"].rolling(10).mean()) / df[
        "t_10y"
    ].rolling(10).mean()
    df["avg_chg_20"] = (df["t_10y"] - df["t_10y"].rolling(20).mean()) / df[
        "t_10y"
    ].rolling(20).mean()

    # iterate const.HistorySimilarity.LABELS to generate label values.
    # yield_chg_fwd_n = t_10y(t+n) - t_10y(t)
    for day, name in const.HistorySimilarity.LABELS_YIELD_CHG.items():
        df[name] = df["t_10y"].shift(-day) - df["t_10y"]
    return df


def calculate_all_similarity(df, features: list, metric="euclidean"):
    """
    Calculate the Euclidean distance between each pair of rows in a DataFrame for the given features.

    Args:
        df (DataFrame): The input DataFrame containing the data. It must include a 'date' column.
                        Dates' types are supposed to be dt.date.
        features (list): The list of feature column names to be used in the distance calculation.
        metric (str): The type of similarity. Must be either "euclidean" or "cosine".
    Returns:
        similarity_matrix(DataFrame): A n * n matrix.
                         Index and columns are the dates, and the values are the distances between index and dates.
                         Dates' types are the same as df['date'].
    """
    if metric not in ["euclidean", "cosine"]:
        raise ValueError("sim_type must be either 'euclidean' or 'cosine'")

    # Separate dates from features
    df = df.dropna(subset=features)
    dates = df["date"]
    feature_data = df[features].values

    # The result is a 2D array where the element at position (i, j) is
    # the distance between the i-th and j-th row of the DataFrame
    distances = scipy.spatial.distance.cdist(feature_data, feature_data, metric=metric)

    # Create a DataFrame where index and columns are the dates, and the values are the distances
    similarity_matrix = pd.DataFrame(distances, index=dates, columns=dates)
    similarity_matrix = similarity_matrix.reset_index()
    similarity_matrix = similarity_matrix.rename(columns={"index": "date"})

    return similarity_matrix


def read_processed_data_from_csv(distance_type: str):
    """
    Read feature_label file and similarity matrix from csv files.
    Convert objects of "date" column to dt.date type.
    Convert column names of dates in similarity matrix to dt.date type.

    Args:
        distance_type(str): The type of distance. Must be either "euclidean" or "cosine".
    Returns:
        feature_lable_df(DataFrame): The feature_label DataFrame.
        similarity_matrix(DataFrame): The similarity matrix DataFrame.
    """
    if distance_type not in ["euclidean", "cosine"]:
        raise ValueError("distance_type must be either 'euclidean' or 'cosine'")
    if distance_type == "euclidean":
        simi_file = const.HistorySimilarity.SIMI_EUCLIDEAN
    else:
        simi_file = const.HistorySimilarity.SIMI_COSINE

    data_path = os.path.join(const.PATH.STRATEGY_POOL, "history_processed.csv")
    feature_label_df = pd.read_csv(data_path, parse_dates=["date"])
    feature_label_df["date"] = feature_label_df["date"].dt.date

    data_path = os.path.join(const.PATH.STRATEGY_POOL, simi_file)
    similarity_matrix = pd.read_csv(data_path, parse_dates=["date"])
    similarity_matrix["date"] = similarity_matrix["date"].dt.date
    rename_dict = {}
    for col in similarity_matrix.columns:
        if col == "date":
            continue
        try:
            rename_dict[col] = datetime.date.fromisoformat(col)
        except ValueError:
            logger.info(f"Column {col} is not in ISO format.")
            continue
    similarity_matrix = similarity_matrix.rename(columns=rename_dict)

    return feature_label_df, similarity_matrix


def main():
    # data = pd.read_csv(
    #     os.path.join(const.PATH.STRATEGY_POOL, const.HistorySimilarity.SRC_NAME),
    #     parse_dates=["date"],
    # )
    connection_string = (f'mysql+mysqlconnector://{const.MYSQL_CONFIG.USER}:{const.MYSQL_CONFIG.PASSWORD}'
                         f'@{const.MYSQL_CONFIG.HOST}/{const.MYSQL_CONFIG.DATABASE}')
    engine=create_engine(connection_string)
    data_query = f"SELECT * FROM {const.PATH.STRATEGY_POOL.replace(const.PATH.MAIN, '').replace('/', '')}_{const.HistorySimilarity.SRC_NAME.replace('.csv', '')}"
    data=pd.read_sql(data_query,engine)
    engine.dispose()
    data = feature_engineering(
        df=data,
        yield_pctl_window=const.HistorySimilarity.PARAMS["YIELD_PCTL_WINDOW"],
        yield_chg_pctl_window=const.HistorySimilarity.PARAMS["YIELD_CHG_PCTL_WINDOW"],
        yield_chg_window_long=const.HistorySimilarity.PARAMS["YIELD_CHG_WINDOW_LONG"],
        yield_chg_window_short=const.HistorySimilarity.PARAMS["YIELD_CHG_WINDOW_SHORT"],
        stock_return_window=const.HistorySimilarity.PARAMS["STOCK_RETURN_WINDOW"],
        stock_return_pctl_window=const.HistorySimilarity.PARAMS["STOCK_RETURN_PCTL_WINDOW"],
        hs300_pctl_window=const.HistorySimilarity.PARAMS["HS300_PCTL_WINDOW"],
    )
    data.to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, const.HistorySimilarity.FEATURE_FILE),
        index=False,
        encoding="utf-8",
    )

    distance_df = calculate_all_similarity(
        data, const.HistorySimilarity.FEATURES, metric="euclidean"
    )

    distance_df.to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, const.HistorySimilarity.SIMI_EUCLIDEAN),
        encoding="utf-8",
        index=False,
    )

    distance_df = calculate_all_similarity(
        data, const.HistorySimilarity.FEATURES, metric="cosine"
    )

    distance_df.to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, const.HistorySimilarity.SIMI_COSINE),
        encoding="utf-8",
        index=False,
    )


if __name__ == "__main__":
    main()
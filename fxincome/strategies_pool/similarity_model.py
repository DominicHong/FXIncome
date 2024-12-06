import numpy as np
import pandas as pd
import os
import logging
import sqlite3
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)
import datetime
from pandas import DataFrame

import fxincome.strategies_pool.similarity_process_data
from fxincome import logger, handler, const


def print_prediction_stats(train_df, test_df, distance_min, distance_max, smooth_c):
    # Make logger simple
    handler.setFormatter(logging.Formatter(fmt="%(message)s"))
    logger.addHandler(handler)

    # Print train set and test set info
    logger.info(f"train size: {len(train_df)}; test size: {len(test_df)}")
    logger.info(
        f"train start date: {train_df['date'].iloc[0]}; train end date: {train_df['date'].iloc[-1]}"
    )
    logger.info(
        f"test start date: {test_df['date'].iloc[0]}; test end date: {test_df['date'].iloc[-1]}"
    )
    logger.info(
        f"Distance_Min: {distance_min}; Distance_Max: {distance_max}; Smooth_C: {smooth_c}"
    )

    # Print prediction stats
    for day in const.HistorySimilarity.LABELS_YIELD_CHG.keys():
        logger.info(f"###### Predicions of {day} days in the future ######")
        # Keep only rows where predictions are not NaN.
        # Nan predictions mean no similar dates are found in the history.
        # valid_predictions = test_df[~test_df[f"pred_{day}"].isna()]
        valid_predictions = test_df.dropna(subset=[f"pred_{day}"])
        logger.info(
            f"Valid_prediction_ratio: {len(valid_predictions) / len(test_df):.2f}"
        )
        pred_df = test_df.dropna(subset=[f"pred_{day}", f"actual_{day}"])
        pred_values = pred_df[f"pred_{day}"]
        actual_values = pred_df.loc[pred_values.index, f"actual_{day}"]
        logger.info(
            f"Accuracy: {accuracy_score(actual_values, pred_values) * 100:.2f}%"
        )
        logger.info(confusion_matrix(actual_values, pred_values))
        logger.info(
            f"Precision: {precision_score(actual_values, pred_values, zero_division=0) * 100:.2f}%"
        )
        logger.info(
            f"Recall: {recall_score(actual_values, pred_values, zero_division=0) * 100:.2f}%"
        )


def train_test_split(df: DataFrame, train_ratio: float = 0.8, gap: int = 30):
    """
    Split the data into training and testing sets.

    Args:
        df (DataFrame): The input DataFrame containing the data. It must include a 'date' column.
        train_ratio (float): The ratio of the data to be used for training. The rest will be used for testing.
        gap (int): The number of trade days between the end of training set and beginning of testing sets.
    Returns:
        tuple: A tuple containing two DataFrames:
            - train_df: The training set.
            - test_df: The testing set.
    """
    data = df.sort_values(by="date")
    train_size = int(len(data) * train_ratio)
    train_df = data.iloc[:train_size]
    test_df = data.iloc[(train_size + gap):]

    return train_df, test_df


def avg_yield_chg(
    given_date,
    simi_matrix_with_yield_chg,
    distance_min: float,
    distance_max: float,
    smooth_c: float = 5.0,
    yield_chg_fwd:str ="yield_chg_fwd_10",
):
    """
    Calculate the weighted average yield change for a given date based on the similarity matrix.

    The weights are calculated as the inverse of the distance, meaning that smaller distances
    will have larger weights and larger distances will have smaller weights. The distances considered
    are those within the range specified by distance_min(included) and distance_max(NOT included).

    Args:
        given_date: Calculate other dates' weighted average yield change similar to this date.
                    Its type should be of the same type as the date columns of simi_matrix_with_yield_chg.
        simi_matrix_with_yield_chg (DataFrame): The similarity matrix with yield changes.
        distance_min (float): included.
        distance_max (float): NOT included.
        smooth_c(float): smooth the weight differences caused by the inverse of the distance.
                        Weights are closer to distance average weights as smooth_c increases.
        yield_chg_fwd (str, optional): The column name for the forward yield change. Defaults to "yield_chg_fwd_10".

    Returns:
        yield_chg_avg(float): The weighted average yield change for the given row.
                            If no similar dates are found, return np.nan.
        close_rows(DataFrame): The rows in the similarity matrix that are within the specified distance range.
    """

    # Find the column with the same date as the given date
    close_rows = simi_matrix_with_yield_chg[["date", given_date, yield_chg_fwd]]
    close_rows = close_rows[
        (close_rows[given_date] >= distance_min)
        & (close_rows[given_date] < distance_max)
    ]

    if close_rows.empty:
        return np.nan, close_rows

    # Drop dates when yield_chg_fwd is unavailable yet.
    close_rows = close_rows.dropna(subset=[yield_chg_fwd])

    # Calculate weighted average of yield_chg_fwd, weighted by INVERSE of distance(smaller distance, higher weight)
    weights = 1 / (close_rows[given_date] + smooth_c)
    weights = weights / weights.sum()
    yield_chg_avg = (close_rows[yield_chg_fwd] * weights).sum()

    return yield_chg_avg, close_rows


def predictions_test(
    similarity_df: DataFrame,
    sample_df: DataFrame,
    distance_min: float,
    distance_max: float,
    smooth_c: float,
    train_ratio: float,
    gap: int,
):
    """
    Predict the yield change direction in n days and calculate the prediction accuracy of the test set.
    The sample_df will be split into training and testing sets based on the train_ratio and gap.
    The yield change direction is determined by yield_chg_fwd in the form of "yield_chg_fwd_n".
    n must be either 5, 10, 20, or 30. The prediction is based on the weighted average yield change
    for each row in the test DataFrame, where the weights are calculated based on the similarity matrix.

    Args:
        similarity_df (DataFrame): Distance matrix, including column 'date'(m rows),
                                                              column ['2010-1-27', '2011-12-1'...](also m columns)
        sample_df (DataFrame): includes column 'date' and columns 'yield_chg_fwd_n'.
        distance_min (float): included.
        distance_max (float): NOT included
        smooth_c(float): smooth the weight differences caused by the inverse of the distance.
                    Weights are closer to distance average weights as smooth_c increases.
        train_ratio (float): The ratio of the data to be used for training. The rest will be used for testing.
        gap (int): The number of trade days between the end of training set and beginning of testing sets.
    Outputs:
        A test_predictions.csv file in the strategy_pool folder.
        Prints prediction statistics for each label in const.HistorySimilarity.LABELS.
    """

    # Add columns 'yield_chg_fwd_n' to similarity_df.
    # similarity_df now includes:
    # column 'date'(m rows), column ['2010-1-27', '2011-12-1'...](also m columns)
    # plus n columns 'yield_chg_fwd_n'(depends on const.HistorySimilarity.LABELS).
    sample_df = sample_df[["date"] + list(const.HistorySimilarity.LABELS_YIELD_CHG.values())]
    train_df, test_df = train_test_split(sample_df, train_ratio=train_ratio, gap=gap)
    similarity_df = pd.merge(sample_df, similarity_df, left_on="date", right_on="date")

    # Dates in history are used to calculate weighted average yield change.
    # Rows on test_df["date"] should NOT be included, since they are not in history.
    similarity_df = similarity_df[~similarity_df["date"].isin(test_df["date"])]

    for day, name in const.HistorySimilarity.LABELS_YIELD_CHG.items():
        test_df[f"pred_weight_{day}"] = test_df.apply(
            lambda row: (
                avg_yield_chg(
                    row["date"],
                    similarity_df,
                    yield_chg_fwd=name,
                    distance_min=distance_min,
                    distance_max=distance_max,
                    smooth_c=smooth_c,
                )[0]
            ),
            axis=1,
        )
        # If weighted change > 0, prediction = 1;
        # if weighted change <= 0, prediction = 0;
        # else it cannot find similar dates within distance scope, then prediction = nan;
        test_df[f"pred_{day}"] = test_df[f"pred_weight_{day}"].apply(
            lambda x: 1 if x > 0 else 0 if x <= 0 else np.nan
        )
        test_df[f"actual_{day}"] = test_df[name].apply(
            lambda x: 1 if x > 0 else 0 if x <= 0 else np.nan
        )

        test_df = test_df.dropna(subset=[f"actual_{day}"])

    test_df.to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, "test_predictions.csv"),
        index=False,
    )
    print_prediction_stats(train_df, test_df, distance_min, distance_max, smooth_c)


def predict_yield_chg(
    dates_to_pred: list,
    similarity_df: DataFrame,
    sample_df: DataFrame,
    distance_min: float,
    distance_max: float,
    smooth_c: float,
):
    """
    Given some dates, predict the yield change direction in n days.
    The yield change direction is determined by yield_chg_fwd in the form of "yield_chg_fwd_n".
    n must be either 5, 10, 20, or 30. The prediction is based on the weighted average yield change
    of similar past dates, where the weights are calculated based on the similarity matrix.

    Args:
        dates_to_pred (list): A list of dates(datetime.date) to predict yield change.
        similarity_df (DataFrame): Distance matrix, including column 'date'(m rows),
                                                              column ['2010-1-27', '2011-12-1'...](also m columns)
        sample_df (DataFrame): includes column 'date' and columns 'yield_chg_fwd_n'.
        distance_min (float): included.
        distance_max (float): NOT included
        smooth_c(float): smooth the weight differences caused by the inverse of the distance.
                    Weights are closer to distance average weights as smooth_c increases.
    Returns:
        result_df(DataFrame): Predictions with at least columns 'pred_n' and 'pred_weight_n'.
        similar_dates(dict): DataFrames containing the predictions. One dataframe for each given date.
    """

    sample_df = sample_df[["date"] + list(const.HistorySimilarity.LABELS_YIELD_CHG.values())]
    sample_df=sample_df.sort_values(by="date",ascending=True)
    combined_df = pd.merge(sample_df, similarity_df, left_on="date", right_on="date")
    combined_df=combined_df.sort_values(by='date',ascending=True)

    # Predict only on given dates.
    result_df = sample_df[sample_df["date"].isin(dates_to_pred)].copy()

    # Predict
    similar_dates = {}
    for index, row in result_df.iterrows():
        # To record distances and labels for similar dates with the same date (row["date"])
        similar_dates_with_one_date = pd.DataFrame()
        for day, label_name in const.HistorySimilarity.LABELS_YIELD_CHG.items():
            # Use only past dates to predict the future
            history_df = combined_df[combined_df["date"] < row["date"]][:-day+1]

            yield_chg_avg, close_rows = avg_yield_chg(
                row["date"],
                history_df,
                yield_chg_fwd=label_name,
                distance_min=distance_min,
                distance_max=distance_max,
                smooth_c=smooth_c,
            )
            result_df.at[index, f"pred_weight_{day}"] = yield_chg_avg
            result_df.at[index, f"pred_{day}"] = (
                1 if yield_chg_avg > 0 else 0 if yield_chg_avg <= 0 else np.nan
            )
            # Merge predictions of different LABELS for similar dates with the same date(row["date"])
            if similar_dates_with_one_date.empty:
                similar_dates_with_one_date = close_rows
            else:
                similar_dates_with_one_date = pd.merge(
                    similar_dates_with_one_date,
                    close_rows,
                    on=["date", row["date"]],
                    how="outer",
                )
        similar_dates[row["date"]] = similar_dates_with_one_date

    conn = sqlite3.connect(const.DB.SQLITE_CONN)
    result_df.to_sql(
        const.DB.HistorySimilarity_TABLES["PREDICTIONS"],
        conn,
        if_exists="replace",
        index=False,
    )
    conn.close()

    return result_df, similar_dates


def main():
    all_samples, distance_df = (
        fxincome.strategies_pool.similarity_process_data.read_processed_data(
            "euclidean"
        )
    )
    # predictions_test(
    #     similarity_df=distance_df,
    #     sample_df=all_samples,
    #     distance_min=0.00,
    #     distance_max=0.25,
    #     smooth_c=5,
    #     train_ratio=0.90,
    #     gap=30,
    # )

    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 10, 18)
    dates_to_predict = [
        start_date + datetime.timedelta(days=x)
        for x in range((end_date - start_date).days + 1)
    ]
    predictions, similar_dates_dict = predict_yield_chg(
        dates_to_pred=dates_to_predict,
        similarity_df=distance_df,
        sample_df=all_samples,
        distance_min=0.00,
        distance_max=0.25,
        smooth_c=5,
    )
    for date, similar_dates_df in similar_dates_dict.items():
        print(date)
        print(similar_dates_df)


if __name__ == "__main__":
    main()

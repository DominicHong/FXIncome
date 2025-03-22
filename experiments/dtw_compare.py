import datetime
import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from random import randint
from dtw import *

data_path = 'd:/ProjectRicequant/stocks/'


def normalize(periods: list[pd.DataFrame]):
    #  The first day's price is set to 1. Consecutive prices are adjusted as the ratios of first day's price.
    new_periods = []
    for p in periods:
        p = p.copy()
        unit = p.iloc[0].close
        p.close = p.close / unit
        new_periods.append(p)
    return new_periods


def drawdown(df: pd.DataFrame):
    """
    Calculate the max drawdown for each day from the beginning of the dataframe.
    Args:
        df (Dataframe): Dataframe of a selected period.

    Returns:
        max_dd (float): max drawdown during this period
        df_dd (Dataframe): the max drawdown period from peak to trough
    """
    df = df.copy()
    # We call the days from beginning to x day as WINDOW. As x is increasing, this WINDOWS is expanding.
    # for every day, find the peak price in that WINDOW.
    peak = df.close.cummax()
    # for every day, calculate the drawdown from peak to that day.
    df['daily_dd'] = df.close / peak - 1.0
    # for every day, calculate the minimum (negative number) drawdown. This is the max drawdown in that WINDOW.
    mdd_in_window = df.daily_dd.cummin()
    # find the max of max drawdown in every WINDOW. This is the max drawdown in the whole period.
    max_dd = mdd_in_window.min()
    # trough date is the first day when it hits max drawdown.
    trough_df = df[df.daily_dd == max_dd]
    trough_date = trough_df.iloc[0].date
    # peak date is the last peak day before trough day
    peak_df = df[(df.date < trough_date) & (df.daily_dd == 0)]
    peak_date = peak_df.iloc[-1].date
    df_dd = df[(df.date >= peak_date) & (df.date <= trough_date)]

    return max_dd, df_dd


def drawup(df: pd.DataFrame):
    """
    Calculate the max drawup for each day from the beginning of the dataframe.
    Args:
        df (Dataframe): Dataframe of a selected period.

    Returns:
        max_du (float): max drawup during this period
        df_du (Dataframe): the max drawup period from trough to peak
    """
    df = df.copy()
    # We call the days from beginning to x day as WINDOW. As x is increasing, this WINDOWS is expanding.
    # for every day, find the trough price in that WINDOW.
    trough = df.close.cummin()
    # for every day, calculate the drawup from trough to that day.
    df['daily_du'] = df.close / trough - 1.0
    # for every day, calculate the maximum drawup. This is the max drawup in that WINDOW.
    mdu_in_window = df.daily_du.cummax()
    # find the max of max drawup in every WINDOW. This is the max drawup in the whole period.
    max_du = mdu_in_window.max()
    # peak date is the first day when it hits max drawup.
    peak_df = df[df.daily_du == max_du]
    peak_date = peak_df.iloc[0].date
    # trough date is the last trough day before peak day
    trough_df = df[(df.date < peak_date) & (df.daily_du == 0)]
    trough_date = trough_df.iloc[-1].date
    df_du = df[(df.date >= trough_date) & (df.date <= peak_date)]

    return max_du, df_du


def split_period(df: pd.DataFrame, num: int):
    """
    Split a dataframe into n consecutive periods.
    Args:
        df (Dataframe): Dataframe to be split.
        num (int): The original df will be split into this number of periods.
    Returns:
        df_slices (list): A list of Dataframes, each of which is a subset of the whole period.
    """
    period_len = len(df) // num
    df_slices = [df.iloc[i: i + period_len] for i in range(0, len(df), period_len)]

    return df_slices


def random_period(df: pd.DataFrame, num: int, length: int):
    """
    Randomly pick n periods, each with a fixed length.
    Args:
        df (Dataframe): Dataframe to pick from
        num (int): pick n periods
        length (int): the fixed length of each period

    Returns:
        A list of Dataframes, each of which is a subset of the whole period
    """
    #  Randomly select start dates without replacement.
    starts = random.sample(range(0, len(df) - length), num)
    return [df.iloc[start: start + length] for start in starts]


def calc_dtw(periods: list[pd.DataFrame]):
    """
    Given a list of time  series, calculate the DTW distance matrix for each of them.
    Args:
        periods (list[Dataframe]): A list of dataframes for different time series.
    Returns:
        df_dist_matrix (Dataframe): A DTW distance matrix for each pair of input time series.
        The matrix is a square, with shape of (num of periods, num of periods)
        Index and column names of the returned matrix dataframe are like 'start_date-end_date'.
    """
    ori_periods = normalize(periods)
    periods = []
    names = []
    #  Make sure no duplicated periods.
    for p in ori_periods:
        name = p.iloc[0].date.strftime('%Y%m%d') + '_' + p.iloc[-1].date.strftime('%Y%m%d')
        if name not in names:
            names.append(name)
            periods.append(p)
    distances = []
    for x_df, y_df in itertools.product(periods, periods):
        #  Assuming trade days are equally distributed in both two periods, we can ignore the timesteps.
        x = list(x_df.close.values)
        y = list(y_df.close.values)
        distance = dtw(x, y, keep_internals=True).distance
        distances.append(distance)
    data_matrix = np.array(distances).reshape(len(periods), len(periods))
    df_matrix = pd.DataFrame(data=data_matrix, index=names, columns=names)
    return df_matrix


def show_closest_periods(dist_matrix: pd.DataFrame, periods: list[pd.DataFrame], nclosest: int, path: str):
    """

    Args:
        dist_matrix (Dataframe): Distances between pairs of timeseries.
        periods (list[DataFrame]): Timeseries input for distance calculation.
        nclosest (int): This function will find the top n closest pair of timeseries and draw their figures.
        path (str): Location where the generated images will be stored.
    Returns:
        A heatmap of distance matrix.
        N figures for the pairs of timeseries.
    """
    #  Get rid of 0 distances (itself to itself).
    dist_matrix = dist_matrix.where(dist_matrix > 0.0, other=np.nan)
    #  Get row and column names of the top n closest pairs. 'Closest' means the shortest distances between the pair.
    #  n_closest_names is: [('startdate_enddate', 'startdate_enddate'), ('startdate_enddate', 'startdate_enddate')...]
    n_closest_names = dist_matrix.stack().nsmallest(nclosest).index.to_list()
    for name1, name2 in n_closest_names:
        dates = name1.split('_')
        start1 = datetime.datetime.strptime(dates[0], '%Y%m%d')
        end1 = datetime.datetime.strptime(dates[1], '%Y%m%d')
        dates = name2.split('_')
        start2 = datetime.datetime.strptime(dates[0], '%Y%m%d')
        end2 = datetime.datetime.strptime(dates[1], '%Y%m%d')
        distance = dist_matrix.at[name1, name2]
        for p in periods:
            if p.iloc[0].date == start1 and p.iloc[-1].date == end1:
                period_1 = p
                # p.plot(x='date', y='close')
                # plt.savefig(f'./results/images/{distance:.2f}_{n1}.jpg', dpi=600)
            elif p.iloc[0].date == start2 and p.iloc[-1].date == end2:
                period_2 = p
                # p.plot(x='date', y='close')
                # plt.savefig(f'./results/images/{distance:.2f}_{n2}.jpg', dpi=600)
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(range(len(period_1)), period_1.close, label=f'{name1}')
        ax.plot(range(len(period_2)), period_2.close, label=f'{name2}')
        ax.grid(axis="y")
        ax.set_ylim(0.6, 1.4)
        ax.set_title("HS300 Periods")
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Price")
        ax.legend()
        plt.savefig(f'{path}{distance:.2f}_{name1} vs {name2}.jpg', dpi=600)
        plt.close()
    plt.figure(figsize=(20, 20))
    sns.heatmap(dist_matrix, square=True, annot=True, cmap='Blues')
    plt.savefig(f'{path}matrix.jpg', dpi=600)
    plt.close()


def main():
    df = pd.read_csv(data_path + 'hs300.csv', parse_dates=['date'])
    df = df[(df.date.dt.year >= 2005) & (df.date.dt.year < 2023)]
    dd_threshold = -0.2
    du_threshold = 0.2
    drawdowns = []
    drawups = []

    samples = split_period(df, num=20)
    for sample in samples:
        max_dd, df_dd = drawdown(sample)
        if max_dd < dd_threshold:
            drawdowns.append(df_dd)
        max_du, df_du = drawup(sample)
        if max_du > du_threshold:
            drawups.append(df_du)
    dist_matrix = calc_dtw(drawdowns)
    show_closest_periods(dist_matrix, normalize(drawdowns), 10, './results/images/drawdowns/')
    dist_matrix = calc_dtw(drawups)
    show_closest_periods(dist_matrix, normalize(drawups), 20, './results/images/drawups/')

    # samples = random_period(df, num=100, length=100)
    # for sample in samples:
    #     max_dd, df_dd = drawdown(sample)
    #     if max_dd < dd_threshold:
    #         drawdowns.append(df_dd)
    #     max_dd, df_du = drawup(sample)
    #     if max_dd > du_threshold:
    #         drawups.append(df_du)
    # dist_matrix = calc_dtw(drawdowns)
    # show_closest_periods(dist_matrix, normalize(drawdowns), 10, './results/images/drawdowns/')
    # dist_matrix = calc_dtw(drawups)
    # show_closest_periods(dist_matrix, normalize(drawups), 10, './results/images/drawups/')


if __name__ == '__main__':
    main()

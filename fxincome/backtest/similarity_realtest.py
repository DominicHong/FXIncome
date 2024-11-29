# encoding=utf-8
import datetime
from dataclasses import dataclass
import sqlite3
import pandas as pd
from fxincome import const
import backtrader as bt
from fxincome.data_handler import main as data_handler
from fxincome.strategies_pool.similarity_process_data import main as similarity_process_data
from fxincome.strategies_pool.similarity_process_data import read_processed_data
from fxincome.strategies_pool.similarity_model import predict_yield_chg
from fxincome.backtest.similarity_backtest import NTraderStrategy,TradeEverydayStrategy
from fxincome.backtest.similarity_backtest import run_backtest


def similarity_model(porfolio: str, distance_type: str):
    conn = sqlite3.connect(const.DB.SQLITE_CONN)
    try:
        start_date = pd.read_sql(
            f"SELECT max(date) FROM 'strat.hist_simi.temp.predictions_{porfolio}' ", conn).iat[0, 0]
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    except BaseException:
        start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date.today()
    all_samples, distance_df = read_processed_data(distance_type)
    dates_to_predict = pd.date_range(start_date, end_date).date.tolist()
    predictions, similar_dates_dict = predict_yield_chg(
        dates_to_pred=dates_to_predict,
        similarity_df=distance_df,
        sample_df=all_samples,
        distance_min=0.00,
        distance_max=0.25,
        smooth_c=5,
    )


    if predictions.shape[0] == 1:
        return
    predictions[1:].to_sql(
        f"strat.hist_simi.temp.predictions_{porfolio}",
        conn,
        if_exists="append",
        index=False,
    )
    conn.close()


def make_data(asset_code_list: list, strat_date: datetime.date):
    asset_code_set = set(asset_code_list)
    for asset_code in asset_code_set:
        conn = sqlite3.connect(const.DB.SQLITE_CONN)
        sql = f"SELECT * FROM 'strat.hist_simi.raw_backtest' where asset_code='{asset_code.replace('-real','')}' and date>='{strat_date}'"
        df = pd.read_sql(sql, conn)
        sql = f"SELECT date FROM 'strat.hist_simi.processed_backtest' where asset_code='{asset_code}' and date>='{strat_date}'"
        dates = pd.read_sql(sql, conn)['date'].tolist()
        append_data = []
        for row, data in df.iterrows():
            if data['date'] in dates:
                continue
            else:
                data['open'] = input(
                    f"输入{data['date']}日，{data['asset_code']}可以交易的开盘价(净价):")
                data['asset_code'] = asset_code
                append_data.append(data)
        if append_data == []:
            continue
        append_df = pd.DataFrame(append_data)
        print(append_df)
        choice = input("是否确认添加数据？(y/n)")
        if choice == 'y':
            append_df.to_sql(
                "strat.hist_simi.processed_backtest",
                conn,
                if_exists="append",
                index=False,
            )
        else:
            print("已取消添加")

        conn.close()


@dataclass
class Portfolio:
    name: str
    asset_code: str
    distance_type: str
    strat: bt.Strategy
    num_traders: int
    pred_days: int


def main():
    start_date = datetime.date(2024, 11, 11)
    end_date = datetime.date.today()
    portfolios = [
        Portfolio("p1", '240004.IB-real', "euclidean", NTraderStrategy, 10, 10),
        Portfolio("p2", '240004.IB-real', "cosine", NTraderStrategy, 10, 10),
        Portfolio("p3", '240004.IB-real', "euclidean", NTraderStrategy, 5, 5),
        Portfolio("p4", '240004.IB-real', "cosine", NTraderStrategy, 5, 5),
    ]
    data_handler()
    make_data(
        [por.asset_code for por in portfolios if '-real' in por.asset_code], start_date)
    similarity_process_data()
    for por in portfolios:
        similarity_model(por.name, por.distance_type)
        print(
            f"\033[91m==================================Running backtest for {por.name}==================================\033[0m")
        run_backtest(
            strat=por.strat,
            asset_code=por.asset_code,
            start_date=start_date,
            end_date=end_date,
            num_traders=por.num_traders,
            pred_days=por.pred_days,
            pred_table=f"strat.hist_simi.temp.predictions_{por.name}",
            etf_table=f"strat.hist_simi.processed_backtest",
            sizer="all",
            tp_pct=0.0015,
            sl_pct=-0.0008,
            repo_commission=0.001 / 100,
            bond_commission=0.0002,
        )


if __name__ == '__main__':
    main()

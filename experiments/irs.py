import pandas as pd
import os
import pytest
import plotly.express as px
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta
from datetime import datetime


def float_leg_cf(data, roll_date, end_date, lag):
    """
    计算fr007 irs浮息端的应付利息。
    到期日是起息日的对年对月对日。计息基准为ACT/365。
    计息区间为季度，每季度设置为固定的7 * 13 = 91天，最后一个季度的结束日有残端（一般都超过91天）。
    从第一个重置日开始，每7天重置一次fr007。
    如果一个计息区间中有n个利率重置，第i个的利率为Ri，计息期限为Ti（年），按复利计算该计息区间的应付利息：
    应付利息 = (1 + R0 × T0) ×(1 + R1 × T1) … (1 + Rn × Tn) - 1
        Args:
            data(DataFrame): irs原始数据
            roll_date(datetime): 浮息决定日（重置日）， 第一个重置日是起息日的前lag天
            end_date(datetime): 假设该IRS合约的最后到期日，不一定是真正的到期日。
            lag(int): 重置日是起息日之前的几天。
        Returns:
            cf(list): 每个浮息端付息周期的应付利息
    """
    quarter_days = 91
    cf = []
    start_date = roll_date + relativedelta(days=lag)  # roll_date（即重置日）是第一个浮息决定日, 起息日比第一个重置日迟一日
    df = data[data.date.between(roll_date, end_date)]
    #  两个循环。第一个循环每次加91天，季度计息；第二个循环在季度里，每7天累乘得到复利利率。
    while start_date <= end_date:
        last_start = start_date
        start_date = start_date + relativedelta(days=quarter_days)
        if start_date <= end_date:  # 未到达合同结束日，不需要处理最后的残端
            #  每7日取一次fr007，季度最后一个第7日不取， 所以roll_date只能增加91-7=84天。
            df_quarter = df[df.date.between(roll_date, roll_date + relativedelta(days=quarter_days - 7))]
        else:  # 进入合同最后的残端，剩余期限不足一个季度（91天）
            # 先计算7天循环的复利，如果end_date与last_start之间少于7天，df_quarter将为空
            df_quarter = df[df.date.between(roll_date, end_date - relativedelta(days=lag + 7))]
        # 计算季度复利
        rates = df_quarter.iloc[::7, :].fr007.values
        rate_quarter = 1
        for rate in rates:  # 如果end_date与last_start之间少于7天，df_quarter将为空，不会进行此循环
            rate_quarter = rate_quarter * (1 + rate * 7 / 365)
        if start_date > end_date:  # 剩余期限不足一个季度（91天）时，需要确定最后少于7天的残端天数，
            residual_days = (end_date - last_start).days % 7  # 最后少于7天的残端的天数
            rate = df.loc[df.date == (end_date - relativedelta(days=lag + residual_days)), 'fr007'].iat[0]
            rate_quarter = rate_quarter * (1 + rate * residual_days / 365)
        gain = rate_quarter - 1
        cf.append(gain)
        roll_date = roll_date + relativedelta(days=quarter_days)  # 进入下一个季度，重新计算复利
    return cf


def fixed_leg_cf(roll_date, end_date, rate, lag):
    """
    计算fr007 irs固息端的应付利息。
    到期日是起息日的对年对月对日。计息基准为ACT/365
    计息区间为季度，每季度设置为固定的7 * 13 = 91天，最后一个季度的结束日有残端（一般都超过91天）。
        Args:
            roll_date(datetime): 浮息决定日（重置日）， 第一个重置日是起息日的前1天
            end_date(datetime): 假设该IRS合约的最后到期日，不一定是真正的到期日。
            rate(float): 固息
            lag(int): 重置日是起息日之前的几天。
        Returns:
            cf(list): 每个固息端付息周期的应付利息
    """
    quarter_days = 91
    cf = []
    start_date = roll_date + relativedelta(days=lag)  # roll_date（即重置日）是第一个浮息决定日, 起息日比第一个重置日迟一日
    while start_date <= end_date:
        last_start = start_date
        start_date = start_date + relativedelta(days=quarter_days)
        if start_date <= end_date:
            gain = rate * quarter_days / 365
        else:
            gain = rate * (end_date - last_start).days / 365
        cf.append(gain)

    return cf


def preprocess(file_path):
    irs_file = os.path.join(file_path, 'fr007_irs.csv')
    df = pd.read_csv(irs_file, parse_dates=['date'])
    df = df.fillna(method='ffill').dropna()
    df.fr007 = df.fr007 / 100
    df.fr007_1y = df.fr007_1y / 100
    df.fr007_5y = df.fr007_5y / 100
    return df


def compare_save(data, lag):
    """
    测试历史上fr007_5y合约在到期后或数据截止日后的浮息和固息的胜负。
        Args:
            data(DataFrame): irs原始数据
            lag(int): 重置日是起息日之前的几天。
        Returns:
            df(DataFrame): 历史测试结果
    """
    results = []
    col_names = ['roll_date', 'start_date', 'fixed_rate', 'first_fr007', 'fixed_gain', 'float_gain',
                 'diff', 'carry']
    float_win = 0
    fixed_win = 0
    sample_days = len(data) - 92  # 只测试至少存续了一个季度的合约
    for i in range(0, sample_days):
        if is_weekend(data.date.iat[i + lag]):  # 如果起息日是周末则跳过
            continue
        df = data.iloc[i:, :]
        roll = df.date.iat[0]
        start = df.date.iat[0 + lag]
        end = start + relativedelta(years=5)
        if end > data.date.iat[-1]:
            end = data.date.iat[-1]
        year = (end - start).days / 365
        fixed_rate = df.fr007_5y.iat[0 + lag]
        first_fr007 = df.fr007.iat[0]
        float_cf = float_leg_cf(df, roll, end, lag)
        fixed_cf = fixed_leg_cf(roll, end, fixed_rate, lag)
        float_gain = sum(float_cf)
        fixed_gain = sum(fixed_cf)
        if float_gain >= fixed_gain:
            float_win += 1
        else:
            fixed_win += 1
        results.append((
            roll.date(), start.date(), round(fixed_rate * 100, 4), round(first_fr007 * 100, 4),
            round(fixed_gain * 100 / year, 4), round(float_gain * 100 / year, 4),
            round((fixed_gain - float_gain) * 100 / year, 4),
            round((fixed_rate - first_fr007) * 100, 4),
        ))
    print(f'float_win: {float_win}/{sample_days} {float_win / (float_win + fixed_win)}')
    print(f'fix_win: {fixed_win}/{sample_days} {fixed_win / (float_win + fixed_win)}')
    return pd.DataFrame(results, columns=col_names)


def is_weekend(date):
    if date.weekday() < 5:
        return False
    else:
        return True


def plot(df):
    fig = px.line(df, x='start_date', y=['fixed_rate', 'float_gain'])
    fig2 = px.bar(df, x='start_date', y=['diff', 'carry'])
    fig2.update_traces(yaxis="y2")
    subfig = make_subplots(specs=[[{"secondary_y": True}]])
    subfig.add_traces(fig.data + fig2.data)
    subfig.layout.xaxis.title = "Date"
    subfig.layout.yaxis.title = "%"
    # subfig.layout.yaxis2.type = "linear"
    subfig.layout.yaxis2.title = "%"
    # recoloring is necessary otherwise lines from fig und fig2 would share each color
    # subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
    subfig.show()

def main():
    root_path = 'd:/ProjectRicequant/fxincome/'
    lag = 1  # 起息日之前的1天的fr007决定第一个7天区间的浮息
    # data = preprocess(root_path)
    # df = compare_save(data, lag)
    # df.to_csv(os.path.join(root_path, 'irs_processed.csv'), index=False, encoding='utf-8')
    df = pd.read_csv(os.path.join(root_path, 'irs_processed.csv'), parse_dates=['roll_date', 'start_date'])
    plot(df)


if __name__ == '__main__':
    main()


class TestIRS:

    @pytest.fixture(scope='class')
    def global_data(self):
        roll = datetime(2013, 1, 5)
        start = datetime(2013, 1, 6)
        end = datetime(2014, 1, 6)
        lag = 1
        float_cf = [0.008015186, 0.011685569, 0.009643186, 0.011959549, 0.000131233]
        df = preprocess('d:/ProjectRicequant/fxincome/')
        return {'roll': roll,
                'start': start,
                'end': end,
                'lag': lag,
                'float_cf': float_cf,
                'data': df}

    def test_float_leg_cf_first_year(self, global_data):
        data = global_data['data']
        roll = global_data['roll']
        end = global_data['end']
        lag = global_data['lag']
        correct_cf = global_data['float_cf']
        float_cf = float_leg_cf(data, roll, end, lag)
        for a, b in zip(float_cf, correct_cf):
            assert a == pytest.approx(b, abs=1e-6)

    def test_float_leg_cf_last_two_quarters(self, global_data):
        data = global_data['data']
        lag = global_data['lag']
        roll = datetime(2021, 9, 9)
        start = datetime(2021, 9, 10)
        end = datetime(2022, 1, 29)
        correct_cf = [0.005666544, 0.003166498]
        float_cf = float_leg_cf(data, roll, end, lag)
        for a, b in zip(float_cf, correct_cf):
            assert a == pytest.approx(b, abs=1e-6)

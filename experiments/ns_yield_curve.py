import traceback
import math
import os.path
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from pandas_profiling import ProfileReport
from financepy.utils import from_datetime
from financepy.products.bonds import *

root_path = 'd:/ProjectRicequant/fxincome/'
chinabond_ttm_labels = ['6m', '1y', '2y', '3y', '4y', '5y', '6y', '7y', '8y', '9y', '10y']
predicted_ttm_labels = ['pred_6m', 'pred_1y', 'pred_2y', 'pred_3y', 'pred_4y', 'pred_5y', 'pred_6y', 'pred_7y',
                        'pred_8y', 'pred_9y', 'pred_10y']
maturities = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def to_bond_freq(freq):
    if freq == 1:
        return FrequencyTypes.ANNUAL
    elif freq == 2:
        return FrequencyTypes.SEMI_ANNUAL
    elif freq == 4:
        return FrequencyTypes.QUARTERLY
    else:
        raise NotImplementedError("Unknown frequency")


def plot_bond_ylds(yld_date: datetime, bonds: list, ylds: list):
    plt.figure(figsize=(12, 6))
    title = yld_date.strftime('%Y%m%d')
    plt.title(f'CDB YTM @{title}')
    yld_date = from_datetime(yld_date.date())
    x = []
    for bond in bonds:
        years_to_maturity = (bond._maturity_date - yld_date) / 365
        x.append(years_to_maturity)
    bond_ylds_scaled = [n * 100 for n in ylds]
    plt.plot(x, bond_ylds_scaled, 'o')

    plt.xlabel('Time to Maturity (years)')
    plt.ylabel('Yield To Maturity (%)')
    plt.ylim((min(bond_ylds_scaled) - 0.3, max(bond_ylds_scaled) * 1.1))
    plt.grid(True)
    plt.savefig(f'./results/EXCEPTION_ytm_{title}.jpg', dpi=600)
    plt.close()


def plot_ns_params(df: pd.DataFrame):
    df = df.sort_values(by='date')
    first_date = df.date.iloc[0]
    last_date = df.date.iloc[len(df) - 1]
    title = f"{first_date.strftime('%Y%m%d')}-{last_date.strftime('%Y%m%d')}"
    df = df.rename(columns={'sum_beta1_beta2': 'beta1+beta2'})
    df.plot(kind='line', x='date', y=['beta1', 'beta2', 'beta1+beta2'], figsize=(15, 8))
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(f'./results/ns_params_{title}.jpg')
    plt.close()


def preprocess():
    source = 'cdb_valuation.csv'
    result = 'cdb_valuation_processed.csv'
    df = pd.read_csv(os.path.join(root_path, source), parse_dates=['date', 'issue_date', 'maturity_date'],
                     encoding='gb2312')
    df = df[df.coupon_type == '固定利率']
    # df = df[(df.coupon_info == '附息') | (df.coupon_info == '到期一次还本付息')]
    #  Wind的付息次数不正确，到期一次还本付息的付息次数显示为0，应修改为1
    df.loc[df.coupon_info == '到期一次还本付息', 'coupon_freq'] = 1
    df_2019_2022 = df[(df.outstanding_balance >= 800) & (df.date.dt.year >= 2019)]
    df_2017_2018 = df[(df.outstanding_balance >= 500) & (df.date.dt.year >= 2017) & (df.date.dt.year <= 2018)]
    df_2015_2016 = df[(df.outstanding_balance >= 400) & (df.date.dt.year >= 2015) & (df.date.dt.year <= 2016)]
    df_2014 = df[(df.outstanding_balance >= 350) & (df.date.dt.year == 2014)]
    df_2013 = df[(df.outstanding_balance >= 300) & (df.date.dt.year == 2013)]
    df_2010_2012 = df[(df.outstanding_balance >= 250) & (df.date.dt.year >= 2010) & (df.date.dt.year <= 2012)]

    df = pd.concat([df_2010_2012, df_2013, df_2014, df_2015_2016, df_2017_2018, df_2019_2022])
    df = df[df.bond_code.str.len() == 9]
    df = df[(df.maturity_date - df.date).dt.days / 365 <= 10]  # only bonds with maturity <= 10 years
    df.to_csv(os.path.join(root_path, result), index=False, encoding='gb2312')


def check_csv():
    # source = 'cdb_valuation_processed.csv'
    # bond_df = pd.read_csv(root_path + source, parse_dates=['date', 'issue_date', 'maturity_date'],
    #                       encoding='gb2312')
    # bond_dates = set(bond_df.date.dt.date.to_list())
    # ytm_df = pd.read_csv(root_path + 'cdb_ytm.csv', parse_dates=['date'], encoding='gb2312')
    # ytm_dates = set(ytm_df.date.dt.date.to_list())
    # difference = list(ytm_dates - bond_dates)
    # difference.sort()
    # print(difference)
    # print(len(difference))
    # df = pd.read_csv('./results/ns_params_20100104_20220729.csv', parse_dates=['date'])
    # report = ProfileReport(df)
    # report.to_file('./results/ns_params_report.html')
    df = pd.DataFrame(
        {"value": np.random.randn(36)},
        index=pd.date_range("2011-01-01", freq="M", periods=36),
    )
    table = pd.pivot_table(
        df, index=df.index.month, columns=df.index.year, values="value", aggfunc="sum"
    )
    print(df)
    print(table)


def gen_curves(fit_type: str, start: datetime.datetime, end: datetime.datetime, plot: bool) -> pd.DataFrame:
    source = 'cdb_valuation_processed.csv'
    df_src = pd.read_csv(os.path.join(root_path, source), parse_dates=['date', 'issue_date', 'maturity_date'],
                         encoding='gb2312')
    if fit_type == 'ns':
        ns_stats = pd.DataFrame(columns=['date', 'beta1', 'beta2', 'sum_beta1_beta2', 'beta3', 'tau'])
    elif fit_type == 'nss':
        nss_stats = pd.DataFrame(
            columns=['date', 'beta1', 'beta2', 'sum_beta1_beta2', 'beta3', 'beta4', 'tau1', 'tau2'])
    else:
        raise NotImplementedError("Unknown curve fit type")
    ytm_date = start
    while ytm_date <= end:
        df = df_src[df_src.date == ytm_date]
        if len(df) == 0:  # Today is not a buisness day.
            ytm_date += datetime.timedelta(days=1)
            continue
        bonds = []
        ytms = []
        for bond in df.itertuples():
            issue_date = from_datetime(bond.issue_date)
            maturity_date = from_datetime(bond.maturity_date)
            coupon = bond.coupon / 100
            freq = to_bond_freq(bond.coupon_freq)
            accrual_type = DayCountTypes.ACT_ACT_ISDA
            ytm = bond.ytm / 100
            b = Bond(issue_date, maturity_date, coupon, freq, accrual_type)
            bonds.append(b)
            ytms.append(ytm)

        try:
            if fit_type == 'ns':
                ns_bounds = [(-1, -1, -1, 1.3), (1, 1, 1, 3)]
                ns_stats = ns_fit(bonds, ns_stats, ytm_date, ytms, ns_bounds, plot=plot)
            elif fit_type == 'nss':
                nss_bounds = [(0, -1, -1, -1, 1.3, 1.3), (1, 1, 1, 1, 3, 3)]
                nss_stats = nss_fit(bonds, nss_stats, ytm_date, ytms, nss_bounds, plot=plot)
            # curve_function = CurveFitPolynomial(5)
            # fitted_curve = BondYieldCurve(eval_date, bonds, ytms, curve_function)
            # fitted_curve.plot('CDB Yield Curve by Quintic Polynomial')
            # curve_function = CurveFitBSpline()
            # fitted_curve = BondYieldCurve(eval_date, bonds, ytms, curve_function)
            # fitted_curve.plot('CDB Yield Curve by B Spline')
            # curve_function = CurveFitBSpline(4)
            # fitted_curve = BondYieldCurve(eval_date, bonds, ytms, curve_function)
            # fitted_curve.plot('CDB Yield Curve by B Spline power of 4')

        except Exception as e:
            print(f'Exception:{str(e)} {ytm_date}')
            traceback.print_exc()
            plot_bond_ylds(ytm_date, bonds, ytms)

        ytm_date += datetime.timedelta(days=1)
    start = start.strftime('%Y%m%d')
    end = end.strftime('%Y%m%d')
    if fit_type == 'ns':
        ns_stats.to_csv(os.path.join('./results/', f'ns_params_{start}_{end}.csv'), index=False, encoding='utf-8')
        return ns_stats
    elif fit_type == 'nss':
        nss_stats.to_csv(os.path.join('./results/', f'nss_params_{start}_{end}.csv'), index=False, encoding='utf-8')
        return nss_stats
    else:
        raise NotImplementedError("Unknown curve fit type")


def ns_fit(bonds, ns_stats, ytm_date, ytms, bounds, plot=False):
    curve_function = CurveFitNelsonSiegel(bounds=bounds)
    fitted_curve = BondYieldCurve(from_datetime(ytm_date), bonds, ytms, curve_function)
    beta1 = curve_function._beta1
    beta2 = curve_function._beta2
    beta3 = curve_function._beta3
    tau = curve_function._tau
    ns_stats.loc[len(ns_stats)] = [ytm_date, beta1, beta2, beta1 + beta2, beta3, tau]
    if plot:
        today = ytm_date.strftime('%Y%m%d')
        fitted_curve.plot(f'CDB Yield Curve by Nelson Siegel @{today}')
        address = f'./results/images/ns_cdb_ytm_{today}.jpg'
        plt.savefig(address, dpi=600)
        plt.close()
    return ns_stats


def nss_fit(bonds, nss_stats, ytm_date, ytms, bounds, plot=False):
    bounds = bounds
    curve_function = CurveFitNelsonSiegelSvensson(bounds=bounds)
    fitted_curve = BondYieldCurve(from_datetime(ytm_date), bonds, ytms, curve_function)
    beta1 = curve_function._beta1
    beta2 = curve_function._beta2
    beta3 = curve_function._beta3
    beta4 = curve_function._beta4
    tau1 = curve_function._tau1
    tau2 = curve_function._tau2
    nss_stats.loc[len(nss_stats)] = [ytm_date, beta1, beta2, beta1 + beta2, beta3, beta4, tau1, tau2]
    if plot:
        today = ytm_date.strftime('%Y%m%d')
        fitted_curve.plot(f'CDB Yield Curve by Nelson Siegel Svensson @{today}')
        address = f'./results/images/nss_cdb_ytm_{today}.jpg'
        plt.savefig(address, dpi=600)
        plt.close()
    return nss_stats


def curves_stats():
    df = pd.read_csv(root_path + 'cdb_ytm.csv', parse_dates=['date'], encoding='gb2312').set_index('date')
    df['slope'] = df['10y'] - df['6m']
    df['curvature'] = df['10y'] + df['6m'] - 2 * df['2y']
    df.to_csv('./results/chinabond_ytm.csv', encoding='utf-8')
    corr_df = pd.DataFrame(columns=['maturity', 'p_1M', 'p_1Y', 'p_2.5Y'])
    for col in df.columns:
        #  autocorrelation, 1M=21 trade days, 1Y=250 trade days, 2.5Y=625 trade days
        corr_df.loc[len(corr_df)] = [col, df[col].autocorr(21), df[col].autocorr(250), df[col].autocorr(625)]
    corr_df = corr_df.set_index('maturity').round(3)
    chinabond_stats = df.describe().T.round(3)
    chinabond_stats = chinabond_stats.join(corr_df)
    chinabond_stats.to_csv('./results/chinabond_stats.csv', encoding='utf-8')
    ns_df = pd.read_csv('./results/ns_params_20100104_20220729.csv', parse_dates=['date']).set_index('date')
    b1_adfuller = adfuller(ns_df.beta1.to_numpy())
    b2_adfuller = adfuller(ns_df.beta2.to_numpy())
    b3_adfuller = adfuller(ns_df.beta3.to_numpy())
    b1_adf = b1_adfuller[0]
    b2_adf = b2_adfuller[0]
    b3_adf = b3_adfuller[0]
    b1_pval = b1_adfuller[1]
    b2_pval = b2_adfuller[1]
    b3_pval = b3_adfuller[1]
    ns_stats = ns_df[['beta1', 'beta2', 'beta3']].describe().T
    #  Beta值需要放大100倍
    for col in ns_stats.columns:
        if col is not 'count':
            ns_stats[col] = ns_stats[col] * 100
    ns_stats = ns_stats.round(3)
    corr_df = pd.DataFrame(columns=['beta', 'p_1M', 'p_1Y', 'p_2.5Y'])
    for col in ['beta1', 'beta2', 'beta3']:
        #  autocorrelation, 1M=21 trade days, 1Y=250 trade days, 2.5Y=625 trade days
        corr_df.loc[len(corr_df)] = [col, ns_df[col].autocorr(21), ns_df[col].autocorr(250), ns_df[col].autocorr(625)]
    corr_df = corr_df.set_index('beta').round(3)
    ns_stats = ns_stats.join(corr_df)
    ns_stats.at['beta1', 'ADF'] = b1_adf
    ns_stats.at['beta1', 'P_Value'] = b1_pval
    ns_stats.at['beta2', 'ADF'] = b2_adf
    ns_stats.at['beta2', 'P_Value'] = b2_pval
    ns_stats.at['beta3', 'ADF'] = b3_adf
    ns_stats.at['beta3', 'P_Value'] = b3_pval
    ns_stats.to_csv('./results/ns_params_stats.csv')


def fit_stats():
    chinabond = pd.read_csv('./results/' + 'chinabond_ytm.csv', parse_dates=['date'], encoding='gb2312').set_index(
        'date')
    ns = pd.read_csv('./results/' + 'ns_params_20100104_20220729.csv', parse_dates=['date'],
                     encoding='gb2312').set_index(
        'date')
    combined = chinabond.join(ns)
    combined = combined[chinabond.columns.values.tolist() + ['beta1', 'beta2', 'beta3', 'tau']]
    for curve in combined.itertuples():
        ns_curve = CurveFitNelsonSiegel(tau=curve.tau)
        ns_curve._beta1 = curve.beta1
        ns_curve._beta2 = curve.beta2
        ns_curve._beta3 = curve.beta3
        for chinabond_label, pred_label, ttm in zip(chinabond_ttm_labels, predicted_ttm_labels, maturities):
            combined.at[curve.Index, pred_label] = round(ns_curve._interpolated_yield(ttm) * 100, 2)
    combined.to_csv('./results/fit_stat.csv')
    real_fitted_ytm = pd.DataFrame(columns=['Indicator'] + chinabond_ttm_labels)
    real_avgs = [chinabond[t].mean() for t in chinabond_ttm_labels]
    fitted_avgs = [combined[t].mean() for t in predicted_ttm_labels]
    errors = [real - fitted for real, fitted in zip(real_avgs, fitted_avgs)]
    real_fitted_ytm.loc[0] = ['Average Chinabond YTM'] + real_avgs
    real_fitted_ytm.loc[1] = ['Average Fitted YTM'] + fitted_avgs
    real_fitted_ytm.loc[2] = ['Average YTM Errors'] + errors
    real_fitted_ytm = real_fitted_ytm.round(2).set_index('Indicator').T
    real_fitted_ytm.to_csv('./results/chinabond_vs_fitted.csv')
    corr_real_beta = combined[['beta1', 'beta2', 'beta3', '10y', 'slope', 'curvature']].corr().round(2)
    corr_real_beta.to_csv('./results/corr_real_beta.csv')
    real_fitted_ytm['maturity'] = maturities
    ax = real_fitted_ytm.plot(kind='line', x='maturity', y='Average Chinabond YTM', color='red',
                              label='Average Chinabond YTM')
    real_fitted_ytm.plot(kind='scatter', x='maturity', y='Average Fitted YTM', ax=ax, style='*',
                         label='Average Fitted YTM')
    plt.xlabel('Time to Maturity (years)')
    plt.ylabel('Yield To Maturity (%)')
    plt.grid(True)
    plt.savefig(f'./results/chinabond_vs_fitted.jpg')
    plt.close()


def plot_ytm_3d_surface():
    df = pd.read_csv(root_path + 'cdb_ytm.csv', parse_dates=['date'], encoding='gb2312')
    df = df.set_index('date').sort_index()
    x = df.columns
    y = df.index
    z = df.to_numpy()
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(
        title=dict(
            text='CDB YTM between 2010 and 2022',
            x=0.5,
            y=0.9),
        scene=dict(
            xaxis=dict(
                title='maturity',
                backgroundcolor="rgb(200, 200, 230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"),
            yaxis=dict(
                title='date',
                backgroundcolor="rgb(230, 200,230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"),
            zaxis=dict(
                title='ytm',
                backgroundcolor="rgb(230, 230,200)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"),
            aspectratio=dict(
                x=1,
                y=1,
                z=0.6
            )
        )
    )

    fig.show()


def forecast():
    factors = pd.read_csv('d:/ProjectRicequant/fxincome/eco_factors_2010_202207.csv', parse_dates=['date']).set_index(
        'date')
    factors = factors[factors.index >= datetime.datetime(2015, 12, 1)]
    ns_params = pd.read_csv('./results/fit_stat.csv', parse_dates=['date']).set_index('date')
    ns_params = ns_params[['beta1', 'beta2', 'beta3']]
    ns_params = ns_params * 100
    # pd.plotting.autocorrelation_plot(ns_params.beta3)
    X = ns_params.beta1.values
    print(X)
    train, test = X[3000: len(X) - 44], X[len(X) - 44:]
    model = AutoReg(train, lags=22)
    model_fit = model.fit()
    print(f'Coefficients: {model_fit.params}')
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
    for i in range(len(predictions)):
        print(f'predicted={predictions[i]}, expected={test[i]}')
    rmse = math.sqrt(mean_squared_error(test, predictions))
    print(f'RMSE:{rmse:.3f}')
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()

    ns_params = ns_params.groupby(pd.Grouper(freq='MS')).nth(10)  # 假设每个月的第11个交易日或之后才出上个月的经济数据
    ns_params = ns_params[ns_params.index >= datetime.datetime(2015, 12, 1)]
    ppi_adf = adfuller(factors.ppi.to_numpy())
    cpi_adf = adfuller(factors.cpi.to_numpy())
    m2_adf = adfuller(factors.m2.to_numpy())
    delta_adf = adfuller(factors.delta_fin_m2.to_numpy())
    # print(f'ppi adf: {ppi_adf[0]}, p: {ppi_adf[1]}')
    # print(f'cpi adf: {cpi_adf[0]}, p: {cpi_adf[1]}')
    # print(f'm2 adf: {m2_adf[0]}, p: {m2_adf[1]}')
    # print(f'delta adf: {delta_adf[0]}, p: {delta_adf[1]}')
    df = ns_params.merge(factors, on='date')
    df['beta1_lag1'] = df.beta1.shift(-1)
    df['beta2_lag1'] = df.beta2.shift(-1)
    df['beta3_lag1'] = df.beta3.shift(-1)
    df.corr().to_csv('./results/param_factors.csv', index=True)


if __name__ == '__main__':
    # preprocess()
    check_csv()
    # gen_curves('ns', start=datetime.datetime(2010, 1, 4), end=datetime.datetime(2022, 7, 29), plot=True)
    # ns_df = pd.read_csv('./results/ns_params_20100104_20220729.csv', parse_dates=['date'], encoding='gb2312')
    # plot_ns_params(ns_df)
    # curves_stats()
    # fit_stats()
    # plot_ytm_3d_surface()
    # forecast()

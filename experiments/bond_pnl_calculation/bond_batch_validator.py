"""
债券完美拆解公式批量验证程序 - 大样本量版本
样本量扩大100倍：每只债券5000个案例，总共15000个案例

验证目标：
1. YTM变化幅度对公式偏差的影响
2. 持有时间对公式偏差的影响
3. 剩余期限对公式偏差的影响
"""
import os
import pandas as pd
import numpy as np
from financepy.utils import Date, DayCountTypes, FrequencyTypes
from financepy.products.bonds import Bond, YTMCalcType
from datetime import datetime, timedelta
from math import log
import random

random.seed(42)
np.random.seed(42)


def parse_date(date_val):
    """解析Excel中的日期"""
    if pd.isna(date_val):
        return None
    if isinstance(date_val, datetime):
        return date_val.date()
    if isinstance(date_val, str):
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"]:
            try:
                return datetime.strptime(date_val, fmt).date()
            except ValueError:
                continue
    return None


def create_bond_from_info(row):
    """根据基础信息创建Bond对象"""
    issue_date = parse_date(row['起息日期'])
    maturity_date = parse_date(row['到期日期'])
    coupon = row['票面利率(发行时)\n[单位] %'] / 100
    freq = row['每年付息次数\n[单位] 次']

    dc_type = DayCountTypes.ACT_ACT_ICMA
    if freq == 2:
        freq_type = FrequencyTypes.SEMI_ANNUAL
    elif freq == 4:
        freq_type = FrequencyTypes.QUARTERLY
    else:
        freq_type = FrequencyTypes.ANNUAL

    bond = Bond(
        Date(issue_date.day, issue_date.month, issue_date.year),
        Date(maturity_date.day, maturity_date.month, maturity_date.year),
        coupon, freq_type, dc_type
    )
    return bond, coupon, freq, issue_date, maturity_date


def calculate_pnl_for_case(bond, buy_date, sell_date, buy_ytm, sell_ytm, maturity_date):
    """计算单个案例的真实损益和公式损益"""
    buy_dt = Date(buy_date.day, buy_date.month, buy_date.year)
    sell_dt = Date(sell_date.day, sell_date.month, sell_date.year)

    clean_price_buy = bond.clean_price_from_ytm(buy_dt, buy_ytm, YTMCalcType.CFETS)
    accrued_buy = bond.accrued_interest(buy_dt)
    full_price_buy = clean_price_buy + accrued_buy

    clean_price_sell = bond.clean_price_from_ytm(sell_dt, sell_ytm, YTMCalcType.CFETS)
    accrued_sell = bond.accrued_interest(sell_dt)
    full_price_sell = clean_price_sell + accrued_sell

    # 计算真实损益
    coupon_income = 0.0
    coupon_count = 0
    reinvestment_value = 0.0
    # 再投资收益率使用买入和卖出的平均值
    reinvest_ytm = (buy_ytm + sell_ytm) / 2
    for cpn_date, flow_amt in zip(bond.cpn_dts, bond.flow_amounts):
        if buy_dt < cpn_date <= sell_dt:
            cash_flow = flow_amt * 100
            coupon_income += cash_flow
            coupon_count += 1
            # 利息按平均YTM计算简单利息，截止卖出日（类似定期存款）
            cpn_python_date = datetime(cpn_date.y, cpn_date.m, cpn_date.d).date()
            days_reinvest = (sell_date - cpn_python_date).days
            years_reinvest = days_reinvest / 365.0
            # reinvestment_value += cash_flow * (1 + reinvest_ytm * years_reinvest)
            reinvestment_value += cash_flow

    actual_pnl = full_price_sell - full_price_buy + reinvestment_value

    # 计算公式损益
    md_buy = bond.modified_duration(buy_dt, buy_ytm, YTMCalcType.CFETS)
    md_sell = bond.modified_duration(sell_dt, sell_ytm, YTMCalcType.CFETS)

    dt_days = (sell_date - buy_date).days
    dt_years = dt_days / 365.0
    delta_r = sell_ytm - buy_ytm

    # 公式损益中的P1需要加上票息及票息再投资收益
    # full_price_sell += reinvestment_value

    avg_md_price = (full_price_buy * md_buy + full_price_sell * md_sell) / 2
    duration_term = -avg_md_price * delta_r
    income_term = (full_price_buy * log(1 + buy_ytm) + full_price_sell * log(1 + sell_ytm)) / 2 * dt_years
    formula_pnl = duration_term + income_term

    remaining_years_buy = (maturity_date - buy_date).days / 365.0
    remaining_years_sell = (maturity_date - sell_date).days / 365.0

    return {
        'full_price_buy': full_price_buy,
        'full_price_sell': full_price_sell,
        'md_buy': md_buy,
        'md_sell': md_sell,
        'dt_days': dt_days,
        'dt_years': dt_years,
        'delta_r': delta_r,
        'coupon_count': coupon_count,
        'coupon_income': coupon_income,
        'duration_term': duration_term,
        'income_term': income_term,
        'formula_pnl': formula_pnl,
        'actual_pnl': actual_pnl,
        'difference': actual_pnl - formula_pnl,
        # 如果实际损益太小，使用0.5作为分母，本金为100
        'difference_pct': (actual_pnl - formula_pnl) / abs(actual_pnl) * 100 if abs(actual_pnl) > 0.5 else (actual_pnl - formula_pnl) / 0.5 * 100,
        'remaining_years_buy': remaining_years_buy,
        'remaining_years_sell': remaining_years_sell,
    }


def generate_cases_large(bond, coupon, freq, issue_date, maturity_date, n_scenarios=5000):
    """
    大样本量生成案例
    每个场景固定生成n_scenarios个案例，包含上行和下行
    """
    scenarios = []
    today = datetime(2026, 4, 24).date()

    effective_start = max(datetime(2021, 1, 1).date(), issue_date + timedelta(days=30))
    effective_end = min(today, maturity_date - timedelta(days=30))

    if effective_start >= effective_end:
        effective_start = issue_date + timedelta(days=30)
        effective_end = min(maturity_date - timedelta(days=30), today)

    days_range = max(1, (effective_end - effective_start).days)

    # 场景1: YTM变化影响
    # 买入日期 ：在有效范围内随机选择
    # 持有时间 ：固定约2年（700-760天）
    # 买入YTM ：票面利率 ± 1%
    # 卖出YTM ：买入YTM ± 2%（随机变化）

    for i in range(n_scenarios):
        buy_offset = random.randint(0, days_range)
        buy_date = effective_start + timedelta(days=buy_offset)

        hold_days = random.randint(700, 760)
        sell_date = buy_date + timedelta(days=hold_days)
        if sell_date >= maturity_date:
            sell_date = maturity_date - timedelta(days=1)

        buy_ytm = coupon + random.uniform(-0.01, 0.01)
        buy_ytm = max(buy_ytm, 0.001)

        ytm_diff = random.uniform(-0.02, 0.02)
        sell_ytm = buy_ytm + ytm_diff
        sell_ytm = max(sell_ytm, 0.001)

        scenarios.append({
            'group': 'YTM变化影响',
            'buy_date': buy_date,
            'sell_date': sell_date,
            'buy_ytm': buy_ytm,
            'sell_ytm': sell_ytm,
            'ytm_diff_bp': round(ytm_diff * 10000, 2),
            'hold_days': (sell_date - buy_date).days,
            'hold_years': (sell_date - buy_date).days / 365.0,
        })

    # 场景2: 持有时间影响 - 包含YTM小幅波动
    # 买入日期 ：在有效范围内随机选择
    # 持有时间 ：从7天到5年（1825天）
    # 买入YTM ：票面利率 ± 1%
    # 卖出YTM ：买入YTM ± 0.05%（随机小幅波动）

    for i in range(n_scenarios):
        buy_offset = random.randint(0, days_range)
        buy_date = effective_start + timedelta(days=buy_offset)

        hold_days = random.randint(7, 1825)
        sell_date = buy_date + timedelta(days=hold_days)
        if sell_date >= maturity_date:
            sell_date = maturity_date - timedelta(days=1)

        buy_ytm = coupon + random.uniform(-0.01, 0.01)
        buy_ytm = max(buy_ytm, 0.001)

        ytm_diff = random.uniform(-0.0005, 0.0005)
        sell_ytm = buy_ytm + ytm_diff
        sell_ytm = max(sell_ytm, 0.001)

        scenarios.append({
            'group': '持有时间影响',
            'buy_date': buy_date,
            'sell_date': sell_date,
            'buy_ytm': buy_ytm,
            'sell_ytm': sell_ytm,
            'ytm_diff_bp': round(ytm_diff * 10000, 2),
            'hold_days': (sell_date - buy_date).days,
            'hold_years': (sell_date - buy_date).days / 365.0,
        })

    # 场景3: 买入时的剩余期限影响 - 包含上行和下行
    # 买入日期 ：根据目标剩余期限（1-10年）反推
    # 持有时间 ：固定约1年（350-380天）
    # 买入YTM ：票面利率 ± 1%
    # 卖出YTM ：买入YTM ± 0.05%（随机小幅波动）
    for i in range(n_scenarios):
        target_remaining = random.uniform(1, 15)
        buy_date = maturity_date - timedelta(days=int(target_remaining * 365))

        if buy_date < effective_start:
            buy_date = effective_start
        if buy_date > effective_end:
            buy_date = effective_end

        hold_days = random.randint(350, 380)
        sell_date = buy_date + timedelta(days=hold_days)
        if sell_date >= maturity_date:
            sell_date = maturity_date - timedelta(days=1)

        buy_ytm = coupon + random.uniform(-0.01, 0.01)
        buy_ytm = max(buy_ytm, 0.001)

        ytm_diff = random.uniform(-0.0005, 0.0005)
        sell_ytm = buy_ytm + ytm_diff
        sell_ytm = max(sell_ytm, 0.001)

        scenarios.append({
            'group': '剩余期限影响',
            'buy_date': buy_date,
            'sell_date': sell_date,
            'buy_ytm': buy_ytm,
            'sell_ytm': sell_ytm,
            'ytm_diff_bp': round(ytm_diff * 10000, 2),
            'hold_days': (sell_date - buy_date).days,
            'hold_years': (sell_date - buy_date).days / 365.0,
        })

    return scenarios


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_info = pd.read_excel(os.path.join(script_dir, '基础信息.xlsx'))
    df_info = df_info.dropna(subset=['证券代码'])
    df_info = df_info[df_info['证券代码'].astype(str).str.contains(r'\.IB', na=False, regex=True)]

    all_results = []
    N_SCENARIOS = 5000

    for _, row in df_info.iterrows():
        bond_code = row['证券代码']
        bond_name = row['证券简称']
        print(f"\n处理债券: {bond_code} {bond_name}")

        bond, coupon, freq, issue_date, maturity_date = create_bond_from_info(row)
        cases = generate_cases_large(bond, coupon, freq, issue_date, maturity_date, N_SCENARIOS)

        for i, case in enumerate(cases):
            result = calculate_pnl_for_case(
                bond, 
                case['buy_date'], case['sell_date'],
                case['buy_ytm'], case['sell_ytm'],
                maturity_date
            )

            pnl_direction = "盈利" if result['actual_pnl'] > 0 else "亏损" if result['actual_pnl'] < 0 else "持平"
            ytm_direction = "收益率下行" if case['sell_ytm'] < case['buy_ytm'] else "收益率上行"

            record = {
                '验证组别': case['group'],
                '债券代码': bond_code,
                '债券简称': bond_name,
                '票面利率(%)': round(coupon * 100, 2),
                '付息频率(次/年)': int(freq),
                '案例编号': i + 1,
                '买入日期': case['buy_date'].strftime('%Y-%m-%d'),
                '卖出日期': case['sell_date'].strftime('%Y-%m-%d'),
                '持有天数': case['hold_days'],
                '持有年数': round(case['hold_years'], 4),
                '买入时剩余期限(年)': round(result['remaining_years_buy'], 2),
                '卖出时剩余期限(年)': round(result['remaining_years_sell'], 2),
                '买入YTM(%)': round(case['buy_ytm'] * 100, 4),
                '卖出YTM(%)': round(case['sell_ytm'] * 100, 4),
                'YTM变化(bp)': round(case['ytm_diff_bp'], 2),
                'YTM变化方向': ytm_direction,
                '盈亏方向': pnl_direction,
                '期初全价P0': round(result['full_price_buy'], 6),
                '期末全价P1': round(result['full_price_sell'], 6),
                '期初修正久期MD0': round(result['md_buy'], 6),
                '期末修正久期MD1': round(result['md_sell'], 6),
                '期间付息次数': result['coupon_count'],
                '期间票息收入': round(result['coupon_income'], 6),
                '公式-久期项': round(result['duration_term'], 6),
                '公式-收入项': round(result['income_term'], 6),
                '真实总损益': round(result['actual_pnl'], 6),
                '公式计算总损益': round(result['formula_pnl'], 6),
                '差异(真实-公式)': round(result['difference'], 6),
                '差异比例(%)': round(result['difference_pct'], 4),
            }

            all_results.append(record)

            if (i + 1) % 1000 == 0:
                print(f"  已完成 {i+1}/{N_SCENARIOS} 个案例")

    df_result = pd.DataFrame(all_results)
    df_result = df_result.sort_values(['债券代码', '案例编号'])

    output_file = os.path.join(script_dir, 'Scenario_results.xlsx')
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_result.to_excel(writer, sheet_name='全部案例', index=False)

        group_stats = []
        for group in df_result['验证组别'].unique():
            df_group = df_result[df_result['验证组别'] == group]
            # 差异取绝对值后再统计，避免正负抵消
            abs_diff_pct = df_group['差异比例(%)'].abs()
            stats = {
                '验证组别': group,
                '案例数': len(df_group),
                '平均YTM变化(bp)': round(df_group['YTM变化(bp)'].mean(), 2),
                '平均持有天数': round(df_group['持有天数'].mean(), 1),
                '平均买入剩余期限(年)': round(df_group['买入时剩余期限(年)'].mean(), 2),
                '平均差异比例(%)': round(df_group['差异比例(%)'].mean(), 4),
                '平均绝对差异比例(%)': round(abs_diff_pct.mean(), 4),
                '差异比例标准差(%)': round(df_group['差异比例(%)'].std(), 4),
                '绝对差异比例标准差(%)': round(abs_diff_pct.std(), 4),
                '差异比例最小值(%)': round(df_group['差异比例(%)'].min(), 4),
                '差异比例最大值(%)': round(df_group['差异比例(%)'].max(), 4),
                '绝对差异比例最小值(%)': round(abs_diff_pct.min(), 4),
                '绝对差异比例最大值(%)': round(abs_diff_pct.max(), 4),
            }
            group_stats.append(stats)

        df_group_summary = pd.DataFrame(group_stats)
        df_group_summary.to_excel(writer, sheet_name='分组统计', index=False)

        bond_stats = []
        for bond_code in df_result['债券代码'].unique():
            df_bond = df_result[df_result['债券代码'] == bond_code]
            stats = {
                '债券代码': bond_code,
                '债券简称': df_bond['债券简称'].iloc[0],
                '案例数': len(df_bond),
                '盈利案例数': len(df_bond[df_bond['盈亏方向'] == '盈利']),
                '亏损案例数': len(df_bond[df_bond['盈亏方向'] == '亏损']),
                '平均差异比例(%)': round(df_bond['差异比例(%)'].mean(), 4),
                '差异比例标准差(%)': round(df_bond['差异比例(%)'].std(), 4),
            }
            bond_stats.append(stats)

        df_bond_summary = pd.DataFrame(bond_stats)
        df_bond_summary.to_excel(writer, sheet_name='债券统计', index=False)

    print(f"\n{'='*60}")
    print(f"结果已保存到: {output_file}")
    print(f"总共生成案例数: {len(all_results)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

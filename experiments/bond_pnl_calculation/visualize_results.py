"""
债券完美拆解公式验证结果可视化 - 大样本量版本
优化排版，避免重叠
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 使用engine='openpyxl'读取，避免大文件EOFError
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(os.path.join(script_dir, 'Scenario_results.xlsx'), sheet_name='全部案例', engine='openpyxl')

# ========== 定义各子图的绘制函数 ==========

def plot_1_group_comparison(ax):
    """横向对比：三个因素的平均绝对差异比例对比"""
    group_stats = df.groupby('验证组别')['差异比例(%)'].agg(['mean', 'std', 'median']).reset_index()
    group_stats_abs = df.groupby('验证组别')['差异比例(%)'].apply(lambda x: np.abs(x).mean()).reset_index()
    group_stats_abs.columns = ['验证组别', 'mean_abs']
    group_stats = group_stats.merge(group_stats_abs, on='验证组别')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(group_stats['验证组别'], group_stats['mean_abs'], yerr=group_stats['std'],
                   capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
    ax.set_ylabel('平均绝对差异比例 (%)', fontsize=14)
    ax.set_title('横向对比：三个因素对公式偏差的影响程度\n(每组15,000例 | 差异取绝对值)', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    for bar, mean_abs_val, std_val, median_val in zip(bars, group_stats['mean_abs'], group_stats['std'], group_stats['median']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 2,
                 f'绝对值均值: {mean_abs_val:.1f}%\n原始中位数: {median_val:.1f}%\n标准差: {std_val:.1f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

def plot_2_ytm_effect(ax):
    """YTM变化影响 - 散点图+趋势线"""
    df_ytm = df[df['验证组别'] == 'YTM变化影响'].copy()

    for bond in sorted(df_ytm['债券代码'].unique()):
        bond_data = df_ytm[df_ytm['债券代码'] == bond]
        ax.scatter(bond_data['YTM变化(bp)'], bond_data['差异比例(%)'],
                    alpha=0.3, s=8, label=bond)

        z = np.polyfit(bond_data['YTM变化(bp)'], bond_data['差异比例(%)'], 2)
        p = np.poly1d(z)
        x_line = np.linspace(bond_data['YTM变化(bp)'].min(), bond_data['YTM变化(bp)'].max(), 100)
        ax.plot(x_line, p(x_line), linewidth=2, linestyle='--')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('YTM变化 (bp)', fontsize=14)
    ax.set_ylabel('差异比例 (%)', fontsize=14)
    ax.set_title('YTM变化幅度 vs 公式偏差\n(持有期700-760天 | YTM变化±200bp | 含上行/下行)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=12)

def plot_3_hold_effect(ax):
    """持有时间影响 - 散点图+趋势线"""
    df_hold = df[df['验证组别'] == '持有时间影响'].copy()

    for bond in sorted(df_hold['债券代码'].unique()):
        bond_data = df_hold[df_hold['债券代码'] == bond]
        ax.scatter(bond_data['持有天数'], bond_data['差异比例(%)'],
                    alpha=0.3, s=8, label=bond)

        z = np.polyfit(bond_data['持有天数'], bond_data['差异比例(%)'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(bond_data['持有天数'].min(), bond_data['持有天数'].max(), 100)
        ax.plot(x_line, p(x_line), linewidth=2, linestyle='--')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax.set_xlabel('持有天数', fontsize=14)
    ax.set_ylabel('差异比例 (%)', fontsize=14)
    ax.set_title('持有时间 vs 公式偏差\n(持有期7-1825天 | YTM小幅波动±5bp)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=12)

def plot_4_remain_effect(ax):
    """剩余期限影响 - 散点图+趋势线"""
    df_remain = df[df['验证组别'] == '剩余期限影响'].copy()

    for bond in sorted(df_remain['债券代码'].unique()):
        bond_data = df_remain[df_remain['债券代码'] == bond]
        ax.scatter(bond_data['买入时剩余期限(年)'], bond_data['差异比例(%)'],
                    alpha=0.3, s=8, label=bond)

        z = np.polyfit(bond_data['买入时剩余期限(年)'], bond_data['差异比例(%)'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(bond_data['买入时剩余期限(年)'].min(), bond_data['买入时剩余期限(年)'].max(), 100)
        ax.plot(x_line, p(x_line), linewidth=2, linestyle='--')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax.set_xlabel('买入时剩余期限 (年)', fontsize=14)
    ax.set_ylabel('差异比例 (%)', fontsize=14)
    ax.set_title('剩余期限 vs 公式偏差\n(剩余期限1-15年 | 持有期350-380天 | YTM小幅波动±5bp)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=12)

def plot_5_boxplot(ax):
    """三只债券的差异比例分布箱线图"""
    bond_order = ['160205.IB', '200016.IB', '220208.IB']
    box_data = [df[df['债券代码'] == b]['差异比例(%)'].values for b in bond_order]

    bp = ax.boxplot(box_data, tick_labels=bond_order, patch_artist=True, notch=True)
    colors_box = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)

    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.5)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='red', linewidth=2)

    ax.set_ylabel('差异比例 (%)', fontsize=14)
    ax.set_title('三只债券的公式偏差分布对比', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=12)

def plot_6_scatter(ax):
    """持有天数 vs 差异比例 散点图（颜色=|YTM变化|）"""
    scatter2 = ax.scatter(df['持有天数'], df['差异比例(%)'],
                           c=abs(df['YTM变化(bp)']), cmap='plasma', alpha=0.4, s=15)
    cbar2 = plt.colorbar(scatter2, ax=ax)
    cbar2.set_label('|YTM变化| (bp)', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax.set_xlabel('持有天数', fontsize=14)
    ax.set_ylabel('差异比例 (%)', fontsize=14)
    ax.set_title('散点图：持有时间 vs 公式偏差\n(全样本 | 颜色=|YTM变化|)', fontsize=16, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=12)

def plot_7_heatmap(ax):
    """综合热力图：YTM变化 × 持有天数"""
    pivot_data = df.pivot_table(
        values='差异比例(%)',
        index=pd.cut(df['YTM变化(bp)'], bins=[-250, -100, -50, 0, 50, 100, 250]),
        columns=pd.cut(df['持有天数'], bins=[0, 30, 180, 365, 730, 2000]),
        aggfunc='mean',
        observed=False
    )

    im = ax.imshow(pivot_data.values, cmap='RdYlBu_r', aspect='auto')
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels([f'{int(c.left)}-{int(c.right)}' for c in pivot_data.columns], rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels([f'{int(i.left)}-{int(i.right)}' for i in pivot_data.index], fontsize=10)
    ax.set_xlabel('持有天数区间', fontsize=14)
    ax.set_ylabel('YTM变化区间 (bp)', fontsize=14)
    ax.set_title('热力图：YTM变化 × 持有天数 对偏差的影响', fontsize=16, fontweight='bold', pad=20)
    cbar3 = plt.colorbar(im, ax=ax)
    cbar3.set_label('平均差异比例 (%)', fontsize=12)

    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        color='white' if abs(val) > 20 else 'black', fontsize=10, fontweight='bold')


# ========== 绘制组合图 ==========
fig = plt.figure(figsize=(24, 32))

ax1 = plt.subplot2grid((4, 2), (0, 0))
plot_1_group_comparison(ax1)

ax2 = plt.subplot2grid((4, 2), (0, 1))
plot_2_ytm_effect(ax2)

ax3 = plt.subplot2grid((4, 2), (1, 0))
plot_3_hold_effect(ax3)

ax4 = plt.subplot2grid((4, 2), (1, 1))
plot_4_remain_effect(ax4)

ax5 = plt.subplot2grid((4, 2), (2, 0))
plot_5_boxplot(ax5)

ax6 = plt.subplot2grid((4, 2), (2, 1))
plot_6_scatter(ax6)

ax7 = plt.subplot2grid((4, 2), (3, 0), colspan=2)
plot_7_heatmap(ax7)

plt.tight_layout(pad=4.0, h_pad=4.0, w_pad=3.0)
plt.savefig(os.path.join(script_dir, 'Scenario_Analysis.png'), dpi=300, bbox_inches='tight')
print("组合图表已保存到: Scenario_Analysis.png")

plt.show()


# ========== 单独保存每个子图 ==========
subplots_config = [
    (plot_1_group_comparison, '01_横向对比_三个因素平均绝对差异.png', (12, 8)),
    (plot_2_ytm_effect, '02_YTM变化影响.png', (12, 8)),
    (plot_3_hold_effect, '03_持有时间影响.png', (12, 8)),
    (plot_4_remain_effect, '04_剩余期限影响.png', (12, 8)),
    (plot_5_boxplot, '05_三只债券差异分布箱线图.png', (12, 8)),
    (plot_6_scatter, '06_持有天数vs差异比例散点图.png', (12, 8)),
    (plot_7_heatmap, '07_YTM变化x持有天数热力图.png', (16, 10)),
]

for plot_func, filename, figsize in subplots_config:
    fig_single, ax_single = plt.subplots(figsize=figsize)
    plot_func(ax_single)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, filename), dpi=300, bbox_inches='tight')
    print(f"子图已保存到: {filename}")
    plt.close(fig_single)

print("\n所有子图保存完成！")

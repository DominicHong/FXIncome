"""
债券完美拆解公式验证结果可视化 - 大样本量版本
优化排版，避免重叠
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 使用engine='openpyxl'读取，避免大文件EOFError
df = pd.read_excel('债券完美拆解公式验证结果_large.xlsx', sheet_name='全部案例', engine='openpyxl')

fig = plt.figure(figsize=(24, 32))

# ========== 1. 横向对比：三个因素的平均绝对差异比例对比 ==========
ax1 = plt.subplot(4, 2, 1)
# 使用绝对值差异进行统计，避免正负抵消
group_stats = df.groupby('验证组别')['差异比例(%)'].agg(['mean', 'std', 'median']).reset_index()
group_stats_abs = df.groupby('验证组别')['差异比例(%)'].apply(lambda x: np.abs(x).mean()).reset_index()
group_stats_abs.columns = ['验证组别', 'mean_abs']

# 合并数据
group_stats = group_stats.merge(group_stats_abs, on='验证组别')

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax1.bar(group_stats['验证组别'], group_stats['mean_abs'], yerr=group_stats['std'],
               capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
ax1.set_ylabel('平均绝对差异比例 (%)', fontsize=14)
ax1.set_title('横向对比：三个因素对公式偏差的影响程度\n(样本量: 15,000个案例 | 差异取绝对值)', fontsize=16, fontweight='bold', pad=20)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

for bar, mean_abs_val, std_val, median_val in zip(bars, group_stats['mean_abs'], group_stats['std'], group_stats['median']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std_val + 2,
             f'绝对值均值: {mean_abs_val:.1f}%\n原始中位数: {median_val:.1f}%\n标准差: {std_val:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

# ========== 2. YTM变化影响 - 散点图+趋势线 ==========
ax2 = plt.subplot(4, 2, 2)
df_ytm = df[df['验证组别'] == 'YTM变化影响'].copy()

for bond in sorted(df_ytm['债券代码'].unique()):
    bond_data = df_ytm[df_ytm['债券代码'] == bond]
    ax2.scatter(bond_data['YTM变化(bp)'], bond_data['差异比例(%)'],
                alpha=0.3, s=8, label=bond)

    z = np.polyfit(bond_data['YTM变化(bp)'], bond_data['差异比例(%)'], 2)
    p = np.poly1d(z)
    x_line = np.linspace(bond_data['YTM变化(bp)'].min(), bond_data['YTM变化(bp)'].max(), 100)
    ax2.plot(x_line, p(x_line), linewidth=2, linestyle='--')

ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('YTM变化 (bp)', fontsize=14)
ax2.set_ylabel('差异比例 (%)', fontsize=14)
ax2.set_title('YTM变化幅度 vs 公式偏差\n(固定持有期≈2年 | 包含上行和下行)', fontsize=16, fontweight='bold', pad=20)
ax2.legend(fontsize=11, loc='best')
ax2.grid(alpha=0.3, linestyle='--')
ax2.tick_params(axis='both', labelsize=12)

# ========== 3. 持有时间影响 - 散点图+趋势线 ==========
ax3 = plt.subplot(4, 2, 3)
df_hold = df[df['验证组别'] == '持有时间影响'].copy()

for bond in sorted(df_hold['债券代码'].unique()):
    bond_data = df_hold[df_hold['债券代码'] == bond]
    ax3.scatter(bond_data['持有天数'], bond_data['差异比例(%)'],
                alpha=0.3, s=8, label=bond)

    z = np.polyfit(bond_data['持有天数'], bond_data['差异比例(%)'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(bond_data['持有天数'].min(), bond_data['持有天数'].max(), 100)
    ax3.plot(x_line, p(x_line), linewidth=2, linestyle='--')

ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
ax3.set_xlabel('持有天数', fontsize=14)
ax3.set_ylabel('差异比例 (%)', fontsize=14)
ax3.set_title('持有时间 vs 公式偏差\n(固定YTM变化=±50bp | 包含上行和下行)', fontsize=16, fontweight='bold', pad=20)
ax3.legend(fontsize=11, loc='best')
ax3.grid(alpha=0.3, linestyle='--')
ax3.tick_params(axis='both', labelsize=12)

# ========== 4. 剩余期限影响 - 散点图+趋势线 ==========
ax4 = plt.subplot(4, 2, 4)
df_remain = df[df['验证组别'] == '剩余期限影响'].copy()

for bond in sorted(df_remain['债券代码'].unique()):
    bond_data = df_remain[df_remain['债券代码'] == bond]
    ax4.scatter(bond_data['买入时剩余期限(年)'], bond_data['差异比例(%)'],
                alpha=0.3, s=8, label=bond)

    z = np.polyfit(bond_data['买入时剩余期限(年)'], bond_data['差异比例(%)'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(bond_data['买入时剩余期限(年)'].min(), bond_data['买入时剩余期限(年)'].max(), 100)
    ax4.plot(x_line, p(x_line), linewidth=2, linestyle='--')

ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
ax4.set_xlabel('买入时剩余期限 (年)', fontsize=14)
ax4.set_ylabel('差异比例 (%)', fontsize=14)
ax4.set_title('剩余期限 vs 公式偏差\n(固定YTM变化=±50bp, 持有期≈1年 | 包含上行和下行)', fontsize=16, fontweight='bold', pad=20)
ax4.legend(fontsize=11, loc='best')
ax4.grid(alpha=0.3, linestyle='--')
ax4.tick_params(axis='both', labelsize=12)

# ========== 5. 三只债券的差异比例分布箱线图 ==========
ax5 = plt.subplot(4, 2, 5)
bond_order = ['160205.IB', '200016.IB', '220208.IB']
box_data = [df[df['债券代码'] == b]['差异比例(%)'].values for b in bond_order]

bp = ax5.boxplot(box_data, labels=bond_order, patch_artist=True, notch=True)
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

ax5.set_ylabel('差异比例 (%)', fontsize=14)
ax5.set_title('三只债券的公式偏差分布对比', fontsize=16, fontweight='bold', pad=20)
ax5.grid(axis='y', alpha=0.3, linestyle='--')
ax5.tick_params(axis='both', labelsize=12)

# ========== 6. YTM变化 vs 差异比例 散点图（颜色=持有天数） ==========
ax6 = plt.subplot(4, 2, 6)
scatter = ax6.scatter(df['YTM变化(bp)'], df['差异比例(%)'],
                      c=df['持有天数'], cmap='viridis', alpha=0.4, s=15)
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label('持有天数', fontsize=12)
ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
ax6.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax6.set_xlabel('YTM变化 (bp)', fontsize=14)
ax6.set_ylabel('差异比例 (%)', fontsize=14)
ax6.set_title('散点图：YTM变化 vs 公式偏差\n(颜色=持有天数)', fontsize=16, fontweight='bold', pad=20)
ax6.grid(alpha=0.3, linestyle='--')
ax6.tick_params(axis='both', labelsize=12)

# ========== 7. 持有天数 vs 差异比例 散点图（颜色=|YTM变化|） ==========
ax7 = plt.subplot(4, 2, 7)
scatter2 = ax7.scatter(df['持有天数'], df['差异比例(%)'],
                       c=abs(df['YTM变化(bp)']), cmap='plasma', alpha=0.4, s=15)
cbar2 = plt.colorbar(scatter2, ax=ax7)
cbar2.set_label('|YTM变化| (bp)', fontsize=12)
ax7.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
ax7.set_xlabel('持有天数', fontsize=14)
ax7.set_ylabel('差异比例 (%)', fontsize=14)
ax7.set_title('散点图：持有时间 vs 公式偏差\n(颜色=|YTM变化|)', fontsize=16, fontweight='bold', pad=20)
ax7.grid(alpha=0.3, linestyle='--')
ax7.tick_params(axis='both', labelsize=12)

# ========== 8. 综合热力图：YTM变化 × 持有天数 ==========
ax8 = plt.subplot(4, 2, 8)

pivot_data = df.pivot_table(
    values='差异比例(%)',
    index=pd.cut(df['YTM变化(bp)'], bins=[-250, -100, -50, 0, 50, 100, 250]),
    columns=pd.cut(df['持有天数'], bins=[0, 30, 180, 365, 730, 2000]),
    aggfunc='mean',
    observed=False
)

im = ax8.imshow(pivot_data.values, cmap='RdYlBu_r', aspect='auto')
ax8.set_xticks(range(len(pivot_data.columns)))
ax8.set_xticklabels([f'{int(c.left)}-{int(c.right)}' for c in pivot_data.columns], rotation=45, ha='right', fontsize=10)
ax8.set_yticks(range(len(pivot_data.index)))
ax8.set_yticklabels([f'{int(i.left)}-{int(i.right)}' for i in pivot_data.index], fontsize=10)
ax8.set_xlabel('持有天数区间', fontsize=14)
ax8.set_ylabel('YTM变化区间 (bp)', fontsize=14)
ax8.set_title('热力图：YTM变化 × 持有天数 对偏差的影响', fontsize=16, fontweight='bold', pad=20)
cbar3 = plt.colorbar(im, ax=ax8)
cbar3.set_label('平均差异比例 (%)', fontsize=12)

for i in range(len(pivot_data.index)):
    for j in range(len(pivot_data.columns)):
        val = pivot_data.values[i, j]
        if not np.isnan(val):
            ax8.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    color='white' if abs(val) > 20 else 'black', fontsize=10, fontweight='bold')

plt.tight_layout(pad=4.0, h_pad=4.0, w_pad=3.0)
plt.savefig('债券公式偏差分析图表_large.png', dpi=300, bbox_inches='tight')
print("图表已保存到: 债券公式偏差分析图表_large.png")

plt.show()

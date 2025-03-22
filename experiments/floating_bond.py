import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, summation

# 定义符号
r1, r2, yd, s, n, f = symbols('r1 r2 yd s n f')
i = symbols('i', integer=True)

# 定义表达式
term = ((r2 + s)/f) / ((1 + (r2 + yd)/f) ** i)
term_first = ((r1 + s)/f) / (1 + (r2 + yd)/f)  # 单独处理第一项
term_last = (1 + (r2 + s)/f) / ((1 + (r2 + yd)/f) ** n)  # 单独处理最后一项

# 求和公式
PV = term_first + summation(term, (i, 2, n - 1)) + term_last

# 对 PV 关于 R2 求偏导数
rate_duration_formula = diff(PV, r2)

# 对 PV 关于 yd 求偏导数
spread_duration_formula = diff(PV, yd)

# 定义 ttm 和 f 的取值范围
ttm_values = range(1, 11)
f_values = [1, 4, 12, 52]

# 初始化列表来存储结果
rate_duration_results = []
spread_duration_results = []

R1=0.036
R2=0.036
S=-0.0153
Yd=-0.0153
fair_full_price = 1.000066

# 计算每个组合的 rate_duration 和 spread_duration
for ttm in ttm_values:
    for f_value in f_values:
        n_value = ttm * f_value
        rate_duration = rate_duration_formula.subs({r1: R1, r2: R2, s: S, yd: Yd, n: n_value, f: f_value}).evalf()
        spread_duration = spread_duration_formula.subs({r1: R1, r2: R2, s: S, yd: Yd, n: n_value, f: f_value}).evalf()
        rate_duration_results.append((ttm, f_value, rate_duration / -fair_full_price))
        spread_duration_results.append((ttm, f_value, spread_duration / -fair_full_price))

# Convert results to numpy arrays for plotting
rate_duration_results = np.array(rate_duration_results)
spread_duration_results = np.array(spread_duration_results)

# Create 3D plot
fig = plt.figure(figsize=(14, 7))

# Plot rate_duration
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(rate_duration_results[:, 0], rate_duration_results[:, 1], rate_duration_results[:, 2], c='r', marker='o')
ax1.set_xlabel('TTM')
ax1.set_ylabel('F')
ax1.set_zlabel('Rate Duration')
ax1.set_title('Rate Duration vs TTM and F')

# Plot spread_duration
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(spread_duration_results[:, 0], spread_duration_results[:, 1], spread_duration_results[:, 2], c='b', marker='^')
ax2.set_xlabel('TTM')
ax2.set_ylabel('F')
ax2.set_zlabel('Spread Duration')
ax2.set_title('Spread Duration vs TTM and F')

plt.show()
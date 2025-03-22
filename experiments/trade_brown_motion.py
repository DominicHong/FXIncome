import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


def generate_bond_prices(S0=100, mu=0.02, sigma=0.1, days=252, num_paths=1):
    """
    生成债券价格时间序列

    参数:
    S0: 初始价格
    mu: 年化预期收益率
    sigma: 年化波动率
    days: 交易日数量
    num_paths: 模拟路径数量

    返回:
    prices_df: 包含时间序列的DataFrame
    """
    # 将年化参数转换为日度参数
    dt = 1 / 252  # 按照每年252个交易日计算
    mu_daily = mu * dt
    sigma_daily = sigma * np.sqrt(dt)

    # 生成时间序列
    t = np.arange(days)

    # 生成布朗运动增量
    dW = np.random.normal(0, 1, size=(num_paths, days - 1)) * np.sqrt(dt)

    # 计算价格路径
    prices = np.zeros((num_paths, days))
    prices[:, 0] = S0

    for t in range(1, days):
        prices[:, t] = prices[:, t - 1] * np.exp(
            (mu_daily - 0.5 * sigma_daily**2) + sigma_daily * dW[:, t - 1]
        )

    # 创建DataFrame
    dates = pd.date_range(start="today", periods=days, freq="B")
    prices_df = pd.DataFrame(
        prices.T, index=dates, columns=[f"path_{i+1}" for i in range(num_paths)]
    )
    print(prices_df.head())
    return prices_df


def plot_price_paths(prices_df):
    """绘制价格路径图"""
    plt.figure(figsize=(12, 6))
    prices_df.plot(title="Bond Price Simulation - Geometric Brownian Motion")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()


def verify_lognormal_distribution(prices_df):
    """验证对数正态分布"""
    # 选择最后一个时间点的价格进行分析
    final_prices = prices_df.iloc[-1]

    # 创建对数收益率的直方图
    plt.figure(figsize=(12, 6))

    # 绘制直方图和核密度估计
    sns.histplot(data=np.log(final_prices), stat="density", kde=True)

    # 拟合正态分布
    mu, std = stats.norm.fit(np.log(final_prices))
    x = np.linspace(np.log(final_prices).min(), np.log(final_prices).max(), 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, "r-", lw=2, label="Normal Distribution Fit")

    plt.title("Log-normal Distribution Test - Final Prices")
    plt.xlabel("Log Price")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 进行Shapiro-Wilk正态性检验
    statistic, p_value = stats.shapiro(np.log(final_prices))
    print(f"\nShapiro-Wilk正态性检验结果:")
    print(f"统计量: {statistic:.4f}")
    print(f"P值: {p_value:.4f}")
    print(f"结论: {'服从正态分布' if p_value > 0.05 else '不服从正态分布'}")


if __name__ == "__main__":
    # 设置随机种子以保证结果可重复
    np.random.seed(42)

    # 生成100条价格路径
    prices_df = generate_bond_prices(
        S0=100, mu=0.02, sigma=0.1, days=252, num_paths=5
    )

    # 绘制价格路径
    plot_price_paths(prices_df)

    # 验证对数正态分布
    verify_lognormal_distribution(prices_df)

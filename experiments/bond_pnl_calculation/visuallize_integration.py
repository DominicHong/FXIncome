
"""
债券损益积分过程可视化动画
根据《定积分.md》文档实现
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import font_manager
from math import log
import os

plt.rcParams['animation.html'] = 'html5'
plt.rcParams['axes.unicode_minus'] = False


def set_chinese_font():
    """设置中文字体"""
    try:
        font_path = 'C:/Windows/Fonts/msyh.ttc'
        font_prop = font_manager.FontProperties(fname=font_path, size=12)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        return font_prop
    except:
        try:
            font_path = 'C:/Windows/Fonts/simhei.ttf'
            font_prop = font_manager.FontProperties(fname=font_path, size=12)
            plt.rcParams['font.sans-serif'] = ['SimHei']
            return font_prop
        except:
            return None


font_prop = set_chinese_font()


class BondPriceFunction:
    """
    债券连续价格函数 P_cont(r, t)
    根据《定积分.md》文档实现
    """
    
    def __init__(self, face_value=100, coupon_rate=0.03, maturity_years=10, freq=1):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity_years = maturity_years
        self.freq = freq
        self.coupon = face_value * coupon_rate / freq
        self.n_periods = maturity_years * freq
    
    def P_cont(self, r, t):
        """
        计算连续价格 P_cont(r, t)
        公式: P_cont = sum_{i=1 to n} [C_i / (1 + r)^{T_i - t}]
        """
        price = 0.0
        for i in range(1, self.n_periods + 1):
            T_i = self.maturity_years - (self.maturity_years / self.n_periods * (i - 1))
            r_annual = r
            exponent = T_i - t
            discount = (1 + r_annual) ** exponent
            
            if i == self.n_periods:
                cash_flow = self.coupon + self.face_value
            else:
                cash_flow = self.coupon
            
            price += cash_flow / discount
        
        return price
    
    def dP_dr(self, r, t):
        """
        计算 ∂P/∂r = -P * MD
        """
        P = self.P_cont(r, t)
        MD = self.modified_duration(r, t)
        return -P * MD
    
    def dP_dt(self, r, t):
        """
        计算 ∂P/∂t = P * ln(1 + r)
        """
        P = self.P_cont(r, t)
        return P * log(1 + r)
    
    def modified_duration(self, r, t):
        """计算修正久期 MD"""
        P = self.P_cont(r, t)
        if P <= 0:
            return 0.0
        
        duration = 0.0
        for i in range(1, self.n_periods + 1):
            T_i = self.maturity_years - (self.maturity_years / self.n_periods * (i - 1))
            r_annual = r
            exponent = T_i - t
            discount = (1 + r_annual) ** exponent
            
            if i == self.n_periods:
                cash_flow = self.coupon + self.face_value
            else:
                cash_flow = self.coupon
            
            duration += cash_flow / discount * (T_i - t) / (1 + r_annual)
        
        return duration / P


def create_animation():
    """创建债券损益积分过程的三维动画"""
    
    # 初始化债券
    bond = BondPriceFunction(
        face_value=100, 
        coupon_rate=0.03, 
        maturity_years=10, 
        freq=1
    )
    
    # 设置起点和终点
    r0 = 0.02  # 期初YTM
    t0 = 0.0   # 期初时间
    r1 = 0.05  # 期末YTM
    t1 = 2.0   # 期末时间
    
    # 绘制曲面的范围
    r_min = 0.0
    r_max = 0.08
    t_min = 0.0
    t_max = 4.0
    
    # 创建网格
    r_grid = np.linspace(r_min, r_max, 50)
    t_grid = np.linspace(t_min, t_max, 50)
    R, T = np.meshgrid(r_grid, t_grid)
    
    # 计算曲面值
    P_surface = np.zeros_like(R)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            P_surface[i, j] = bond.P_cont(R[i, j], T[i, j])
    
    # 创建图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面
    surf = ax.plot_surface(
        R, T, P_surface, 
        cmap='viridis', 
        alpha=0.6, 
        edgecolor='none', 
        antialiased=True
    )
    
    # 颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('全价 P', fontproperties=font_prop)
    
    # 创建直线路径
    s = np.linspace(0, 1, 100)  # 参数 s 从 0 到 1
    r_line = r0 + s * (r1 - r0)
    t_line = t0 + s * (t1 - t0)
    p_line = np.zeros_like(s)
    
    for i in range(len(s)):
        p_line[i] = bond.P_cont(r_line[i], t_line[i])
    
    # 找到曲面的最低Z值，用于绘制XY平面投影
    z_min = P_surface.min() - 2
    
    # 初始化图形元素
    line_plot, = ax.plot([], [], [], 'r-', linewidth=3, label='曲面上的路径')
    line_projection, = ax.plot([], [], [], 'c--', linewidth=2, label='XY平面直线路径投影')
    point_start, = ax.plot([], [], [], 'go', markersize=10, label='起点 (r0, t0)')
    point_end, = ax.plot([], [], [], 'bo', markersize=10, label='终点 (r1, t1)')
    moving_point, = ax.plot([], [], [], 'ro', markersize=8, label='当前点')
    moving_point_proj, = ax.plot([], [], [], 'co', markersize=6, label='当前点投影')
    integral_line, = ax.plot([], [], [], 'orange', linewidth=2, alpha=0.8)
    
    # 设置轴标签和标题
    ax.set_xlabel('YTM (r)', fontproperties=font_prop, fontsize=12)
    ax.set_ylabel('时间 (t)', fontproperties=font_prop, fontsize=12)
    ax.set_zlabel('全价 P', fontproperties=font_prop, fontsize=12)
    ax.set_title(
        '债券损益的积分过程: ΔP = ∫(∂P/∂r dr + ∂P/∂t dt)', 
        fontproperties=font_prop, 
        fontsize=14, 
        pad=20
    )
    ax.legend(prop=font_prop)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 添加信息文本
    text_info = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, 
                          fontproperties=font_prop, fontsize=10)
    
    # 初始化函数
    def init():
        line_plot.set_data([], [])
        line_plot.set_3d_properties([])
        
        line_projection.set_data([], [])
        line_projection.set_3d_properties([])
        
        point_start.set_data([], [])
        point_start.set_3d_properties([])
        
        point_end.set_data([], [])
        point_end.set_3d_properties([])
        
        moving_point.set_data([], [])
        moving_point.set_3d_properties([])
        
        moving_point_proj.set_data([], [])
        moving_point_proj.set_3d_properties([])
        
        integral_line.set_data([], [])
        integral_line.set_3d_properties([])
        
        text_info.set_text('')
        
        return (line_plot, line_projection, point_start, point_end, 
                moving_point, moving_point_proj, integral_line, text_info)
    
    # 更新函数
    def update(frame):
        idx = frame
        
        # 更新路径
        line_plot.set_data(r_line[:idx+1], t_line[:idx+1])
        line_plot.set_3d_properties(p_line[:idx+1])
        
        # 更新投影路径
        line_projection.set_data(r_line[:idx+1], t_line[:idx+1])
        line_projection.set_3d_properties([z_min] * (idx+1))
        
        # 更新起点和终点
        p0 = bond.P_cont(r0, t0)
        p1 = bond.P_cont(r1, t1)
        
        point_start.set_data([r0], [t0])
        point_start.set_3d_properties([p0])
        
        point_end.set_data([r1], [t1])
        point_end.set_3d_properties([p1])
        
        # 更新当前点
        moving_point.set_data([r_line[idx]], [t_line[idx]])
        moving_point.set_3d_properties([p_line[idx]])
        
        # 更新当前点投影
        moving_point_proj.set_data([r_line[idx]], [t_line[idx]])
        moving_point_proj.set_3d_properties([z_min])
        
        # 更新积分路径显示
        integral_line.set_data(r_line[:idx+1], t_line[:idx+1])
        integral_line.set_3d_properties(p_line[:idx+1])
        
        # 计算当前状态
        delta_r = r_line[idx] - r0
        delta_t = t_line[idx] - t0
        current_p = p_line[idx] - p0
        
        # 计算近似的公式值（梯形法则）
        if idx > 0:
            # 使用当前点和前一个点计算
            r_prev, t_prev = r_line[idx-1], t_line[idx-1]
            r_curr, t_curr = r_line[idx], t_line[idx]
            
            md_prev = bond.modified_duration(r_prev, t_prev)
            md_curr = bond.modified_duration(r_curr, t_curr)
            
            p_prev = bond.P_cont(r_prev, t_prev)
            p_curr = bond.P_cont(r_curr, t_curr)
            
            dr = r_curr - r_prev
            dt = t_curr - t_prev
            
            duration_term = - (p_prev * md_prev + p_curr * md_curr) / 2 * dr
            income_term = (p_prev * log(1 + r_prev) + p_curr * log(1 + r_curr)) / 2 * dt
            formula_delta_p = duration_term + income_term
        else:
            formula_delta_p = 0.0
        
        # 准备信息文本
        info_text = (
            f'当前进度: {idx/len(s)*100:.1f}%\n'
            f'当前位置: r={r_line[idx]:.4f}, t={t_line[idx]:.2f}\n'
            f'累计变化: Δr={delta_r:.4f}, Δt={delta_t:.2f}\n'
            f'真实累计损益: ΔP={current_p:.4f}\n'
            f'公式近似损益: ΔP={formula_delta_p:.4f}'
        )
        
        text_info.set_text(info_text)
        
        return (line_plot, line_projection, point_start, point_end, 
                moving_point, moving_point_proj, integral_line, text_info)
    
    # 创建动画
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(s), 
        init_func=init, 
        interval=50, 
        blit=True
    )
    
    plt.tight_layout()
    
    return anim, fig


if __name__ == '__main__':
    anim, fig = create_animation()
    
    # 保存 GIF 到本地
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(script_dir, 'bond_integration_animation.gif')
    
    print(f"正在保存动画到: {gif_path}")
    print("这可能需要几分钟时间，请稍候...")
    
    # 使用 PillowWriter 保存
    writer = PillowWriter(fps=20, metadata=dict(artist='FXIncome'), bitrate=1800)
    anim.save(gif_path, writer=writer, dpi=100)
    
    print(f"动画已成功保存到: {gif_path}")
    


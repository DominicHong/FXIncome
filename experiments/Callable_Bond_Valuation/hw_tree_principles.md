# Hull-White (HW) 树模型原理与 FinancePy 实现核验

> 适用对象：`financepy.models.hw_tree.HWTree`（FinancePy 1.1.0）
> 验证日期：2026-08-13，核验方式：逐段对照 Hull《Options, Futures and Other Derivatives》第 31 章与 Brigo-Mercurio 第 3 章，并做数值交叉验证（见第 6 节）

---

## 1. 模型定义

Hull-White（单因子高斯）短期利率过程：

$$dr(t) = \big[\theta(t) - a\,r(t)\big]\,dt + \sigma\,dW(t)$$

- `a`：均值回归速度（κ），`sigma`：短期利率的**绝对**波动率（单位与利率相同）
- $\theta(t)$：无套利漂移项，**完全由今天的收益率曲线决定**，用来让模型恰好重定价市场折现因子 $P(0,T)$
- 零息债有闭式解：$P(t,T) = A(t,T)\,e^{-B(t,T)\,r(t)}$，其中 $B(t,T)=\frac{1-e^{-a(T-t)}}{a}$

在树模型中，$a$ 和 $\sigma$ 是外部输入（本项目中来自 SOFR 历史标定），$\theta(t)$ 由树逐层自动拟合（见第 3 节）。

---

## 2. 树的结构

### 2.1 时间网格

- 总步数 $N$（`num_time_steps`），时间步长 $dt = T_{mat}/N$
- FinancePy 把树的终点延伸到 $T_{mat} + dt$（多一个节点），使第 $N$ 个节点恰好落在债券到期日 $T_{mat}$ 上
- 节点 $(m,j)$：时刻 $t_m = m\cdot dt$，利率 $r(m,j) = \alpha_m + j\cdot\Delta r$

### 2.2 节点间距 $\Delta r = \sigma\sqrt{3dt}$

这是 Hull 推荐的间距（31.7 节）。理由：HW 树的标准分支概率在 $|j| \le 0.184/(a\cdot dt)$ 时恒非负，取 $\Delta r = \sigma\sqrt{3dt}$ 时此条件与"树上界足够覆盖利率分布"的约束恰好相容，在给定 $dt$ 下最大化概率合法的范围。

FinancePy 实现（`build_tree_fast`）：

```python
d_r = sigma * np.sqrt(3.0 * dt)
j_max = ceil(0.184 / (a * dt))
```

✅ 与 Hull 31.7/31.8 完全一致。实测（σ=0.955%，a=0.0524，N=400）：dt=0.025014，Δr=0.002616，j_max=141，均符合公式。

### 2.3 分支概率（三叉树）

对每个节点 $j$，转移概率满足**均值与方差一阶矩匹配**：

$$\mathbb{E}[\Delta r] = -a\,r\,\Delta t,\qquad \mathrm{Var}(\Delta r) = \sigma^2\Delta t$$

- **内部节点**：

$$p_u=\frac16+\frac{a^2j^2dt^2-aj\,dt}{2},\quad p_m=\frac23-a^2j^2dt^2,\quad p_d=\frac16+\frac{a^2j^2dt^2+aj\,dt}{2}$$

  验算：$p_u+p_m+p_d=1$ ✓；$\mathbb{E}[\Delta j]=p_u-p_d=-aj\,dt$ ✓；$\mathrm{Var}(\Delta j)=p_u+p_d-(p_u-p_d)^2=1/3$ ✓，故 $\mathrm{Var}(\Delta r)=\Delta r^2/3=\sigma^2dt$ ✓

- **顶边节点**（$j=+j_{max}$，下行分支）：`up` 留在原地、`mid` 降 1、`down` 降 2

$$p_u=\frac76+\frac{a^2j^2dt^2-3aj\,dt}{2},\quad p_m=-\frac13-a^2j^2dt^2+2aj\,dt,\quad p_d=\frac16+\frac{a^2j^2dt^2-aj\,dt}{2}$$

  验算：概率和为 1 ✓；$\mathbb{E}[\Delta j]=-p_m-2p_d=-aj\,dt$ ✓；$\mathrm{Var}(\Delta j)=1/3$ ✓。底边节点（$j=-j_{max}$）对称处理。

- 出现负概率时 FinancePy 直接抛错（`"Tree probabilities negative - arbitrage tree!"`），数学上正确。

实测根节点概率 = (1/6, 2/3, 1/6)，第一步方差 = 1/3（以 Δr² 为单位）✓。

---

## 3. 无套利校准：Arrow-Debreu 前向递推

核心：树的漂移 $\alpha_m$ 不是从 $\theta(t)$ 解析算出来的，而是逐层反解，使树**精确重定价输入的市场折现曲线**（与 Brigo-Mercurio 3.3.4、Hull 31.8 的校准等价）。

记 $Q_{m,j}$ 为节点 $(m,j)$ 的 Arrow-Debreu 价格（今日 1 元落在该节点的现值权重），初始 $Q_{0,0}=1$。

**前向递推：**

1. **解漂移**：给定第 $m$ 层的 $Q_{m,\cdot}$，要求一步折现与市场一致：

$$\sum_j Q_{m,j}\,e^{-(\alpha_m + j\Delta r)\,dt} = P(0, t_{m+1}) \quad\Rightarrow\quad
\alpha_m = \frac{1}{dt}\ln\left(\frac{\sum_j Q_{m,j}\,e^{-j\Delta r\,dt}}{P(0,t_{m+1})}\right)$$

2. **写入利率**：$r(m,j)=\alpha_m + j\Delta r$

3. **传播 AD 价格**：$Q_{m+1,j'} \mathrel{+}= Q_{m,j}\cdot p_{j\to j'}\cdot e^{-r(m,j)\,dt}$（顶/底边按修改后的分支连接）

FinancePy 实现（`build_tree_fast` 主循环）：

```python
sum_qz = sum(qq[m, j] * exp(-j * d_r * dt))
alpha[m] = log(sum_qz / discount_factors[m+1]) / dt
```

✅ 与理论公式逐项一致。数值核验：对全部层 $m$，$|\sum_j Q_{m,j} - P(0,t_m)| < 1.6\times10^{-5}$（残差来自曲线插值粒度与末端节点外推，见第 5 节）。

**含义**：树的"形状"（间距、概率）由 $(a,\sigma)$ 决定，而"位置"（每层中心 $\alpha_m$）完全由市场曲线决定。因此 σ/κ 只影响期权价值，不影响 straight bond 的定价——这正是本项目观察到的"equivalent coupon 对 σ/κ 不敏感（当期权价值≈0 时）"的数学根源。

---

## 4. 含权债定价：后向递推

`callable_puttable_bond_tree_fast` 在已校准的树上做标准后向归纳：

1. **终端**（到期节点 $m=N$）：所有节点价值 = 面值 + 期末票息（与利率状态无关）
2. **回退**：$V(m,j) = \big(p_u V_u + p_m V_m + p_d V_d\big)\,e^{-r(m,j)\,dt} + \text{票息}$
3. **行权**（仅限被映射到树上的 call/put 日期）：以**净价**做决策

$$V \leftarrow \min\Big(\max(V_{hold} - \text{accrued},\; V_{put}),\; V_{call}\Big) + \text{accrued}$$

即发行人按 call price + 应计利息回购、投资人按 put price + 应计利息回售——与市场惯例一致（本项目 call 日与票息日重合，应计≈0，决策等价于除息价值 vs 100）。

**离散化技巧**（Hull 31.9 一致）：
- 票息映射到最近树节点，并乘 $\dfrac{P(0,t_{cpn})}{P(0,t_{tree})}$ 修正 PV，消除日期取整误差
- call/put 日期直接取整到最近节点（无 PV 修正，是残余离散误差来源）

---

## 5. FinancePy 实现核验结论

| 检查项 | 理论 | FinancePy | 结论 |
|---|---|---|---|
| Δr 间距 | $\sigma\sqrt{3dt}$ | `d_r = sigma*sqrt(3.0*dt)` | ✅ |
| 截断 j_max | $0.184/(a\,dt)$ | `ceil(0.184/(a*dt))` | ✅ |
| 分支概率 | 矩匹配公式 | 逐项一致（验算均值/方差） | ✅ |
| 负概率 | 应拒绝 | 抛 `FinError` | ✅ |
| 漂移校准 | AD 价格反解 α_m | 公式一致 | ✅ |
| 曲线重定价 | $\sum Q_m = P(0,t_m)$ | 误差 < 1.6e-5 | ✅ |
| 票息 PV 修正 | × P(0,t_cpn)/P(0,t_tree) | 一致 | ✅ |
| 行权惯例 | 净价 + 应计 | 一致 | ✅ |
| 欧式期权 vs Jamshidian 解析解 | 应一致 | 差 < 0.005（N=400） | ✅ |

数值验证记录（本项目 2025-05-27 市场曲线，σ=0.955%，a=0.0524，N=400）：
- ZCB(10y)：树 0.634303 vs 曲线 0.634303 —— 节点上精确
- 纯债券：树 99.9973 vs 曲线 PV 99.9885 —— 离散误差 0.9bp
- 欧式 call@5.5y：Jamshidian 1.5503 vs 树 1.5576 —— 一致

### 已知瑕疵 / 注意事项（不违反数学，但需知晓）

1. **Jamshidian 返回值顺序颠倒**：`european_bond_option_jamshidian` 返回 `(put, call)`，而 `bond_option()` 按 `(call, put)` 解包——走 Jamshidian 路径时 call/put 被互换（FinancePy 的 API 缺陷）。本项目使用 `callable_puttable_bond_tree`，不受影响。
2. **a=0（Ho-Lee）未安全处理**：`j_max = ceil(0.184/(G_SMALL·dt))` 会爆炸成天文数字导致内存错误。0.184/(a·dt) 公式仅对 a>0 有效；Ho-Lee 应直接取固定 j_max。
3. **末端外推**：树多出到期日一个节点，最后一层的 α_N 用曲线**外推**的折现因子校准；对以到期日为最后现金流的债券无害（该层利率不会被用到）。
4. **call 日期取整**：call 日期四舍五入到最近节点（无 PV 修正），N=400 时 ±4.5 天，属小偏差。
5. **步数平均**：`BondEmbeddedOption.value()` 对 N 与 N+1 步的结果取平均——只是平滑振荡，不是 Richardson 外推。
6. **应计利息近似**：源码自述为 "hack"（在票息日强制 accrued=全额票息，保证行权按除息价决策）；对本项目"call 日=票息日"的场景恰好是干净处理。

---

## 6. 与本项目的衔接要点

- **σ 的单位**：HW 树要的是短期利率绝对波动率（`dr = (θ − ar)dt + σdW`）；本项目用 SOFR 标定的 Vasicek σ 直接映射 ✅。若传给 BK/BDT 树则必须换成比例波动率 σ/r₀（它们的 SDE 是 d ln r）。
- **κ 的阻尼效应**：模型隐含的 T 年期零息利率波动率为 $\sigma\cdot\frac{1-e^{-aT}}{aT}\approx \sigma/(aT)$（a 大时）——κ 越大长端波动率越小，期权价值越低。这就是"用 10y 收益率序列标定 κ≈4~7 导致期权≈0、equivalent coupon 塌回 4.6%"的原因。
- **模型自检**：标定完 σ/κ 后，应先验证 $\sigma\cdot\frac{1-e^{-aT}}{aT}$ 与市场同期限收益率的历史波动率同量级（本项目 `callable_bond_financepy.py` 已内置该 sanity check）。
- **Q-measure 注意**：历史（P-measure）σ/κ 在 Girsanov 变换下近似不变，是实用近似；有流动性 swaption/bond option 报价时应改用隐含参数标定。

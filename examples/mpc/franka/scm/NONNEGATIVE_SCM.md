# 单步修正 SCM：softplus 冲量与精确零接触距离

2026-10-22。新模型使用 `scm_soft_corrected` 后端，版本为
`softplus_nonnegative_gap_v2`。原 `scm_corrected` 保留为硬 max、带符号距离的
对照；其他 MPC 的接触模型没有在本次实验中替换。

## 从接触距离重新推导

设几何计算返回带符号距离 \(\phi_i\)，分离为正、接触为零、穿透为负。
模型中的接触距离定义为

\[
d_i=\begin{cases}\phi_i,&\phi_i>0,\\0,&\phi_i\le0.\end{cases}
\qquad
g=Jv_{\rm free}+d/h,\quad
W=JM^{-1}J^\top+rI,\quad D_i=W_{ii}.
\]

因此接触点和穿透点的距离都**严格等于零**。几何距离不能直接换成普通
softplus，否则 `softplus(0)=T log(2)>0`，会失去这个条件。这里平滑的是原来
**初始冲量的 max**：

\[
x_i=-g_i/D_i,\qquad
p_i^{(0)}=T\log(1+\exp(x_i/T)),\qquad T=10^{-4}\ {\rm N\,s}.
\]

实现按正负分支计算稳定的 softplus，避免大指数溢出。温度沿用已有值，
没有根据本次误差调参。`r=1e-8`，`h=0.002 s`。

对当前接触的耦合二次目标

\[
Q(p)=\tfrac12p^\top Wp+g^\top p
\]

仅沿 \(p=\alpha p^{(0)}\)、\(0\le\alpha\le1\) 做一次解析修正。
令 \(A=(p^{(0)})^\top Wp^{(0)}\)、\(B=g^\top p^{(0)}\)，则

\[
Q(\alpha p^{(0)})=\tfrac12 A\alpha^2+B\alpha,\qquad
\frac{\mathrm d Q}{\mathrm d\alpha}=A\alpha+B.
\]

于是

\[
\boxed{\alpha=\operatorname{clip}(-B/A,0,1),\quad
p=\alpha p^{(0)},\quad
v^+=v_{\rm free}+M^{-1}J^\top p.}
\]

`A=0` 时设 `alpha=1`；因为 `r>0`，此时初始冲量也为零。用
`z=J.T*p0`、`A=z.T*Minv*z+r*||p0||²` 计算分母，不构造完整 W，
不增加接触求解迭代。

相对旧 softplus 模型，唯一的物理公式变化是**两处均用 d 替换 phi**：
初始冲量和 alpha 的分子必须使用同一个 g。只修改其中一处就不再是上述
二次目标的精确线搜索。它保证 `Q(p)<=min(Q(0),Q(p0))`，不保证对 GT 的速度误差下降。

softplus 不具有 max 的正齐次性，不能将 `alpha*softplus_T(x)` 写成
`softplus_T(alpha*x)`。若沿用论文中 `s=alpha*g` 的记号，应直接保留
`p=alpha*p0`，不能沿用原硬 max 的等价改写。

这个距离定义不含基于穿透深度的恢复项，也不投影物体位置；它不意味着下一步
几何穿透一定消失。距离截断、alpha 的 clip 和碰撞接触集合仍可能不可微，
所以 softplus 只平滑冲量激活，不能称整个模型处处光滑。

## 相同 GT 下的实测结果

使用原有 7 个网格物体，每类每物体 32 个初态，GT 路径取 10 步，
共 2912 次预测。GT 沿用耦合摩擦锥 LCP、两个 `0.001 s` 子步，
保留原有带符号距离。与 2026-10-19 的数据逐条核对，初态相同、GT 世界速度
最大分量差为 **0**。下表为误差 mean ± 总体 std。

| 条件 | n | 原 scm_corrected 线速度 (m/s) | 新模型线速度 (m/s) | 原角速度 (rad/s) | 新角速度 (rad/s) |
|---|---:|---:|---:|---:|---:|
| 接触转换 | 224 | 0.227103 ± 0.181346 | 0.258506 ± 0.212979 | 6.310433 ± 4.381346 | 6.528344 ± 4.414267 |
| 滑动 | 224 | 0.120158 ± 0.114341 | 0.151859 ± 0.155423 | 3.830032 ± 3.233829 | 4.274287 ± 3.845997 |
| 粘着 | 224 | 0.107547 ± 0.142831 | 0.162601 ± 0.206796 | 1.232682 ± 1.399864 | 1.980837 ± 2.704865 |
| GT 路径上的单步预测 | 2240 | 0.019757 ± 0.065288 | 0.023501 ± 0.082824 | 0.250876 ± 0.772377 | 0.293427 ± 1.027046 |

GT 路径上的预测每步重置到 GT 状态，不是候选模型独立推进的累计轨迹误差。
四类平均误差均增加；新模型粘着线/角误差相对原硬 SCM 分别增加约 51.19%/60.69%。
GT 路径预测的角速度 P95 从 1.734080 降至 1.697075 rad/s，说明不是所有统计量都变差。

固定全部状态的独立 float64 NumPy 消融分开比较了
`signed_hard_max`、`signed_T_1e-4`、非负距离 `hard_max` 和新 `T_1e-4`。
softplus 本身的影响很小，主要差异来自距离截断。新公式的 NumPy/GPU
世界速度最大分量差为 `2.642353e-5`。

1082 个状态包含负距离约束行，误差变化集中于这些状态；其余 1830 个状态
新旧 softplus 预测误差逐条完全相同。原始静置使用 MuJoCo 软接触并保留残余穿透，
GT 的负距离项会主动产生恢复速度，而新模型按定义取消该项。这是相对当前 GT
误差增加的一个可识别来源，不应把全部差异归因于 softplus。

预测失败 0；5824 个 GT 子步全部通过 LCP 残差证书，最大残差
`9.999978206609583e-9`。89 项相关测试通过，覆盖负/零距离等价、两阶段公式一致、
完整质量耦合、多世界、动态/增量两种投影、离开接触时不按穿透深度施加恢复冲量，
以及既有 LCP 和报告流程。

同步 GPU 单世界完整步耗时：原硬 SCM `1.781464 ± 0.112877 ms`，
新模型 `1.772605 ± 0.088770 ms`。包含碰撞、动力学、接触和积分以及同步，
不含重置、质量更新和诊断下载；这次诊断的差异不能视为稳定的性能提升。

## 原始数据与复现

新数据目录：
`output/diagnostics/velocity_error_nonnegative_gap_20261022/velocity_error_20261022_140430/`。

`velocity_error.json` 的 SHA256 为
`3278ca2601dca4e307b466d050375eb20a8cf4b8456d661327a3833435aa5ab9`。

该目录保存原始逐样本预测、LCP 证书、接触距离/alpha 诊断、耗时，
以及 `nonnegative_gap_comparison.md/.json`、`nonnegative_gap_statistics.csv`、
`nonnegative_gap_paired.csv`、`softplus_sensitivity.json`、`velocity_error_statistics.csv`
和表格片段 `paper_table_rows.tex`。分物体、穿透子集、P95/max 和逐样本改善数
均可从这些报告读取。本次没有覆盖论文中的历史结果表。

```bash
conda run -n pi0 python examples/mpc/fingertips/test/benchmark_velocity_error.py \
  --samples 32 --rollout-steps 10 --seed 0 --softplus-temperature 1e-4 \
  --output-dir output/diagnostics/velocity_error_nonnegative_gap_20261022

conda run -n pi0 python examples/mpc/fingertips/test/report_velocity_error.py \
  <new-run-directory>/velocity_error.json

conda run -n pi0 python examples/mpc/fingertips/test/report_nonnegative_scm.py \
  <new-run-directory>/velocity_error.json \
  --baseline output/diagnostics/velocity_error_softplus_20261019/velocity_error_20261019_141134/velocity_error.json

conda run -n pi0 python examples/mpc/fingertips/test/analyze_softplus_velocity.py \
  <new-run-directory>/velocity_error.json

conda run -n pi0 python -m pytest examples/mpc/franka/scm/tests -q
```

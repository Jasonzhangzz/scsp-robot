# softSCM_corrected：softplus 冲量与全局幅值修正

2026-10-22 更新：当前后端已使用 `d=max(phi,0)`，接触/穿透点距离严格为零，
且初始冲量和 alpha 都使用这个距离。完整推导、配对结果和复现见
[NONNEGATIVE_SCM.md](NONNEGATIVE_SCM.md)。以下是 2026-10-19 带符号距离版本的历史记录。

当时的 `scm_soft_corrected` / 表中的 `softSCM_corrected` 是 `scm_corrected` 的
LogSumExp 平滑版本（`law_version=softplus_impulse_v1`）。旧的弹簧阻尼加全局修正
实现已移除；不再读取该模型的 `tau,zeta`。独立的未修正 `scm_soft` 仍保持其原定义。
生产 MPC 的其他接触模型未因此替换。

令

\[
g=Jv_{\rm free}+\phi/h,\quad
W=JM^{-1}J^\top+rI,\quad D_i=W_{ii},\quad x_i=-g_i/D_i.
\]

硬 SCM 的初始冲量为 `max(x_i,0)`。新模型使用平滑最大值，而不是分类概率 softmax：

\[
p_i^{(0)}=\operatorname{smax}_T(x_i,0)
=T\log(1+\exp(x_i/T)),\qquad T=10^{-4}\ {\rm N\,s}.
\]

GPU 实现分别在 `x>0` 和 `x<=0` 计算，避免指数溢出；两分支在零点的导数均为 1/2。
温度作用于**冲量**，而不是速度或接触力；`0 <= smax_T(x,0)-max(x,0) <= T log(2)`。
因此修改质量、步长或单位时，需要重新考虑温度尺度。这里的温度预先固定，没有按表中误差择优拟合。

随后直接缩放这个冲量，保留硬 SCM 的自由运动和同一全局方向二次目标：

\[
\alpha=\operatorname{clip}
\left(-\frac{g^\top p^{(0)}}{(p^{(0)})^\top Wp^{(0)}},0,1\right),\quad
p=\alpha p^{(0)},\quad v^+=v_{\rm free}+M^{-1}J^\top p.
\]

分母为零时定义 `alpha=1`。分母仍按 `q=J.T*p0`、`q.T*Minv*q+r*p0.T*p0`
计算，不构造完整接触矩阵，也不增加接触求解迭代。

softplus 不具有 max 的正齐次性，所以一般不能把 `alpha*softplus_T(x)` 改写为
`softplus_T(alpha*x)`；实现保留前一种顺序。
`max` 被平滑后，`clip` 和碰撞接触集合的变化仍可能不可微，不能称整个模型处处光滑。

## clip 可以怎样平滑

设 `A=p0.T*W*p0>0`，`x=-g.T*p0/A`，沿该方向相对自由运动的二次目标变化是

\[
\Delta Q(\alpha)=A(\alpha^2/2-x\alpha).
\]

**方案一：两个 softplus 之差。**令

\[
C_\epsilon(x)=\epsilon\log(1+e^{x/\epsilon})
-\epsilon\log(1+e^{(x-1)/\epsilon}).
\]

它是 C-infinity 的平滑夹紧，值域在 `(0,1)`，导数为
`sigmoid(x/epsilon)-sigmoid((x-1)/epsilon)`，随 epsilon 趋零收敛到 clip。
但当 `x<0` 时仍有正输出：原来最优解应为零反力，现在会沿上升方向施加非零反力，
因此不能保留全局线搜索的目标不增保证。不是简单地换成 sigmoid(x)：后者也不是 `[0,1]`
投影的近似。`epsilon` 是无量纲幅值过渡宽度，与冲量温度 T 是不同参数。

**方案二：保留零反力的 C1 过渡。**取 `0<epsilon<1/2`：

\[
C^{1}_\epsilon(x)=
\begin{cases}
0,&x\le0,\\
\epsilon(2t^2-t^3),\ t=x/\epsilon,&0<x<\epsilon,\\
x,&\epsilon\le x\le1-\epsilon,\\
1-(1+\epsilon-x)^2/(4\epsilon),&1-\epsilon<x<1+\epsilon,\\
1,&x\ge1+\epsilon.
\end{cases}
\]

每个连接点的值与一阶导数连续。`x>0` 时有 `0<=C1_epsilon(x)<=min(x,1)`，
因而 `Delta Q<=0`；`x<=0` 时仍精确返回零。这保留“不比自由运动的接触二次目标更差”，
但不再是该方向的精确最小点，也不保证优于原始 `p0`，不保证参考速度误差下降。
C1 可以支持常规一阶优化，但它仍有分段表达，且没有消除碰撞集合的非光滑变化。

建议需要平滑 alpha 时优先试方案二；若要求 C-infinity 且能接受零反力附近的偏置，
可选择方案一。本次论文仅评估 **softplus 初始冲量 + 原 hard clip**，两种 clip 替代
仅作独立离线分析，没有悄悄改变 TABLE II 的模型。

## 可复现运行

使用现有 pi0 环境及 scm/.deps 的固定依赖，从仓库根目录运行：

```bash
conda run -n pi0 python examples/mpc/fingertips/test/benchmark_velocity_error.py \
  --samples 32 --rollout-steps 10 --seed 0 --softplus-temperature 1e-4 \
  --output-dir output/diagnostics/velocity_error_softplus_20261019

conda run -n pi0 python examples/mpc/fingertips/test/report_velocity_error.py \
  <run-directory>/velocity_error.json --tex tro_revision/experiment.tex

conda run -n pi0 python examples/mpc/fingertips/test/analyze_softplus_velocity.py \
  <run-directory>/velocity_error.json
```

报告器只接受带新 `law_version` 和修正后世界角速度定义的数据，拒绝将历史旧模型
当作当前 softSCM_corrected。它从同一份原始逐样本数据重算均值、总体标准差、P95、
最小值加粗及表格，缺失预测会报错。

本次顺便修正了两子步参考解的角速度坐标换算：使用最终姿态把最终局部角速度
转换到世界坐标；DART 多子步辅助函数同样修正。所有模型按修正后的参考一起重算。
初态仍沿用 MuJoCo 软接触静置生成，残留穿透会被硬参考修正；该协议限制已写入正文。

## 2026-10-19 实测记录

原始数据：`output/diagnostics/velocity_error_softplus_20261019/velocity_error_20261019_141134/velocity_error.json`。

SHA256：`f10133184f9eecdcb1622e9998171ca69da7d5bc1e4bb563f09688d1ef7586c8`。

共 2912 次预测、5824 个参考子步；论文模型失败数为零。最大 LCP 残差 `9.999978206609583e-09`。72 项相关测试通过，覆盖 GPU 公式、自动微分、重置、旧参数独立性、LCP 证书和角速度坐标转换。

| 状态 | 线速度误差 mean ± std (m/s) | 角速度误差 mean ± std (rad/s) | P95 线速度 | P95 角速度 |
|---|---:|---:|---:|---:|
| transition | 0.227103 ± 0.181346 | 6.310433 ± 4.381346 | 0.674163 | 15.432358 |
| sliding | 0.120163 ± 0.114341 | 3.830072 ± 3.233847 | 0.401918 | 10.904507 |
| sticking | 0.107547 ± 0.142831 | 1.232682 ± 1.399864 | 0.439079 | 4.049739 |
| rollout | 0.019758 ± 0.065289 | 0.250857 ± 0.772376 | 0.110519 | 1.734080 |

敏感性分析使用全部 2912 个固定初态，由 GPU 提供几何和自由运动，独立 NumPy 计算冲量；不是重新运行候选模型的轨迹。基准温度下 NumPy 与 GPU 的世界速度最大分量差为 `3.0090823116779575e-05`。

| 初始冲量 | Sliding 线/角均值 | Sticking 线/角均值 | Rollout 线/角均值 |
|---|---:|---:|---:|
| hard_max | 0.120158 / 3.830031 | 0.107547 / 1.232682 | 0.019757 / 0.250876 |
| T_1e-5 | 0.120158 / 3.830031 | 0.107547 / 1.232682 | 0.019757 / 0.250876 |
| T_1e-4 | 0.120163 / 3.830072 | 0.107547 / 1.232682 | 0.019758 / 0.250857 |
| T_1e-3 | 0.120713 / 3.834527 | 0.107984 / 1.262449 | 0.019851 / 0.251056 |

在这个温度范围内，softplus 与硬 SCM 的整体误差接近；T=1e-3 时 sticking 角速度均值略增。该结果不支持宣称平滑提高了预测精度。

epsilon=0.01 的两种 clip 替代在实测固定状态中都没有出现大于 1e-12 的目标上升，但这不能替代保证：softplus 冲量产生了 12 个未夹紧 alpha<=0 的样本（数值影响很小）。C1 版本在 100006 个标量网格点上通过范围、精确零输出及目标不增检查。

LogSumExp clip 的明确反例：A=1、x=-0.001，平滑 alpha=0.00644397，Delta Q=2.7206319e-05>0；硬 clip 和 C1 均返回零。细节见原始目录中的 `softplus_sensitivity.json`、`clip_scalar_study.json`。

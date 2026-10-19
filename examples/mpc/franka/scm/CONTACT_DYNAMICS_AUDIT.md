# 接触动力学核对与修复（2026-10-16）

结论：DART 可以直接使用 MuJoCo 的接触几何；softSCM 保留 Delassus 对角后，增加一个无新拟合参数的全局幅值校正，可以接近 ComFree 的持续接触表现，但会损害冲击响应；ContactSDF 原实现确有严重的时间尺度错误，修复后也不能把未训练的动态扩展当作上游训练后的准动态模型。

## 1. 实际代码位置与 DART

`compare_prediction.py` 原先的 `lcp` 是 QuantEcon Lemke 求解的耦合棱锥 LCP，**不是 DART**。
DART 和 ContactSDF 的实际比较代码位于 `examples/mpc/fingertips/test/benchmark_velocity_error.py`。
那个脚本原先就从 MuJoCo 获取接触点，并没有调用 DART collision detector；问题还包括接触方程的转写和比较条件。

新增 `contact_dynamics.py`，让两处脚本共用以下流程：

1. MuJoCo 计算当前接触点、法向、间隙、材质混合后的摩擦系数和接触点 Jacobian。
2. MuJoCo 提供完整质量矩阵和 **无约束** 加速度 `qacc_smooth`，不读取其约束反力作为模型反力。
3. 独立计算 DART boxed contact impulse，再用 MuJoCo 的位姿积分更新状态。

逐项核对了 [DART ContactConstraint v6.16.0](https://github.com/dartsim/dart/blob/v6.16.0/dart/constraint/ContactConstraint.cpp) 和 [PGS](https://github.com/dartsim/dart/blob/v6.16.0/dart/constraint/PgsBoxedLcpSolver.cpp)：

- CFM 应为 `W_ii *= 1 + 1e-5`，旧脚本却加了绝对常数 `1e-5`。
- 默认最大穿透恢复速度应为 `0.001 m/s`，旧脚本使用 `0.1 m/s`。
- 使用 DART 默认 ODE 切向基，求解后把力转换回 MuJoCo 接触坐标系。
- 法向冲量非负，两个切向分别受 `[-mu*p_n, mu*p_n]` 约束；这是 DART 的 box friction。
- 使用确定性顺序、零初值、30 次迭代的 PGS；输出记录其收敛标志。PGS 结果不等同于严格精确的物理真值。

`compare_prediction.py` 新增 `--engines dart`，保留原有 `lcp` 的含义。它们的摩擦模型不同，不能简单作为同一问题的两个求解器互换。
该 DART backend 是经过原生二进制交叉验证的 NumPy 方程移植，不是运行一个完整 DART world。

另修复了速度比较脚本的两个错误：模式判定现在计算接触点相对线速度，不再把静止地面的空间速度/角速度分量当作滑动速度；rollout 的 `next` 现在推进完整的两个 1 ms 子步。

## 2. ComFree 与 softSCM 的差异

设 `u=J*v_free`，`D=diag(J*M^-1*J.T)+r`。当前 softSCM 为

```text
p_i = max(-(a*u_i + b*phi_i/h)/D_i, 0)
a = (2*zeta*x+x²)/(1+2*zeta*x+x²)
b = x²/(1+2*zeta*x+x²),  x=h/tau
```

本仓库固定版本 ComFree 的冲量则可直接整理为

```text
p_i = max(-m_row_i*((d+k*h)*u_i + k*phi_i), 0)
m_row_i = impedance_i / (invweight_i*(1-impedance_i))
```

这里的 `k,d` 是代码中的系数，不能不经时间尺度换算就视为普通 SI 弹簧/阻尼常数。
`m_row` 使用 body translational invweight 和随间隙变化的阻抗，同一接触的各条棱共享质量尺度；它不是当前精确的 `1/D_i`。
两者都在无接触预测速度上逐行求反力，ComFree 在这段实现里也没有多接触迭代求解。

因此差异有三个来源：有效行质量不同；速度/间隙增益及其时间尺度不同；逐行反力叠加后缺少耦合反馈。
即使每一条对角质量更精确，同一刚体上的多点、多棱也会重复响应同一运动。直接加总并不保证总反力更准确。
ComFree 的系数和阻抗在本数据上起到了更强的幅值正则化作用；这不是“指定阻尼就必然更物理”的证据。

原比较还存在一个明确的不公平条件：ComFree 使用 `condim=4` 的扭转摩擦，softSCM 却裁成 `condim=3`。
现在速度比较统一为三维点接触。CPU MuJoCo 与 vendored MJWarp 的接触集合仍可能有数值差别，不能把全部差距归因于接触律。

“避免任意指定阻尼”是合理的建模诉求，但 softSCM 的 `tau,zeta` 本身已隐含 `k_i=m_i/tau²`、`d_i=2*zeta*m_i/tau`。
保持 `D` 的物理意义，并不消除接触柔顺性参数。数值参考是 MuJoCo 软接触时，完全刚性的模型也未必有最小预测误差。

`compare_prediction.py` 的默认 `tau,zeta` 由 `.05,1` 改为既有 2 ms 物理基准校准值 `.008,.25`。
`.05,1` 是较长 MPC 步长的保守初值。此次没有改生产 MPC 的默认参数，也没有用本次新种子重新拟合参数。

### 历史实现：弹簧阻尼加幅值修正（已移除）

以下记录旧实验的实现。2026-10-19 起，`scm_soft_corrected` 已替换为基于
`scm_corrected` 的 softplus 平滑版本，不再使用这里的 `a,b,tau,zeta`。
当前公式和 `clip` 平滑讨论见 [SOFTPLUS_SCM.md](SOFTPLUS_SCM.md)。

新增独立选项 `scm_soft_corrected`，先按原式得到 `p0`，再只加一个标量：

```text
g_soft = a*J*v_free + b*phi/h
q = J.T*p0
alpha = clip(-g_soft.T*p0 / (q.T*M^-1*q + r*p0.T*p0), 0, 1)
p = alpha*p0
```

零反力时定义 `alpha=1`。仍直接使用 `D`，不构造完整 Delassus，不反演接触矩阵，不增加迭代或拟合系数。
这是沿 `p0` 方向最小化耦合二次目标的标量解，保证该目标不增；不保证相对参考速度的误差下降。
它可以抑制重复接触导致的过强反力，但不能重新分配接触力，也不能修复方向错误。
`alpha` 随输入变化后，整体映射也不再保持原逐行模型的分段仿射结构；不能直接沿用原 MPC 的导数结论。
所以此次把它作为 benchmark 选项，没有自动替换 MPC 接触模型。

## 3. ContactSDF 的错误与修复边界

核对了 [上游 dynamics/lse_dynamics.py](https://github.com/asu-iris/ContactSDF/blob/3a21edf8e51788f72a7d77259af4dd584d5ec43f/dynamics/lse_dynamics.py)，commit `3a21edf8e51788f72a7d77259af4dd584d5ec43f`。
上游是带可学习度量的 **准动态** 模型，其核心为

```text
z_query = Q^-1/2*b
A = J*Q^-1/2/h,  offset = phi/h
normalize each row using sqrt(||A_i||²+1e-2)
distance = max(logsumexp(sigma*(-offset-A*z_query))/sigma, 0)
z_next = z_query - distance*gradient(distance)
v_next = Q^-1/2*z_next/h
```

旧脚本把真实物理质量直接当作上游的准动态度量 `Q`，并令 `b=force`。无接触时因此得到 `v= M^-1*force/h`；
真实动态更新应为 `v_current+h*M^-1*force`。在 `h=.002` 时，单看加速度引起的增量，这两者相差 `1/h²=250000` 倍。
旧实现还丢掉输入速度、丢掉质量矩阵的非对角/质心偏移耦合，并手工近似外力和偏置项。

现在抽出了与上游一致的 `lse_project`。为参加本次动态速度比较，明确使用

```text
Q = M/h²
b_eff = M*v_current/h + f_smooth
z_query = sqrt(M)*v_free
v_next = M^-1/2*z_next
```

完整质量矩阵通过对称特征分解处理；空接触时精确恢复无接触动态预测。
归一化的 `+1e-2`、LSE、外层 `max(0,...)` 和梯度都与上游一致。没有为了降低误差偷偷增加投影迭代。

输出中明确标记：这是 **MuJoCo collision + ContactSDF LSE 的物理质量动态扩展**，不包含上游 C-SDF 几何，也不是上游训练后的准动态模型。
修复量纲不意味着它会接近 ComFree：一次 LSE 最坏违约近似投影不是完整多接触求解，结果也依赖学习后的度量和 sharpness。
本次没有七个网格物体的上游训练参数，不能声称复现了其训练后性能。修复后的剩余误差已完整保留。

## 4. 新种子验证

七个仓库网格，`seed=20261016`，每物体每模式 32 个初始样本；rollout 每个初始样本推进 10 个参考窗口。
共 2912 次共享状态查询，所有模型均有完整有限预测。参考为两步 1 ms 的 DART 接触方程移植；预测步长 2 ms。
rollout 统计也是在参考路径上逐步重新查询的预测误差，不是各模型自主长时 rollout 的轨迹误差。

以下每格为平均线速度误差 m/s / 平均角速度误差 rad/s；原始文件保存了标准差和样本数。

| 模式 | softSCM | softSCM + alpha | ComFree | ContactSDF 动态扩展 |
|---|---:|---:|---:|---:|
| 冲击 transition，n=224 | .2549 / 5.9936 | .6374 / 3.3039 | .6137 / 3.2168 | .2877 / 8.1923 |
| 滑动 sliding，n=224 | .1180 / 2.1901 | .0206 / .5346 | .0148 / .5340 | .2442 / 5.6607 |
| 静止 sticking，n=224 | .0173 / .3552 | .0143 / .0443 | .0141 / .0539 | .0776 / .9561 |
| 参考路径 rollout，n=2240 | .0299 / .7760 | .0275 / .3191 | .0278 / .3181 | .0782 / 1.5743 |

可支持的结论：一个标量修正在持续接触中接近 ComFree，静止角速度误差更小；冲击线速度误差显著变大，不能宣称全面对齐。
原 softSCM 在冲击线速度上本来就比 ComFree 好。后续若要统一覆盖冲击与持续接触，需要有针对性的接触耦合/恢复律设计，不能仅凭这一标量结果替换所有工况。

数据位置：

- `output/diagnostics/contact_audit_holdout/velocity_error_20261016_212812/velocity_error.json`：上述新种子数据。
- `output/diagnostics/contact_audit_prediction/`：原比较入口的 box/sphere/elephant，90 共享状态、六个 engine 的 smoke test。
- `output/diagnostics/contact_audit_validation/summary.json`：上游核对结果与原生 DART 验证信息。

自动测试：`examples/mpc/franka/scm/tests` 共 **74 passed**（包含 GPU 数值一致性、冲量/力坐标转换、自由运动量纲和 rollout 窗口回归）。

数值核对：直接执行上游 `LseDynamic.D_SDFInit`，三个步长共 300 个随机输入，最大投影差 `6.89e-15`。
与官方 `dartpy==6.16.7` 的 C++ PGS 交叉验证 300 个随机问题，最大冲量差 `1.56e-15`。
其中原生 PGS 有 4 个随机问题达到默认迭代上限；没有把有限输出等同于严格收敛。
16 个原生问题的输出保存在 `vendor/dart_contact/pgs_reference.json`，作为可重复的独立测试基准。

复现命令（在仓库根目录）：

```bash
conda run -n pi0 python examples/mpc/fingertips/test/benchmark_velocity_error.py \
  --samples 32 --rollout-steps 10 --seed 20261016 \
  --output-dir output/diagnostics/contact_audit_holdout

conda run -n pi0 python examples/mpc/franka/scm/compare_prediction.py \
  --headless --no-plots \
  --engines mjwarp comfree scm_soft scm_soft_corrected dart contactsdf \
  --shapes box sphere elephant --samples-per-mode 10 \
  --output-dir output/diagnostics/contact_audit_prediction

conda run -n pi0 python -m pytest -q examples/mpc/franka/scm/tests
```

## 5. 补充：小步长速度指标的基线检查

在同一新种子、相同 2912 个输入上，补算了无接触自由动力学和保持当前速度两种基线。
保持速度仅用于检验指标的辨识力，不能作为物理接触模型。平均线速度误差（m/s）：

| 模式 | 无接触自由动力学 | 保持当前速度 | ComFree | 原 softSCM |
|---|---:|---:|---:|---:|
| transition | .7476 | .7287 | .6137 | .2549 |
| sliding | .0276 | .0164 | .0148 | .1180 |
| sticking | .0199 | .0020 | .0141 | .0173 |
| reference-path rollout | .0333 | .0298 | .0278 | .0299 |

因此，前文“校正后接近 ComFree”仅指在这个一步误差指标上接近，不能推出接触动力学已经准确。
尤其在 sticking，保持当前速度比 ComFree 的线速度误差低很多；在 sliding，ComFree 相比保持速度的改善也很有限。
2 ms 内输入速度占据总速度的大部分，较弱的接触修正可能得到较小总速度误差。
冲击工况的结果又表明 ComFree 并不普遍优于原 softSCM。不能把全部优劣归结为某个行质量更准确。

下一步若评判接触模型本身，应同时比较接触引起的速度修正/冲量、静态承重误差，以及自主长时 rollout；
接触修正接近零时不宜单独使用逐样本相对误差。还需要对全局增益在质量、步长、摩擦和接触数变化下做相同预算的敏感性检查。
新增数据与诊断脚本：`output/diagnostics/comfree_explanation/free_baseline.json`、`free_baseline.py`。

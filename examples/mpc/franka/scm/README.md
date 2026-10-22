> 2026-10-22 更新：`scm_soft_corrected` 使用 softplus 冲量和 `d=max(phi,0)`，接触及穿透点距离严格为零；初始冲量和全局幅值修正均已更新。推导、与相同 GT 的配对评估见 [NONNEGATIVE_SCM.md](NONNEGATIVE_SCM.md)。硬 `scm_corrected` 保留作对照。
>
> 2026-10-17 更新：可独立运行的未修正 `scm` 已移除，默认与 `three` 组合改用 `scm_corrected`。下面涉及原始 SCM 的数值表、推导和旧命令属于历史记录；重现实验应显式选择仍保留的 corrected/soft 分支。七物体点力闭环实验见 `examples/mpc/fingertips/test/README_scm_solvers.md`。其他 MPC 的 softSCM 保留。

# ComFree / MJWarp / SCM 接触预测对比

迁移来源：[asu-iris/comfree_warp](https://github.com/asu-iris/comfree_warp)，固定到
`ba8b9969f5f7859d51b1f738f458f8d2f49be080`。

上游提供以下入口：

- [`test_local/test_franka_grasp.py`](https://github.com/asu-iris/comfree_warp/blob/ba8b9969f5f7859d51b1f738f458f8d2f49be080/test_local/test_franka_grasp.py)：切换 MuJoCo / MJWarp / ComFree，执行 Panda 抓取、抬升和扰动，报告方块高度与成功情况。
- [`test_local/test_viewer.py`](https://github.com/asu-iris/comfree_warp/blob/ba8b9969f5f7859d51b1f738f458f8d2f49be080/test_local/test_viewer.py)：切换后端观察不同 XML 场景。
- [`test_local/test_throuput_hand.py`](https://github.com/asu-iris/comfree_warp/blob/ba8b9969f5f7859d51b1f738f458f8d2f49be080/test_local/test_throuput_hand.py)：比较吞吐量。

本目录迁移了抓取控制流程、Franka 场景及全部所需网格，以及三个基础物理场景。
新增了相同初始状态和相同预设输入的成对运行、默认可视化、轨迹导出、几何穿透和状态差异统计。
公开上游没有找到论文中的穿透汇总、摩擦曲线或稳定性扫描实验脚本；这里新增的统计不能直接称为论文实验的复现。
新增 `compare_prediction.py` 比较 separation、sliding、sticking 三种状态下的共享状态单步预测，包含 SCM。
`compare_physics.py` 记录连续仿真的穿透与摩擦诊断，也支持 SCM。

## 环境与运行

使用现有 Conda `pi0` 环境，需要 NVIDIA CUDA GPU 和 Git。
`pi0` 已有 MuJoCo 3.11 和 Warp 1.17；上游固定依赖 MuJoCo 3.6.0。
所需 ComFree、MuJoCo 3.6.0 和 Warp 1.16.0 安装到本目录 `.deps/`，
脚本使用 `pi0` 的 Python 加载它们。NumPy 等通用依赖沿用 `pi0`，其共享包版本保持原有设置。

```bash
cd /home/zz/scsp-robot/examples/mpc/franka/scm
conda activate pi0
python -m pip install --target .deps --no-deps -r requirements.txt
```

本机依赖已准备好，激活 `pi0` 后可直接运行后面的命令。
`mjwarp` 使用上游内置的 `comfree_warp.mujoco_warp`，保持与 ComFree 共用的碰撞、积分实现一致。
它对应固定上游版本，不代表本机独立安装的最新 `mujoco-warp`。
首次运行会编译 Warp 内核，需要额外时间。
固定上游的空传感器阶段在 Warp 1.16 下存在编译错误；这些场景没有传感器，脚本对两种 GPU 引擎均跳过该空阶段，动能由 CPU 计算。
125 物体场景会用到上游稀疏 Newton 求解器，它也存在临时 tile 未初始化的编译错误。
`upstream_compat.py` 在首次运行该场景时自动为 `.deps` 中的依赖补充一次 tile 初始化，保留原文件为 `solver.py.upstream`；
补丁可重复执行，只修改本目录的依赖。接触公式、Hessian 计算及原求解器保持原有算法。

## Separation / sliding / sticking 的 prediction error

```bash
python compare_prediction.py --output-dir outputs/prediction
```

默认开可视化弹窗，包含 MJWarp、ComFree、SCM、`scm_corrected`（解析幅值修正）。默认方块（多点接触）和球（单点接触），
每种状态、每种形状各生成 100 个共享查询。无界面全形状实验：

```bash
python compare_prediction.py --headless --shapes box sphere cylinder capsule ellipsoid \
  --samples-per-mode 100 --output-dir outputs/prediction_modes
```

也支持仓库 `envs/assets/objects/` 的六个 STL 物体：
`elephant`、`stanford_bunny2`、`teapot`、`foam_brick`、`rubber_duck`、`mug`。
`stanford_bunn2` 是 `stanford_bunny2` 的兼容别名。默认仍是 box/sphere，六物体运行命令为：

```bash
python compare_prediction.py --engines mjwarp scm \
  --shapes elephant stanford_bunny2 teapot foam_brick rubber_duck mug \
  --modes separation sliding sticking --samples-per-mode 100 \
  --output-dir outputs/prediction_objects_new
```

物体保留 STL 原始尺度，沿用对应 `env_fingertips_<物体>.xml` 的 geom 四元数，
按编译后网格的最低点放在地面上方 5 mm，再由 MJWarp 落定。总质量由 `--mass` 指定，
惯性由 MuJoCo 根据网格推断；碰撞使用单个网格的凸包，不是非凸分解。
网格在初始 `--settle-steps` 后仍有运动时，会延长落定到最多该步数的五倍，
实际步数记录为 `settle_steps_used`；达到上限仍未落定则报错，不把晃动位姿当作静止支撑。
模型 XML、源网格路径、SHA256、尺度和摆放四元数保存到每个配置；回放检查网格是否一致。

### 为什么原来的画面主要像下落？

三种模式共用物体--平面模型，但显式生成不同的初始速度/外力（下文列出）。
原显示只依次展示各模型推进一个时间步的状态：默认 h=.002 s，切换模型/样本时又重置。
因此滑动只有毫米级位移，sticking 接近不动，空中样本的随机高度变化最显眼；画面不是连续轨迹。

现在窗口标明物体、后端、requested/reference/outcome 模式和样本编号，并显示速度及外力大小。
每种 requested 模式的第一个输入先播放 .6 s 的独立 MJWarp 预览，以便观察落下/滑动/受力保持支撑。
预览中的 current 模式会随接触变化而更新；预览不计入误差统计，也不用于生成下一查询。
`--preview-seconds 2` 可延长，`--preview-seconds 0` 关闭；`--display-pause .1` 延长单步结果的显示。
`--headless` 关闭窗口和预览。只看 sliding 可加 `--modes sliding`，只看 sticking 可加 `--modes sticking`。
终端在每种模式结束时输出实际参考模式计数，不把请求的类别直接当作测量标签。

`--frictions 0.2 0.5 1`、`--timesteps 0.002 0.01` 扫描参数；
`--contact-stiffness`、`--contact-damping` 改 ComFree 增益，`--scm-regularization` 改 SCM 的 r。
`--seed` 默认 42。同一输出配置目录已存在时拒绝覆盖，应换一个 `--output-dir`。

每个配置先由 MJWarp 自由落下至静止，再生成三组初始条件：

- separation：空中运动或从接触面离开。
- sliding：支撑位姿上赋予随机方向的切向速度、少量角速度和切向外力。
- sticking：初始速度为零，施加低于静摩擦限的切向外力。

sliding 的小角速度绕世界 z 轴，随后转换为 free-joint 局部角速度，
避免倾斜网格绕局部 z 轴时人为引入接触法向运动。

各模型获得完全相同的位姿、速度、外力和时间步长。每次查询重置求解器 warm start，
只推进一轮；预测结果不用于生成下一个查询。因此这些误差不包含长轨迹累积误差。
MJWarp 是数值参考，比较结果不等同于真实物理测量误差。

分析分组采用**输入状态的接触模式**，由 MJWarp 在该输入位姿的求解接触点、接触载荷和输入速度测量：
法向载荷小于 `--force-threshold`（默认 1e-4 N）时为 separation；
有载荷且切向速度超过 `--slip-threshold`（默认 1e-3 m/s）时为 sliding，否则为 sticking。
有载荷且输入法向速度超过 `--normal-threshold`（默认 1e-2 m/s）时单列 transition。
不以“要求生成的模式”直接代替测量标签；原始标签和实际标签均保存。
`reference_outcome_mode`、`predicted_mode` 使用反作用后的速度，单独反映预测模式转换。
同一刚体的加载接触点只要有一个滑移，刚体就归入 sliding。

| 文件 | 内容 |
|---|---|
| `manifest.json` | 命令参数、环境、上游版本、统计定义 |
| `summary.csv` | 配置 × 后端 × 参考模式的全部统计汇总 |
| `prediction_mean_std.csv/.md` | 单步误差的 mean±std 主汇总表，含有效样本数；每个配置也保存一份 |
| `<配置>/states.npz` | 原始 qpos/qvel、外力、自由速度和各模型预测，无需 pickle |
| `<配置>/predictions.csv` | 逐次速度/位姿误差、方向角、相对误差、摩擦、接触数和耗时 |
| `<配置>/contacts.csv.gz` | 原生求解 gap、frame、系数、实际接触力/力矩、相对速度、摩擦功率 |
| `<配置>/poststate_contacts.csv.gz` | 统一 CPU 碰撞流程计算的预测结束后的几何穿透；不替换 GPU 反作用力 |
| `<配置>/metrics.json` | 按模式的 mean/std/RMSE/p95/max、方向一致率、未定义方向数、失败数和模式混淆 |
| `<配置>/prediction_errors.png/.pdf` | 误差 mean±std 图，误差棒为标准差（不是置信区间），原始分布见 CSV |
| `<配置>/model.xml`, `settled_state.npz` | 模型和支撑初始位姿 |

速度误差分别用 m/s、rad/s；位姿误差分别用 m、四元数测地角 rad。
整体运动误差与方向采用惯性加权范数，避免直接混加平移与角速度；
分别报告总速度、速度增量、接触修正三个口径。
任一方向的范数低于 `--direction-threshold` 时方向未定义（CSV 留空、JSON 为 null），
不将静止样本记为方向正确。sticking 重点看绝对误差和错误运动量。
失败模型保留样本和原因，不从分母删除；失败参考列为 invalid_reference。

### SCM 口径

SCM 在四个非负棱锥面乘子描述的点接触上应用时变对角近似；与
`planning/scm_contact_models.py::surrogate_response` 的四行参考采用相同对角近似思想。
每步用当前 J 和完整惯性重算 `diag(J M^-1 J.T)+rI`，不加 ComFree 额外阻尼：

```text
v_free = v + h M^-1 f_smooth
p = max(-(J v_projected + gap/h) / (diag(J M^-1 J.T) + r), 0)
v_next = v_free + M^-1 J.T p
```

p 为冲量，记录的力为 p/h，r 与逆质量同单位。
默认 `--scm-motion dynamic` 取 `v_projected=v_free`，是对角 SCM 接触律的 full-dynamic 单步扩展。
`--scm-motion increment` 取 `v_projected=v_free-v`，接近当前 CSO 的增量投影。
有非零初速度的 sliding 实验必须注明版本；两者不能混称为同一论文预测器。
这里使用硬 max，不加 SoftPlus，也不调用 CSO/contact selection。
当前 `planning/mlqp_point.py` 的常数阻抗与论文时变对角阻抗不同，实验不会将二者误称为相同实现。
论文三分量接触力的文字表述与此处四面乘子实现也需区别，不能用本实验替代逐项公式一致性检查。

预测场景均为 condim=3，没有扭转/滚动摩擦。三种 GPU 引擎共用碰撞、接触行生成和 Euler 积分，
只替换接触反作用。独立刚体中心惯性在局部角速度坐标中为常数，但 J 随状态改变，D 逐步重算。
SCM 不要求完整 Delassus 可逆。

### SCM 解析幅值修正

新增独立后端 `scm_corrected`，原 `scm` 接触律保留。先按 SCM 计算冲量 p，
再沿该冲量方向，在当前冻结系数的耦合二次目标上进行一次解析线搜索：

```text
g = J v_projected + gap/h
W = J M^-1 J.T + rI
alpha = clip(-(g.T p)/(p.T W p), 0, 1)   # p=0 时 alpha=1
p_corrected = alpha p
v_next = v_free + M^-1 J.T p_corrected
```

alpha 为每个世界一个标量，所有接触冲量共同缩放；自由运动与外力保持原来的计算。
实现利用 `(J.T p).T M^-1 (J.T p) + r ||p||²` 计算分母，
不构造或求逆完整 W，不读取 MJWarp 预测，无需辨识或训练。
它保证沿 `[0,p]` 的匹配耦合二次目标不增大，不直接保证相对 MJWarp 的速度误差。
`increment` 模式使用自己的增量 g；当前实验结论均为 `dynamic` 模式。
GPU 增加一个计算/缩放内核；当前实现用每世界线程归约，适用于本次小刚体场景，
大接触数/多世界下的吞吐量与全局单标量效果尚未验证。

以下命令回放原实验的精确输入（默认有弹窗；无显示服务时显式加 `--headless`）：

```bash
python compare_prediction.py --headless --shapes box sphere cylinder capsule ellipsoid \
  --samples-per-mode 100 --replay-dir outputs/prediction_modes \
  --output-dir outputs/prediction_amplitude
python analyze_amplitude.py --input-dir outputs/prediction_amplitude
```

`--engines mjwarp comfree scm scm_corrected` 可显式指定四模型。
重复运行须更换 prediction 输出目录；已有配置目录拒绝覆盖。
`analyze_amplitude.py` 只读取数据，生成合并五形状的 mean±std 表、图、逐样本配对比较，
以及输入一致性检查 `replay_verification.json`。
`predictions.csv`/`metrics.json` 另存修正系数 `scm_amplitude_scale`。

本次精确回放五形状 × 三模式 × 100 状态 = 1500 状态、6000 次预测，全部完成。
核对四模型输入与原始数据逐项相等；原 MJWarp/ComFree/SCM 预测速度与上次数据也逐项相等。
配置 h=.002、mu=.5、mass=.15、ComFree (k,d)=(.1,.001)、SCM r=1e-8。

| 模式 | 模型 | 线速度误差 mean±std (m/s) | 角速度误差 mean±std (rad/s) |
|---|---|---:|---:|
| sliding | ComFree | 0.0584 ± 0.0222 | 0.992 ± 0.721 |
| sliding | SCM | 0.3830 ± 0.1952 | 5.722 ± 3.237 |
| sliding | SCM 修正 | 0.2180 ± 0.0935 | 3.547 ± 2.152 |
| sticking | ComFree | 0.0177 ± 0.0011 | 0.0110 ± 0.0123 |
| sticking | SCM | 0.0914 ± 0.0166 | 0.0147 ± 0.0182 |
| sticking | SCM 修正 | 0.0244 ± 0.0118 | 0.0099 ± 0.0061 |

每行 n=500，std 为总体标准差。separation 三者误差均接近浮点零。
修正使 sliding/sticking 平均线速度误差下降 43.1%/73.3%，这两组所有样本线速度误差均改善。
sliding 角速度误差全部改善；sticking 角速度误差仅 41.2% 样本改善，均值下降不代表逐样本保证。
修正后 sliding 线/角速度误差仍约为 ComFree 的 3.74/3.58 倍；sticking 线速度误差约为 1.38 倍，
角速度误差均值稍低。在 sticking 的 box/cylinder 子组中修正模型线速度误差低于 ComFree，
其余三形状较高。因此目前可说显著缓解幅值过强，尚不能称为整体媲美 ComFree。

同步完整 step 的观测均值：sliding 原 SCM 1.947 ms、修正 1.973 ms；
sticking 原 SCM 1.836 ms、修正 1.866 ms，约增 1.3%/1.6%。
这些数值包含主机发起调用与设备同步，不含首次编译、输入上传、读回、CPU 诊断和渲染；
单世界计时不能替代 GPU 批量吞吐量实验。

连续滑动补充实验与论文 IV-C 一样保存水平速度/z 轨迹：

```bash
python compare_physics.py --headless --engine four --scene sliding --steps 1000 \
  --timestep .002 --contact-stiffness .1 --contact-damping .001 \
  --output-dir outputs/sliding_amplitude
```

四模型均完成 1000 步；原 SCM 的大幅反弹明显减弱。
2 s 时 SCM 修正的水平速度约 0.000398 m/s，原 SCM 为 0.389 m/s，ComFree 约为零。
终态动能分别约 3.75e-5、0.192、1.04e-12 J；速度/高度衰减过程仍有差异，不能仅依据终态宣称模型等价。
轨迹见 `outputs/sliding_amplitude/stability_trajectory_time.png/.pdf`，
全接触穿透 mean±std、接触数/耗时散点和原始接触数据也保存。
结论限于这些参数和 full-dynamic 点接触模型，不直接证明原准动态 CSO 的 guidance 质量。

### ComFree 阻尼消融与优化适用性

为区分阻尼贡献，在相同 1500 个已保存输入上，仅把 ComFree 的 d 从 .001 改为 0，
k=.1、h=.002、mu=.5 和其余参数保持不变。所有输入及 MJWarp 参考预测速度逐项核对相等。
结果在 `outputs/prediction_comfree_no_damping/damping_ablation.md/.csv`。

| 模型 | sliding 线速度误差 (m/s) | sliding 角速度误差 (rad/s) | sticking 线速度误差 (m/s) |
|---|---:|---:|---:|
| ComFree d=.001 | 0.0584 ± 0.0222 | 0.9919 ± 0.7210 | 0.0177 ± 0.0011 |
| ComFree d=0 | 0.0694 ± 0.0287 | 0.9563 ± 0.8904 | 0.0191 ± 0.0002 |
| SCM 修正 | 0.2180 ± 0.0935 | 3.5470 ± 2.1522 | 0.0244 ± 0.0118 |

std 精确值以数据文件为准；去除阻尼后 ComFree 的单步线速度误差仍明显低于 SCM 修正，
因此差距不能仅归因于阻尼。此消融不评估无阻尼的长轨迹稳定性。
SCM 的一步 gap/h 恢复尺度、逐面有效惯性与 ComFree 的 gap-dependent impedance 和全局增益不同；
统一 alpha 不能改变冲量分量比例/激活集，也只优化 SCM 对应的耦合目标。

幅值校正仍是解析表达式，冻结系数和激活分支内可求导，
`d(alpha*p)/du = alpha*dp/du + p*dalpha/du`。
它使原来冻结几何下的分段仿射映射变为分段有理映射；max/clip 边界及 p=0 的比值需处理。
优化版可以平滑截断并保护分母，但需要另行验证梯度与求解效率；本次实现验证的是前向预测。
原 ComFree 论文用 SoftPlus 平滑 max，并通过 CasADi/IPOPT 实现 MPC。
本次仿真依赖的 `comfree_core/_src/forward.py` 设置 `enable_backward=False`，
模型公式适合优化并不意味着整个公开 Warp 仿真程序已支持反向传播。

## Section IV 指标与辅助实验

### 耦合 LCP 与 soft-SCM 的六物体比较

`compare_prediction.py --engines mjwarp lcp scm_soft` 增加了耦合 LCP 后端：

```text
W = J M^-1 J.T + r I
g = J v_free + gap/h
0 <= p perp W*p+g >= 0
v_next = v_free + M^-1 J.T p
```

它与 SCM 共用 GPU 碰撞、四面摩擦棱锥行、无接触动力学和 Euler 积分，保留完整接触耦合。
单环境的小型 LCP 默认由专用 QuantEcon v0.11.2 的 Lemke 算法在 CPU float64 中求解，
每次独立检查原始未缩放方程中的非负性和互补残差；
超过 `--lcp-tolerance` 或 `--lcp-max-iterations` 则记录失败，不把未收敛解用于统计。
默认 r=1e-8、残差容差=1e-8、迭代上限=2000。该后端用于精度比较，不作为 GPU 求解吞吐量基准。
`--lcp-solver active-set` 选择独立的可行 active-set QP 算法用于交叉验证，默认是 `lemke`，
不会在 Lemke 失败时自动切换算法。官方 v0.11.2 的三个求解模块原样保存在
`vendor/quantecon_lcp/`，附 MIT 许可和源地址；Numba 使用现有 pi0 环境。

本轮六物体使用旧基础几何体数据校准的 tau=.008 s、zeta=.25，不重新拟合六物体测试数据：

```bash
python compare_prediction.py --headless --engines mjwarp lcp scm_soft \
  --shapes elephant stanford_bunny2 teapot foam_brick rubber_duck mug \
  --modes separation sliding sticking --samples-per-mode 100 \
  --frictions .5 --timesteps .002 --mass .15 --seed 20261009 \
  --scm-response-time .008 --scm-damping-ratio .25 \
  --output-dir outputs/soft_scm_lcp_objects_new
python report_lcp_comparison.py --input-dir outputs/soft_scm_lcp_objects_new
```

正式 Lemke 数据在 `outputs/soft_scm_lcp_lemke_objects_20261009/`：共1800个共享输入，5400次预测，无失败，
每模式600输入；最大 LCP 互补残差3.33e-16。报告逐项验证三个后端的输入一致。
早期 active-set 数据保存在 `outputs/soft_scm_lcp_objects_20261009/`，只作为独立校验。
两次输入逐项一致，线速度最大差异4.17e-9 m/s，角速度最大差异4.88e-7 rad/s；
逐配置审计在正式目录的 `lcp_cross_validation.json`。
`lcp_comparison.md` 和 `lcp_comparison_table.tex` 仅有 MJWarp 和 LCP 两个基准行，每项为 **Soft-SCM 与该行基准的逐样本差异**，三种模式各占三个指标列，mean±std 保留两位小数。
LaTeX 表格使用 `table*` 双栏宽度和 `scriptsize` 字体，只有两个数据行。
穿透差为两模型预测后最大穿透深度之差的绝对值（各模型无接触时深度为0），先逐样本计算差再求均值和标准差；
线速度和角速度差为两模型预测速度向量之差的欧氏范数。
穿透差显示单位为 μm；sliding/sticking 的速度差为 mm/s、rad/s；separation 用 nm/s、nrad/s 显示数值舍入差异。
`lcp_comparison_pooled.csv`、`lcp_comparison_by_object.csv` 和 `soft_scm_pairwise_errors.csv` 保存配对差异；原始 mean/std/max 为 SI 单位，display 字段的单位和比例单独记录。
三个模型自身的绝对穿透和相对 MJWarp 的速度误差另存为 `per_engine_statistics_si.csv`、`per_engine_by_object_si.csv`，全接触点穿透统计单独保留。
分离输入在这一时间步内没有接触冲量，模型共用自由动力学；SCM 与 LCP 的速度完全相同，真实零值不会因减小单位而改变。

可从同一份缓存回放输入，检查 SCM 有限响应公式对应的运动误差界及其保守性：

```bash
python analyze_motion_bound.py \
  --input-dir outputs/soft_scm_lcp_lemke_objects_20261009 \
  --output-dir outputs/soft_scm_lcp_lemke_objects_20261009/motion_bound_analysis
```

该分析重建共同 GPU 接触行并验证预测与缓存一致，不改写原实验数据；输出逐查询 CSV 和按模式 JSON。
误差界采用质量加权速度范数，不能与表格中的 mm/s、rad/s 数值混用。
额外的同响应耦合模型用于分离对角近似误差和响应差异，不等同于 MJWarp。
完整精度保存于 `lcp_comparison_pooled.csv`，逐物体统计见 `lcp_comparison_by_object.csv`，
全接触点穿透另存 `poststate_all_contact_depth_mm.csv`，soft-SCM 与 MJWarp/LCP 的直接差异另存
`soft_scm_pairwise_errors.csv`。统计定义、输入检查和求解残差见 `lcp_comparison_report.json`。

复查两个求解器的完整数据：

```bash
python validate_lcp_runs.py --active-set-dir outputs/soft_scm_lcp_objects_20261009 \
  --lemke-dir outputs/soft_scm_lcp_lemke_objects_20261009 \
  --output outputs/soft_scm_lcp_lemke_objects_20261009/lcp_cross_validation.json
```

### 保留时变阻抗的有限恢复 SCM

新增实验后端 `scm_soft`，保留 `D=diag(J M^-1 J.T)+rI`，把运动项和间隙恢复项分别乘以
两个由响应时间 tau、阻尼比 zeta 推导的系数，不使用额外投影：

```text
x=h/tau
a=(2*zeta*x+x*x)/(1+2*zeta*x+x*x)
b=x*x/(1+2*zeta*x+x*x)
p=max(-D^-1*(a*J*v_projected+b*gap/h),0)
```

这是逐行有效质量 spring–damper 的隐式闭式离散；tau→0 恢复原 SCM。
冻结几何后仍分段仿射，可以沿用 SoftPlus 获得平滑接触项。
推导、局部梯度、误差 lemma 所需修改和论文适用范围见 [SCM_minimal_revision.md](SCM_minimal_revision.md)。
这是新增的全动态基准候选模型，尚未修改 `tro_revision` 源文件或实际 CSO。

训练只用旧 seed=42 数据，在预先定义的 72 组网格上选择 tau=.008 s、zeta=.25。
新 seed=20261009 的1500个输入实际GPU验证中，sliding线速度误差为 .0154±.0107 m/s，
当前 ComFree 为 .0593±.0218 m/s；sticking 为 .0082±.0019 vs .0177±.0011 m/s。
sliding 角速度误差也改善，sticking 平均角速度误差大致相当。
这些参数经过数值参考校准，不能称为无数据调参；也没有等预算重调 ComFree。

```bash
python calibrate_soft_scm.py --headless --input-dir outputs/prediction_modes \
  --output-dir outputs/soft_scm_calibration_new
python compare_prediction.py --headless --engines mjwarp comfree scm scm_corrected scm_soft \
  --shapes box sphere cylinder capsule ellipsoid --samples-per-mode 100 --seed 20261009 \
  --scm-response-time .008 --scm-damping-ratio .25 --output-dir outputs/soft_scm_holdout_new
python report_soft_scm.py --input-dir outputs/soft_scm_holdout_new
```

仿真/采集入口默认有弹窗，`--headless` 关闭；离线报告入口不运行仿真、不打开窗口。
软化后端的通用默认值为 tau=.05、zeta=1；上述经校准的结果须显式传入 .008/.25。
`compare_physics.py --engine five` 可同时运行五个模型。

固定参数的跨 mu=.2/.8、h=.001/.004 验证数据在 `outputs/soft_scm_parameter_holdout/`，
共完成17配置、2550个新输入。四种非胶囊形状的组合全部完成；部分胶囊配置在参考静止准备阶段
失败或未评估，失败日志与范围见 manifest.json。主要线速度/惯性加权误差继续低于当前 ComFree，
高摩擦下 sticking 角速度误差仍可能更高，尚不能宣称全面优于 ComFree。
核心验证开始时与一个短轨迹采集进程重叠，当前研究不使用这些计时宣称软化模型的速度优势。

连续滑动1000步的五模型轨迹与Section IV统计在 `outputs/sliding_soft_scm/`；
新模型数据来自新的采集，四个基线复用同初始状态、相同参数的 `sliding_amplitude/`，来源记录在 metrics.json。
接触原始数据保留在各源目录。新的最终动能约5.73e-8 J，位置轨迹RMSE .0742m，ComFree为 .0956m。
此结果仍不能替代准动态 CSO 的单位/公式一致性、gradient 和 contact-guidance 验证。

论文 Fig. 2 解释三种模式的面激活；**Section IV 没有按三种模式统计 prediction error**。
以上单步误差是本项目新增实验。Section IV 实际指标：

| 论文部分 | 原指标 | 本目录入口 |
|---|---|---|
| IV-A / Fig. 3 | 所有检测接触的穿透 mean±std，mm | `penetration_mean_std.csv/.md` 表格；不是每步最大穿透的平均 |
| IV-B / Fig. 5 | 球 z 角速度、圆柱滚动 COM 速度随仿真步数的轨迹 | `torsional_velocity` / `rolling_velocity` 的 CSV、PNG、PDF |
| IV-C / Fig. 6 | 方块水平速度、z 的轨迹；增益图横轴为步数，步长图横轴为秒 | `stability_trajectory_step` / `stability_trajectory_time` 的双子图与原始 CSV |
| IV-D / Fig. 7 | 全仿真步耗时与总接触数的关系图，512 并行环境 | `runtime_vs_contacts.csv/.png/.pdf`，注明本实验单环境；不声称复现 512 环境负载 |
| IV-E / Fig. 8 | 不同并行环境数的吞吐量 mean±std | 当前未测量；不从单环境耗时伪造该项 |

```bash
python compare_physics.py --scene primitives --engine three --steps 1000 --output-dir outputs/penetration_three
python compare_physics.py --scene ball_rotation --engine three --steps 1000 --initial-speed 3 \
  --friction-coefficient 0.01 --output-dir outputs/torsion_three
python compare_physics.py --scene cylinder_rolling --engine three --steps 500 --initial-speed 1 \
  --friction-coefficient 0.01 --output-dir outputs/rolling_three
python compare_physics.py --scene sliding --engine three --steps 1000 --output-dir outputs/sliding_three
```

默认均开窗口，只有 `--headless` 关闭。扭转/滚动系数分别可取 .001、.01、.05、.1；
IV-A ComFree 增益为 (.1,.001)、(.1,.005)、(.3,.001)、(.3,.005)、(.5,.001)、(.5,.005)。
辅助实验不声称场景和数值完全复现论文。
主报告 `paper_metrics.json` 逐项标明对应论文图、单位、统计定义和输出文件。
Fig. 5 的球轨迹保留有符号 z 角速度，避免用模长掩盖反向旋转；Fig. 5/6 保留每一步的数据，不用时间均值替代轨迹。
mean±std 中 std 为总体标准差（ddof=0），不是标准误或置信区间；论文未指定 ddof，本实现明确记录此选择。
旧版 `decay_curves` 的混合三面板图和 max/RMSE 等仍作为历史诊断数据，不作为论文对应的主报告。

已保存的数据可以重新生成这些表格和图，无需再次仿真：

```bash
python paper_reporting.py --input-dir outputs/prediction_modes outputs/torsion_three \
  outputs/rolling_three outputs/sliding_three outputs/penetration_three
```

该命令只处理数据文件，不调用仿真或启动窗口；仿真入口仍然默认弹窗，只有 `--headless` 关闭。
额外输出 `<engine>_contacts.csv.gz`、全接触 mean±std、摩擦功率/离散功、原始速度/高度曲线。
摩擦功率为原生接触切向力/力矩与反作用后相对速度的内积，负值表示耗散；不用 CPU 替代反作用力。
`friction_work_J` 是该功率乘 h 的后点矩形积分诊断，不等同于单步动能的精确变化。
原始点接触 SCM 没有球单点的扭转阻力；`--scm-friction matched` 是额外 6D 扩展，须单独注明。

## 先前迁移的抓取入口

完整 Franka 成对对比（3320 步，6.64 秒仿真时间）：

```bash
python compare_franka.py --engine both --output-dir outputs/franka
```

默认打开原生可视化窗口，各后端按顺序显示。也可以同时包含 MuJoCo CPU，或使用 `--headless` 关闭窗口：

```bash
python compare_franka.py --engine all --output-dir outputs/franka_all
python compare_franka.py --engine both --headless --output-dir outputs/franka_headless
```

短运行用于检查依赖与接口，不评价抓取成功；`--steps-scale` 会改变动作时序，正式对比使用默认值 1：

```bash
python compare_franka.py --engine both --headless --max-steps 10 --output-dir outputs/smoke_franka
```

基础场景包括 125 个物体掉落/堆叠、球的扭转摩擦、圆柱滚动：

```bash
python compare_physics.py --scene primitives --engine both --steps 1000 --output-dir outputs/primitives
python compare_physics.py --scene ball_rotation --engine both --steps 1000 --initial-speed 1 --output-dir outputs/ball_rotation
python compare_physics.py --scene cylinder_rolling --engine both --steps 1000 --initial-speed 1 --output-dir outputs/cylinder_rolling
```

基础场景也默认弹出窗口，只有 `--headless` 关闭窗口。球的初始速度单位为 rad/s，圆柱为 m/s；圆柱同时设置匹配半径的滚动角速度。
可以用 `--timestep`、`--contact-stiffness`、`--contact-damping` 更改参数。
基础场景的凸碰撞迭代上限 `--ccd-iterations` 默认 100：上游默认 35 在堆叠场景中出现迭代不足提示，
用 35 可以恢复原 XML 默认值；该参数对所有后端使用相同设置。
ComFree 默认刚度 0.2、阻尼 0.001，来自上游抓取示例；它们不是针对所有场景优化后的参数。
`--device cuda:0` 指定 GPU；缓冲区不足时使用 `--nconmax` 和 `--njmax` 调大，脚本会拒绝溢出结果。
所有模型路径相对于脚本文件解析，运行不依赖当前工作目录。

## 结果解释

每次运行在 `--output-dir` 下生成：

- `<engine>.npz`：完整数组；包含初始状态以及每一步结束后的状态。
- `<engine>.csv`：对应的时间、qpos/qvel 和诊断；Franka 还包含 ctrl、方块位置、阶段编号。
- `metrics.json`：版本、参数、上游版本、各后端汇总及相对 MJWarp 的差异。

Franka 记录最终/最小/最大方块中心高度，以及上游的成功标准：完整运行结束后 z > 0.18 m，且全程最小 z > 0.115 m。
失败抓取属于比较结果，默认仍保存结果并正常退出；用 `--require-success` 可要求完整抓取成功。

`max_penetration_m` 是每个输出状态上 `max(0, -contact.dist)` 的最大值，采用同一 MuJoCo CPU 碰撞流程计算几何诊断。
它不使用 CPU 接触力来推进 GPU 轨迹，也不等同于论文所有接触穿透样本的均值/标准差。
基础场景还输出动能和速度信息。`qvel_rmse_per_dof` 保留每个自由度的差异，平移和转动分别有 m/s 与 rad/s 单位；
Franka 另有方块位置 RMSE/max（m）。

这是各自推进的整段轨迹比较。轨迹误差会积累，不能直接当作“同一状态下单步预测误差”；
共享状态单步预测使用 `compare_prediction.py`，其误差统计单独保存。
这里包含 CPU 诊断、GPU 同步和文件输出，不用于测量上游论文中的仿真吞吐量。

## 首次迁移的完整运行记录

在本机 RTX 5060 Ti、MuJoCo 3.6.0、Warp 1.16.0 上完成：

- 完整 Franka 3320 步，三种后端均完成抓取；结果位于 `outputs/franka_all/`。
- 球旋转、圆柱滚动、125 物体场景各 1000 步，ComFree/MJWarp 状态均为有限数值；结果位于对应 `outputs/<scene>/`。
- 核对相同初始状态、Franka 相同控制、时间与数组尺寸、CSV/NPZ/JSON 输出，以及 XML/网格/纹理与固定上游版本一致。
- 兼容补丁重复执行和从其他工作目录启动均通过检查。

`outputs/<scene>/run.log` 保留本次运行的完整日志。堆叠场景在凸碰撞迭代上限 100 下仍有迭代不足提示；
当前参数下最大几何穿透约为 MJWarp 71.0 mm、ComFree 64.5 mm。
这些结果体现了当前场景与参数下的数值行为；评估论文中的精度还需要对照其参数设置、碰撞收敛情况与统计口径。
基础场景的有限数值检查只说明输出没有 NaN/Inf。

切换到 `pi0` 后，已验证 Franka 的 ComFree/MJWarp 无界面短运行（10 步），以及两个入口的可视化开关分支：
默认调用窗口启动接口，`--headless` 跳过窗口。`pi0` 共享包仍为 MuJoCo 3.11 / Warp 1.17。

本次五种形状各 300 个共享状态（每种模式 100 个），共 1500 个状态、4500 次预测，
数据保存在 `outputs/prediction_modes/`。三种参考模式各有 500 个状态，均完成、输出有限。
当前 h=.002、mu=.5、ComFree (k,d)=(.1,.001)、SCM r=1e-8、dynamic 模式下，
separation 的三模型结果接近；sliding / sticking 的 SCM 误差明显，
这批数据不能直接支持“SCM 在多数接触状态下准确”的结论，需要按目标 CSO 的状态与输入分布进一步评估。

测试命令（在本目录运行）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

覆盖全接触加权统计、静止方向、模式标签、摩擦功率符号/坐标系、
SCM GPU 与 NumPy 对角公式一致性、查询重置、无接触时四模型自由运动一致性，以及默认弹窗/显式 headless 分支。
26 项测试通过，SCM、幅值修正、有限恢复版本的 dynamic/increment、单点/多点分别核对，
另核对非对角惯性的幅值修正、精确输入回放和有限恢复版本冻结几何下的力雅可比（默认与校准参数）；
报告测试额外检查有符号扭转速度、全接触 mean±std，以及拒绝用旧数据的每步最大值替代全接触统计。
GPU 项需要 CUDA。当前执行环境没有显示服务，实际弹窗未打开；窗口分支使用 mock 验证。

`outputs/penetration_three/` 补充了三个模型各 1000 步的全接触穿透表。
该次运行仍有上游 CCD 迭代不足提示，日志保存在 `run.log`，统计呈现对齐不代表碰撞检测已证实收敛。
穿透表需要结合速度/高度轨迹判断物理行为；SCM 接触次数较少时，较小的穿透均值本身不意味着更高仿真质量。

## 来源与许可

`assets/franka_cube_grasp/` 原样复制自上游同名目录；`assets/physics/` 三个原始场景复制自 `benchmark/test_data/`，
新增 `sliding.xml` 和 `compare_prediction.py` 生成的模型为本地实验场景。
`compare_franka.py` 的 IK、控制时序和外力沿用上游抓取脚本，其余比较与导出实现为本地新增。
保留了上游仓库许可说明 `UPSTREAM_LICENSE.txt` 和 ComFree Core 的 `COMFREE_CORE_LICENSE.txt`。
ComFree Core 使用上游的非商业学术研究许可；其内置 MuJoCo Warp 使用 Apache 2.0，第三方资产沿用各自许可。

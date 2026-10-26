# SCM（3 GS）与完整 Coulomb NCP 的接触动力学对比

入口：`benchmark_collision_prediction_ncp.py`。
复用 `benchmark_collision_prediction.py` 的双自由刚体场景、CoACD 资产流程、采样配方、
20 ms 时间窗、误差指标与实测接触分类，以及
`benchmark_collision_prediction_isaacgym.py` 的原生 PhysX worker。
原有两个入口的默认行为保持不变。

## 上游调查

检查了 [DAIRLab/contact-nets](https://github.com/DAIRLab/contact-nets) 的提交
`c3dc21c94577ff4ed99123f0b00bfe57bb2d1d90`：

- [`contactnets/interaction/lcp.py`](https://github.com/DAIRLab/contact-nets/blob/c3dc21c94577ff4ed99123f0b00bfe57bb2d1d90/contactnets/interaction/lcp.py)
  装配 Stewart–Trinkle 速度级动力学，包含完整的 `J M^-1 Jᵀ` 和 `phi / dt`。
  摩擦通过有限个 `G_bases` 方向离散化，且求解前加 `0.001 I`。
- [`contactnets/utils/solver_utils.py`](https://github.com/DAIRLab/contact-nets/blob/c3dc21c94577ff4ed99123f0b00bfe57bb2d1d90/contactnets/utils/solver_utils.py)
  调用 `lemkelcp`，不是精确圆形 Coulomb 锥的 NCP 求解器。

因此仅参考上游动力学装配，独立实现 `models/ncp_contact.py`；没有把上游 LCP 更名为 NCP，
也无需安装其旧 PyTorch/训练依赖。未复制上游实现。

## NCP 方程与求解

每个点接触有一个法向冲量和两个切向冲量，单位 N·s：

```text
v_free = v + h a_smooth
W      = J M^-1 J^T
v_next = v_free + M^-1 J^T p
u      = J v_next
w_n    = u_n + phi/h

0 <= p_n  perpendicular  w_n >= 0
||p_t||_2 <= mu p_n
p_t in argmin_{||z||_2 <= mu p_n} u_t^T z
```

保留完整质量矩阵、刚体之间以及所有候选接触之间的耦合；使用**圆形** Coulomb 摩擦锥。
没有 Delassus 对角化、摩擦棱锥/box、softplus、接触顺应性或冲量正则项。
法向方程不加 `mu ||u_t||`，避免关联锥优化引入滑动抬升。

用以下不光滑自然映射的零点求解，无平滑近似：

```text
R_n = [p_n - max(0, p_n - rho w_n)] / rho
R_t = [p_t - proj_disk(mu max(p_n,0))(p_t - rho u_t)] / rho
```

解析广义 Jacobian + SciPy TRF 求解；冗余接触先用约束生成寻找活跃接触，
随后核验**所有原始候选点**的完整自然映射，只有全部通过才接受。
这一步不合并接触、不丢弃未检验约束。必要时对完整问题使用投影迭代和 TRF 重试。
投影迭代以残差控制收敛，有迭代预算；不是固定 sweep 的近似模型。
`rho` 和变量尺度只用于数值求解，不改变上述方程。

基准默认验收 `max |R| <= 1e-6 m/s`，可通过 `--ncp-tolerance 1e-7` 收紧。
数值容差不代表接触律的平滑或正则化。`optimizer_success=True` 本身不足以验收；
未达残差要求的有限迭代也记录 `failed`，不发布预测终态。
还保存法向不等式违反量、锥违反量、互补乘积和最大耗散缺陷。

这里“完整”指完整耦合的三维刚性**点接触律**。时间仍为一阶速度级冲量离散化，
每个子步冻结几何和 Jacobian；不声称连续时间/位形完全隐式求解。
恢复系数为零；沿用原场景 `condim=3`，无扭转或滚动摩擦。

## 几何与公平对比

| 方法 | 时间推进 | 接触信息 |
|---|---|---|
| `scm` | 20 ms × 1 | FCL 起点 manifold，现有 smooth Coulomb box，3 GS sweeps |
| `ncp` | 20 ms × 1 | **同一份** FCL 起点 manifold，完整 NCP |
| `ncp_substeps` | 2 ms × 10 | 首步相同 manifold，其后独立重检测并解 NCP |
| `mujoco` / `isaacgym` | 2 ms × 10 | 原生碰撞、原生求解，仅作为数值对照 |

报告同时以 `ncp`、`ncp_substeps` 和选定原生引擎为比较基准：
**`reference=ncp` 的 SCM 行**直接比较相同步长、相同接触几何下的接触律差异；
子步结果用于观察时间离散化与重检测的影响。原生引擎对照还包含碰撞流形与软接触差异，
不能将这类误差全部归因于求解器或称为真实物理误差。

- 使用 FCL convex/sphere/box 的碰撞和距离查询。地面按所有凸顶点计算解析平面距离，
  保留方块四个底角，避免 FCL 单支撑点丢失平面接触力矩。
- 圆柱显式为 96 边凸柱，SCM/NCP 共用这一几何。MuJoCo 对照是解析圆柱；
  这一几何差异保留在对照误差里，不声称完全相同。
- 复杂物体逐个 CoACD 凸块处理，不把整个凹物体替换为单凸包。默认真实尺度阈值 1 mm、
  seed=42，显式 decimate 到每块至多 64 顶点，`--coacd-max-vertices` 可调整。
  参数和版本参与内容哈希缓存；质量、质心和惯性沿用源模型，不按重叠块叠加。
  不声称顶点简化后仍有严格 1 mm Hausdorff 误差保证。
- 复用旧查询时**直接复用旧凸块**，不重新分解或简化，其源文件哈希接受验证。
- 默认候选距离为 10 mm，记录有符号 gap；正 gap 直接进入 `phi/h`，不强制加载。
  它是离散碰撞候选范围，不是 CCD；高速运动可用 `--candidate-margin` 做敏感性测试。
- FCL 近相切时 libccd 偶尔返回距离失败；先重试 FCL independent GJK。
  若球–凸块表面见证退化，则对**同一凸几何**使用球心到三角形的最近点恢复法向和 gap。
  不膨胀形状、不借用 MuJoCo 接触；重试与修复计数记录在 manifest。
- 不按 CoACD 接缝去重；超容量即失败，不静默截断。MuJoCo 仅向预测器提供
  编译惯性、自由运动、点 Jacobian 和位姿积分，预测器不调用其碰撞或约束求解器。
- 采样准备可用 MuJoCo 落定，独立于预测计时。标签从原生参考的接触力和速度**实测**，
  不将 `requested_mode` 当作标签，不保证小规模 smoke 覆盖所有实测类别。

## 运行

仓库根目录执行，输出目录必须不存在：

```bash
python -m pip install -r requirements-ncp.txt
ENTRY=examples/mpc/fingertips/test/benchmark_collision_prediction_ncp.py

# 无需 GPU：三个基本体和一个有五个 CoACD 块的复杂物体。
python "$ENTRY" run --objects sphere cube cylinder foam_brick \
  --samples-per-mode 2 --output-dir outputs/ncp_smoke

# 使用已安装的原生 IsaacGym worker；缺少 SDK 时明确失败，不退回 MuJoCo。
python "$ENTRY" run --objects sphere cube cylinder foam_brick \
  --samples-per-mode 1 --reference-engine isaacgym \
  --isaacgym-python /home/lab423/miniconda3/envs/acc/bin/python \
  --output-dir outputs/ncp_isaac_smoke

# 完全复用旧 MJWarp 或 IsaacGym 已选查询、外力、标签和终态，不重新采样。
python "$ENTRY" run --input-dir outputs/OLD_COMPLETE_DATA \
  --reference-engine saved --output-dir outputs/ncp_saved
# 可用 --objects sphere cube --limit-per-object 8 做小规模验证。

# 离线重新统计 / 确定性重放 NCP。
python "$ENTRY" report --input-dir outputs/ncp_smoke
python "$ENTRY" replay --query outputs/ncp_smoke/cube/queries/cube_s42_sticking_000000.npz

OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_ncp_contact.py tests/test_collision_prediction_ncp.py \
  tests/test_collision_prediction_external_contacts.py

# 十物体，每个配方 20 条，共 800 条查询；实测分类可能不同。
nohup python -u "$ENTRY" run \
  --objects sphere cube cylinder elephant foam_brick mug piggy_bank rubber_duck stanford_bunny2 teapot \
  --samples-per-mode 20 --output-dir outputs/ncp_full \
  > outputs/ncp_full.log 2>&1 < /dev/null &
```

## 输出与已验证结果

`manifest.json` 保存实际参数、依赖版本、运行状态、失败数、几何修复计数和源码哈希；
`implementation/` 保存运行源码快照。`results.log` 逐查询刷新；`predictions.csv` 与
`summary.csv` 提供原有绝对/相对误差和 object×实测mode×起点接触数分组，
`summary.log` 是便于查看的池化摘要。
`timing.csv` 汇总各物体/方法的 CPU 耗时均值、中位数与 p95；起点碰撞另计，
子步 NCP 包含后续重检测，原生参考计时还包含诊断读取，不能解释为 GPU 吞吐加速比。
每个物体有独立模型/资产、`solver_diagnostics.jsonl` 和 `queries/*.npz`。
NPZ 保留初态、外力、初始 manifold、各方法终态与轨迹，以及各 NCP 子步的
`J`、`M^-1`、`v_free`、有符号 gap、摩擦、冲量，可独立复核方程。
失败不填零；空组和相对误差分母过小的项继续使用原有未定义统计。
不对算法失败的样本只报告成功子集均值而隐藏失败数。

2026-10-27 在本机 Python 3.12、MuJoCo 3.12、FCL 0.7.0.11、CoACD 1.0.14 上验证：

- MuJoCo 对照：四物体共 32 条输入，SCM、NCP 单步及 NCP 十子步均成功。
- 原生 GPU PhysX 对照：同四物体共 16 条输入，上述方法均成功，无引擎回退。
- 回归测试：28 passed，2 skipped（原有可选 CUDA parity 测试）。
- 求解器与集成测试覆盖解析单点解、圆盘滑动、无 dilatancy、完整惯性耦合、
  切向基变换不变性、冗余接触、广义 Jacobian、失败验收、FCL 法向、禁止调用
  MuJoCo 接触求解器，以及保存/回放。

这些是 smoke 结果，不是统计结论；完整批次会继续保留并报告数值失败。

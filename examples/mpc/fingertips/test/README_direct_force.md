# Lambda 接触点直接施力

`test_scm_mujoco.py` 和 `test_mpc_isaac.py` 现在共用
`examples.mpc.franka.ik2.params.build_lambda_optimizer`。
保留 Isaac 原有准动态模型，SCM 预测也直接调用该优化器的同一 CasADi 图。
原先独立的六自由度物理 SCM/LCP 不再用于这条执行路径。

| 默认模型参数 | 两端共同值 |
| --- | --- |
| 接触响应 | 九自由度、分组耦合 soft-SCM、quasi-dynamic |
| 预测周期 | 0.02 s |
| Q 对角线 | [50, 50, 50, 0.05, 0.05, 0.05] |
| 驱动刚度 | 600 |
| SCM 响应时间 / 阻尼比 / 正则项 | 0.05 s / 1 / 1e-8 |
| 最优驱动力上限 | 3 N，来自 min(10 N, 600 × 0.005) |
| 摩擦锥 / 指尖接触摩擦系数 | pyramid / 1.2 |
| 摩擦 / 力正则权重 | 0.03 / 0.01 |
| 环境接触数上限 | 4 |
| 目标代价 | 共用 ObjectCostStages：翻正后达到目标位姿 |

Q 是准动态模型的等效矩阵，不是物体真实质量矩阵。默认预测初速度沿用 Isaac
的零速度约定。测试分别从两端入口解析默认参数，核对模型参数、采样点和矩阵；
还将 SCM 导出的预测与 Isaac 的 NumPy 实现、ACADOS 候选预测图比较，容差 1e-12。
这证明的是 lambda_optimizer 一致；完整 MuJoCo 和 PhysX 仿真器仍有差异。

## 执行修复

每 20 ms 从真实 MuJoCo 状态和桌面接触重新选择 `best_contact_point`，
把对应最优力从接触系变换到物体系，在该表面点用 `mj_applyFT` 施加。
**默认完整力施加 0.5 ms，随后 19.5 ms 撤力推进物理仿真。**
这样保留最优力的幅值和方向，通过执行时长适配准动态预测与真实惯性之间的差异。
该执行器是理想点力执行器，不包含实体指尖的碰撞响应。

物理子步为 0.5 ms，每周期 40 个子步；每个子步更新随物体运动的作用点和方向。
运行中物体位姿和速度由 `mj_step` 产生，不写预测位姿，不使用物理仿真预演选择力。
默认不裁剪力或力矩。可用 `--force-apply-duration` 调整脉冲长度；
使用其他长度后的成功率需要重新验证。

初始化时，物体质心和惯性轴对齐参考 Isaac 生成 URDF 的零偏移/单位轴，
惯性对角线使用相同包围盒公式。桌面与物体的滑动摩擦设为参考有效配对值 0.7；
关闭 SCM 未建模的扭转/滚动摩擦约束。关闭实体指尖碰撞，避免额外推动物体。
停滞时使用已有接触进度反馈切换接触区域；翻正后的阶段判断采用控制周期内观测运动。

进入目标附近后撤力，若漂出阈值则继续规划。位置误差须小于 0.02 m，
姿态误差 `1 - (q · q_target)^2` 须小于 0.015；至少连续 10 个无外力周期达标，
并且平动速度小于 0.05 m/s、角速度小于 0.5 rad/s，才报告成功。

JSON 中 `optimal_force_contact` 是优化结果，`applied_force_body` 是脉冲瞬时力/力矩，
`force_body` 是整个控制周期的平均力/力矩，`actual_applied_impulse_body` 是实际积分冲量。
停稳期间实际施力为零；预测驱动力、预测接触反力和实际脉冲冲量是不同量。

## 运行与验证

在仓库根目录运行（本次使用 pi0 环境）：

```bash
conda run --no-capture-output -n pi0 python examples/mpc/fingertips/test/test_0902.py \
  --direct_force --headless --obj foam_brick --trial_num 10 \
  --max_rollout_length 5000 --force-output output/direct_force/foam_brick.json
```

`--obj` 可替换为 `elephant` 或 `stanford_bunny2`。查看仿真时把 `--headless`
替换为 `--viewer`。JSON 和同名 CSV 保存逐步轨迹；达到步数上限仍未成功则返回非零退出码。
原 SCM 入口也使用同一执行流程：

```bash
conda run --no-capture-output -n pi0 python examples/mpc/fingertips/test/test_scm_mujoco.py \
  --surrogate --steps 5000 --force-output output/direct_force/scm.json
```

2026-10-13，MuJoCo 3.11.0 / CasADi 3.7.2，种子 0–9，默认随机初始倾斜，
每轮上限 5000 步的实测结果：

| 物体 | 成功并停稳 | 完成步数范围 | 最大最终位置误差 | 最大最终姿态误差 |
| --- | --- | --- | --- | --- |
| foam_brick | 10/10 | 488–752 | 0.016515 m | 0.002520 |
| elephant | 10/10 | 172–3351 | 0.016064 m | 0.000125 |
| stanford_bunny2 | 10/10 | 171–1328 | 0.016787 m | 0.004135 |

合计 30/30，未发生 MuJoCo 不稳定，未触发力裁剪。原 SCM 入口另测默认物体，
488 步成功并停稳。轨迹、汇总与日志位于
`output/diagnostics/scm_shared_model_20261013/`，`summary.json` 是本次最终结果。
早期失败试验保留为压缩文件。旧 `direct_force_20261013/` 对应先前不同模型的结果，
不能用作本次统一模型的验证。上述成功率只覆盖已测试状态和默认执行参数。

```bash
SCSP_PLANNER_ONLY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  conda run --no-capture-output -n pi0 python -m pytest -q \
  tests/test_direct_contact_force.py tests/test_grouped_contact_response.py \
  tests/test_shared_cartesian_contact.py tests/test_franka_contact_limits.py \
  tests/test_franka_scene_friction.py
```

43 项通过，包含两端默认参数与预测一致性、实际点力/力矩、脉冲冲量、
旋转时作用点更新和任务完成后无外力停稳验证。`--execution pose` 保留原实验模式，
不作为实际施力成功的证据；独立 QP/LCP 模式不能替代该共享 soft-SCM 模型。

## 耗时统计与独立 QP/LCP 对比

新增统计口径：

- `execution_time_s`：从首轮规划开始到成功停稳的墙钟耗时，包含规划、物理推进
  和循环内处理，不包含场景初始化或结果文件写入。失败时为 null，另存已运行时间
  `elapsed_wall_time_s`；成功耗时的均值只使用成功任务。
- `planning_time_ms`：每一轮完整接触点与力规划的墙钟耗时，包括全部候选点的
  力求解、预测和选择，排除实际物理执行。统计包含成功和失败任务的规划轮次。
- `simulation_time_s`：MuJoCo 仿真时间，独立于墙钟耗时。

JSON 和 CSV 保存逐轮原始数据；任务汇总保存 mean、std、median、p95、min、max、n。
统计 std 是总体标准差。成功任务耗时为 null 时不能当作 0 秒参与均值。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n pi0 \
  python examples/mpc/fingertips/test/benchmark_contact_models.py \
  --objects foam_brick,elephant,stanford_bunny2 --models scm,qp,lcp \
  --trials 10 --steps 5000 --output-dir output/contact_models
```

这会运行 3 物体 × 10 种子 × 3 模型，共 90 轮任务。顺序运行，避免方法间竞争 CPU。
相同物体/种子的初始位姿会在汇总时自动核对。逐任务保存轨迹、时序和失败原因；
`summary.json`、`summary.csv`、`summary.md` 给出总体及分物体统计。
`--trial-start` 指定起始种子；`--resume` 继续相同参数的已有结果，
`--analyze-only --output-dir ...` 重新汇总。旧轨迹没有墙钟计时，不能反推 execution time。

SCM 保留参考 Isaac 模型与 ACADOS 求解方式。QP/LCP 在同样的候选点上，
各自用自身模型优化力，然后通过同一个理想点力执行器驱动物体；
不是只替换执行仿真器，也不是用 SCM 规划后再给另一个模型评分。
两种物理预测器都把实际脉冲冲量平均到一个 20 ms 预测步；执行仍为 40 个物理子步。

QP 使用独立 MuJoCo model/data：金字塔摩擦锥、原生 Newton 接触约束求解，
每个候选力都重新求解接触 QP。在固定观测状态下缓存碰撞与速度相关量，
通过 `mj_forwardSkip` 重算力相关动力学；测试与完整 `mj_forward` 的结果核对。
这是原生软接触 QP，参见
[MuJoCo 官方约束求解说明](https://mujoco.readthedocs.io/en/stable/computation/index.html#constraint-solver)。

LCP 使用真实广义质量矩阵和耦合摩擦射线接触 Jacobian：
`W = J M^-1 J^T + 1e-8 I`，`g = J v_free + phi/h`，
求解 `0 <= p ⟂ Wp + g >= 0`。
调用仓库已有的 QuantEcon Lemke 库，进行对角缩放，并在原未缩放系统中检查
非负性、松弛非负性和互补残差；残差超过 1e-6 不接受。
这是正则化摩擦射线 LCP，不代表所有刚性 Coulomb 接触 LCP 形式。

**QP 是隐式动力学：外层力优化的每次模型计算都包含内层接触 QP。**
当前基线使用数值局部敏感度：在同一观测状态、同一接触点的力种子上，
对三个力分量各扰动 1e-4 N，重新求解 QP/LCP，构造预测位姿关于力的 Jacobian。
外层 SLSQP 在该局部线性预测上优化完整位姿代价和力正则项，保持原有力界和摩擦锥；
局部目标的梯度由 CasADi 计算。随后用投影后的可行力重新求解真实接触模型，
以真实预测位姿给该候选点评分。

每点目前只进行一次局部动力学线性化，属于近似局部力优化；没有迭代更新到
原隐式问题收敛，也没有使用 KKT 解析敏感度。接触活跃集切换可能使有限差分
敏感度不稳定。SLSQP 的收敛标志仅表示局部子问题状态，
`inner_nonconverged` 会保留停止/未收敛计数；使用的是可行局部结果，不会退回 SCM。
因此 planning time 对比包含 ACADOS 与数值敏感度/SLSQP 实现的差异，
不能解释为只更换接触模型方程产生的耗时差异。

2026-10-13 配对流程验证：三个物体各使用种子 0，每模型 3 个任务，上限 5000 步。
下表是均值 ± 总体标准差；execution time 按任务统计，planning time 汇集逐轮数据。

| 模型 | 成功并停稳 | execution time (s) | planning time (ms) |
| --- | --- | --- | --- |
| SCM | 3/3 | 5.810 ± 2.232 | 15.812 ± 3.454 |
| MuJoCo QP | 3/3 | 24.092 ± 12.597 | 100.429 ± 11.441 |
| LCP | 3/3 | 40.643 ± 32.651 | 97.323 ± 10.946 |

完整数据在 `output/diagnostics/contact_models_timing_final_20261013/`。
初始状态配对核对通过，三个模型各自完成了真实物理执行。
这是 9 轮配对流程验证，不是上述 90 轮全量实验；不能据此推断大规模成功率。
SCM 的既有 30/30 验证仍保留在前述共享模型目录，其旧数据未记录墙钟计时。
`contact_models_timing_partial_20261013/` 保留额外试跑和中止记录，
不并入这个完整配对组的均值。新增计时、基线独立性和退化 LCP 测试后，相关回归共 58 项通过。

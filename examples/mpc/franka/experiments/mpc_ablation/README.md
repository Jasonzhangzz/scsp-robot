# Franka MPC 消融入口

从仓库根目录运行，使用与 `ik2/test_mpc_isaac.py` 相同的 Python/Isaac 环境：

```bash
python examples/mpc/franka/ik2/test_mpc_ablation.py --a1 --headless --sync-planner --obj elephant
python examples/mpc/franka/ik2/test_mpc_ablation.py --a2 --viewer --sync-planner --obj elephant
python examples/mpc/franka/ik2/test_mpc_ablation.py --a3 --obj elephant
python examples/mpc/franka/ik2/test_mpc_ablation.py --a4 --headless --sync-planner --obj elephant
python examples/mpc/franka/ik2/test_mpc_ablation.py --a5 --headless --sync-planner --obj elephant
python examples/mpc/franka/ik2/test_mpc_ablation.py --a6 --headless --sync-planner --obj elephant
python examples/mpc/franka/ik2/test_mpc_ablation.py --a8 --headless --sync-planner --obj elephant
```

实现入口也可直接运行：`examples/mpc/franka/experiments/test_mpc_ablation.py`。
每次选择一个消融；不指定时为 `full` 基线。也支持 `--ablation-mode a1` 等形式。

与基线共享的初始到目标 XY 距离：`--rotation` 为 0.20–0.30 m，`--flipping` 为 0.12–0.20 m。目标在 Franka 世界系的 `x ∈ [0.30, 0.70] m, y ∈ [-0.30, 0.30] m` 内采样；目标高度由物体网格和目标姿态确定。平移使用独立的 seed 随机流，固定试验 seed 的初始姿态、目标姿态及物理参数采样保持一致。

| 参数 | 行为 |
| --- | --- |
| `--a1` | 只把 best-contact 的球心跟踪点 `best_contact_point_world` 送入 MPC，不求解或切换到 p_arm。沿用基线 best-contact 的 top-k guide 选择。 |
| `--a2` | 只跟踪独立求解的 `p_arm_world`。跳过 best-contact 排序、接受判据和到达后的 best-contact 替换；有真实接触时沿用实测接触几何。 |
| `--a3` | 离线拟合物体 mesh 的 OBB，并缓存为 actor-frame box mesh。Lambda 每轮在 OBB 面上搜索候选接触点，用当前/目标位姿评价并生成 `best_contact_point_world`；MPC、cost、verify/RS、真实 mesh 碰撞和执行器保持基线架构。 |
| `--a4` | 在每次规划前把物体位姿、速度、重力、质量、惯量、碰撞几何和桌面同步到独立 MuJoCo 预测场景。使用 `test_scm_mujoco.py` 同一个 `PointContactBaseline('qp')` 原生接触 QP 预测器生成候选得分；外层力优化为 SLSQP。不回退到 SCM。 |
| `--a5` | 使用原有 `ForcePopulationSearch('cma-es')` 求解相同的 Lambda 接触力目标及可行力域。MPC 仍使用原来的 solver。 |
| `--a6` | 用 AdamW 直接优化 MPC 整段 Cartesian 控制序列。ComFree 前向 rollout 计算整段 cost，对每个控制维度做中心差分得到数值梯度，不构建 MPC/CSO 的 SQP 或 IPOPT 求解器，不求解 best_contact_point 或 p_arm。 |
| `--a8` | 从首步固定使用基线第二阶段 `pose` cost，包括位置、完整四元数、upright 和 tilt barrier；不进入 `flip`，不发生阶段切换或回退。保留原动力学与稳定翻转统计。 |

A1/A2 让 MPC 直接优化指定接触点的跟踪轨迹，屏蔽 lift/place via 与 verify 切换；保留物体目标、控制代价、动力学、力限制和执行约束。球心目标包含与基线相同的指尖半径及表面间隙。

场景/OSC/接触几何和参数基于当前 `runtime/`：参数继承 `runtime.params.ExplicitMPCParams`，执行器直接复用 `runtime.mpc`，除 A6 外的求解器复用 `planning.mpc_explicit.MPCExplicit`；A6 使用独立的 `adamw.py` 规划循环与 `planning/franka_adamw_ablation.py` 求解器。本目录的通信、rollout 和请求生命周期以当前 runtime 为基准，仅改动消融分发和统计。旧 `experiments/ik2` 消融脚本不参与新入口。

基线参数继续有效，包括 `--obj`、`--rotation`、`--flipping`、`--trial-start`、`--trial_num`、`--max_rollout_length`、`--sync-planner`、`--async-planner`/`--trial`、`--headless`、`--viewer`、`--num-envs`、`--physics-backend mjwarp` 和 `--solver-backend`。A6 固定使用 Isaac Gym/PhysX 执行，传入 `--physics-backend mjwarp` 会报错；其 `--solver-backend` 不影响 AdamW。其他模式继续支持上述后端。这些消融使用 Franka OSC；`--floating-fingertip` 和替换整个规划器的 `--sampling-c3` 不适用于消融模式。

可选消融参数：

- `--a4-qp-maxiter 12`
- `--a3-obb-cache-dir PATH`（默认 `~/.cache/scsp_robot/a3_obb`，遵循 `XDG_CACHE_HOME`）
- `--a5-population-size 24 --a5-population-iterations 12`（依赖已有 requirements 中的 `cma>=4,<5`）
- `--a6-center-weight 0.05`
- `--a6-adamw-iterations 30 --a6-adamw-lr 0.1 --a6-adamw-weight-decay 0.01`
- `--a6-fd-epsilon 1e-5`（控制增量的中心差分扰动，单位 m）

A3 的 OBB 只在首次规划器初始化时拟合一次，按 mesh 内容和算法版本缓存，并支持多个试验进程共用缓存。拟合使用 `trimesh.bounds.oriented_bounds`；OBB 的中心偏移和旋转完整保留，当前/目标世界变换为 `T_world_actor × T_actor_obb`，不会把 actor/质心原点改到 OBB 中心。候选点采样自 OBB 的平面面片，使用面法线，避免把八个角的平均法线作为接触法线。候选 guide 不投影回原始 mesh，因此允许出现用户指定的虚拟接触/穿模几何。

Isaac Gym 仍加载真实 mesh，MPC 仍接收原来的实测接触和支持几何、使用原来的动力学及 cost；Lambda 的已观测接触分支也保留基线逻辑。对于 mug，执行路径仍绕开真实杯壁，不用 OBB 替换执行器的避障几何。`--rollout-trace-jsonl` 的 setup/cycle 分别记录目标/当前 OBB 的世界变换，试验指标记录 `lambda_contact_geometry=obb` 和 OBB 拟合信息。

也可以在控制循环外预先拟合（应传入已变换到 actor 坐标系的 mesh，运行时的准确路径会记录在 `a3_obb.source_mesh`）：

```bash
python -m examples.mpc.franka.experiments.mpc_ablation.obb \
  --mesh envs/assets/objects/elephant.stl --cache-dir /tmp/a3_obb_cache
```

A6 使用 `--mpc-horizon`（默认 5）。每轮 AdamW 批量评估 `2 × horizon × 3` 个扰动 rollout，梯度包含接触几何随预测位姿的变化。学习率基于控制上界归一化，例如 `.1` 表示每次约移动控制上界的 10%。每个 MPC 周期平移上一周期的最优控制序列作为初值，重置 AdamW 动量，返回所有已评估迭代中 cost 最低且有限的控制序列；不会退回 SQP 或 Lambda。

A6 的每步 cost 为物体任务 cost、`50‖u‖²` 和 `contact_coef × a6_center_weight × ‖p_tip − center_geometry_world‖²`，终端保留 `10 ×` 物体任务 cost。中心是随预测位姿变换的 actor-frame AABB 中点，为尚未接触时提供靠近物体的梯度。目标仍沿用当前 flip/pose 阶段，但 cost 不依赖 verify、via 或 best-contact 参考。控制范围、首步 slew/floor/edge 范围与接触力上限通过投影执行；预测状态边界使用 L1 + 二次软惩罚。A6 不创建 Lambda 容器、接触 policy、verify/RS/dwell tracker，也不填充 best-contact/p_arm 占位字段。求解器的 cost 参数只有目标位姿和物体任务权重（共 19 维）。每次生成机器人轨迹后，直接把首步控制增量通过 `kind=trajectory` 发送给 Isaac Gym 的 OSC 执行器。`--rollout-trace-jsonl` 保存控制序列和预测机器人轨迹。ComFree 仅用于规划器内部 rollout，物理仿真全部由 Isaac Gym/PhysX 完成。A6 固定使用 ComFree，不支持 online SCM 或 `--tool-orientation-control mpc`。

注意：参考的 `/home/zz/retarget/run.py --adamw` 使用 rollout 自动微分；这里按数值梯度要求使用中心差分，CasADi 只执行前向函数，不生成梯度。数值差分的开销随 horizon 和 AdamW 迭代次数增加。

```bash
python examples/mpc/franka/experiments/test_mpc_ablation.py --a6 \
  --headless --sync-planner --obj elephant --mpc-horizon 5 \
  --a6-adamw-iterations 30 --a6-adamw-lr 0.1
```

`--ranking-metrics-jsonl` 的每条试验记录包含消融模式、后端、阶段、权重以及 A4/A5 求解统计；`--rollout-summary-json` 保存汇总结果。

## Verify 敏感性分析

`--ema` 和 `--threshold` 分别执行一维网格扫描，使用 full 基线、headless 仿真；不能与 `--a1` 等消融同时使用。每个网格点运行相同的 `--trial-start` / `--trial_num`，保持任务、物体和初始状态 seeds 一致；默认同步规划，仍可显式指定 `--async-planner`。每个点的成功率为成功次数除以完整试验次数，任务失败也计入分母；缺失、重复、中止的试验或子进程异常会停止扫描，已完成点的日志和汇总保留。

```bash
# EMA 系数 beta，默认网格 0.045,0.09,0.18,0.36,0.72；基线 0.18
python examples/mpc/franka/ik2/test_mpc_ablation.py --ema \
  --ema-grid 0.045,0.09,0.18,0.36,0.72 \
  --obj elephant --trial-start 0 --trial_num 100 \
  --sensitivity-output output/ablation/ema_elephant --plot

# 接受阈值，默认网格 0.15,0.30,0.45,0.60；基线 0.45
python examples/mpc/franka/ik2/test_mpc_ablation.py --threshold \
  --threshold-grid 0.15,0.30,0.45,0.60 \
  --obj elephant --trial-start 0 --trial_num 100 \
  --sensitivity-output output/ablation/threshold_elephant --plot
```

`--ema` 只改变实际 `verify_cost` 更新系数 `verify_beta`：上升时 `v ← (1−β)v + β·tightness`，下降时沿用基线 `0.35β`。β 必须在 `(0, 1]`；诊断用的 `value_alpha/value_rho` 和旧 window 参数保持不变。

`--threshold` 改变当前有效的 `accept_enter_tightness`，通过新参数 `--verify-accept-threshold` 传入 tracker：同一 patch、靠近 p_arm 时，`tightness < threshold` 才接受，阈值越大越宽松。已接受后的释放阈值固定为 `--verify-release-threshold 0.65`；网格需满足 `0 ≤ threshold ≤ release ≤ 1`。best sample 的直接接受分支沿用基线。这里扫描的不是成功判定阈值，也不是已经不参与接受判定的历史 window/cost 阈值。模型置信度的尺度参数 `--model_cost_error_threshold` 保持原值。

输出目录必须为空；省略 `--sensitivity-output` 时自动创建 `output/mpc_sensitivity/<模式>_<香港时间戳>`。文件包括：

- `grid.json`：网格、固定参数、trial 范围和每个点的完整执行命令。
- `cell_000/run.log` 等：每个点独立的 stdout/stderr；`trials.jsonl` 保存逐次试验结果，`rollout_summary.json` 保存该点的运行汇总。
- `sensitivity.log`：JSONL 格式的网格成功率；`sensitivity_summary.csv` / `.json` 保存参数值、试验数、成功数、成功率和相对基线的成功率差值（0–1 单位）。自定义网格不含基线时差值为空。
- 加 `--plot` 后生成 `success_rate.svg` / `.png`：横轴为 EMA β 或接受阈值，纵轴为成功率。

先加 `--dry-run` 可以检查命令，不导入仿真器、不创建文件。绘图可在已有结果上单独执行，无需重新仿真：

```bash
python -m examples.mpc.franka.experiments.mpc_ablation.sensitivity \
  output/ablation/ema_elephant/sensitivity_summary.json
```

```bash
python examples/mpc/franka/ik2/test_mpc_ablation.py --a5 --headless --sync-planner \
  --obj elephant --trial-start 0 --trial_num 10 \
  --ranking-metrics-jsonl output/ablation/a5_trials.jsonl \
  --rollout-summary-json output/ablation/a5_summary.json

SCSP_PLANNER_ONLY=1 python -m pytest -q examples/mpc/franka/experiments/mpc_ablation/test_ablation.py
```

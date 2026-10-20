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

`--contactsdf` 分发到独立的 `examples/mpc/franka/contactsdf_run.py`：使用官方
ContactSDF 的 SDF 接触检测、`LseDynamic`、IPOPT MPC 和目标公式，仅适配 Franka
单指维度与执行接口。不创建 Lambda，不运行 verify、CSO、approach、分阶段目标或
多初值搜索，不能与 A1–A8 等本地消融组合。固定同步规划、原生 Isaac、每进程一个场景。
来源、适配范围和测试说明见 [ContactSDF 复现说明](../../../../../docs/franka_ik2/CONTACTSDF.md)。

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

`--ema` 和 `--threshold` 分别执行一维网格扫描，使用 full 基线、headless 仿真；不能与 `--a1` 等消融同时使用。`--ema` 默认 `--sensitivity-cohort mixed`（`--threshold` 仍默认单物体 `range`，也可显式选择 `mixed`）：从已有基线结果中固定以下 5 条轨迹（4 success、1 failed）：`stanford_bunny2` 成功和失败各 1 条，以及 `elephant`、`piggy_bank`、`rubber_duck` 各 1 条成功轨迹。物体名称使用代码中的标准名称（bunny2 / elpahnt / duck 分别对应上述名称）。筛选使用最终任务 `success`，不使用中间翻转完成标记。基线成功率为 4/5；后续每个网格点始终运行相同的 5 个物体/seed，不再按扰动后的成功与否重新抽样。失败也计入分母，缺失、重复、中止的试验或子进程异常会停止扫描，已完成点的日志和汇总保留。

混合队列固定使用同步规划、每个子进程 1 条轨迹。基线只读取保存的结果，不运行仿真。`--baseline-results FILE_OR_DIR` 可重复传入，按顺序优先复用对应 JSONL、JSON，或目录中的 `trials.jsonl` / `baseline_summary.json`；旧结果缺少 beta 时从同目录 `config.json` / `source_config.json` 或所属扫描的 `grid.json` 读取。默认查找仓库已有的 `output/ablation/ema_baseline_full_*/selection`、`outputs/rs_table_vi_contact5_*/flipping`、`outputs/rs_current_*/baseline`。只使用对应 beta、任务、后端的完整同步 full 结果；若缺少所需成功/失败案例则报错，不补跑基线。不要同时传入 `--obj` / `--trial_num` / `--async-planner` / `--trial`。若需要原先单物体连续 seeds 的行为，显式使用 `--sensitivity-cohort range`，此时这些运行参数仍可使用。

```bash
# 默认：以当前 beta=0.18 为中心，按 +/-0.02、0.04、0.08、0.16 逐圈扩大
# 顺序：0.18,0.16,0.20,0.14,0.22,0.10,0.26,0.02,0.34
python examples/mpc/franka/ik2/test_mpc_ablation.py --ema \
  --sensitivity-output output/ablation/ema_mixed --plot

# 自定义扰动半径和已有基线结果来源；越界的 beta 自动跳过
python examples/mpc/franka/ik2/test_mpc_ablation.py --ema \
  --ema-offsets 0.02,0.04,0.08,0.16,0.32 \
  --baseline-results outputs/rs_current_20261013_162818/baseline --dry-run

# 接受阈值，默认网格 0.15,0.30,0.45,0.60；基线 0.45；显式使用混合队列
python examples/mpc/franka/ik2/test_mpc_ablation.py --threshold --sensitivity-cohort mixed \
  --threshold-grid 0.15,0.30,0.45,0.60 \
  --sensitivity-output output/ablation/threshold_mixed --plot
```

`--ema` 的每个扰动点直接调用 `examples/mpc/franka/ik2/test_mpc_isaac.py`，使用该入口的仿真、规划器、求解器和默认参数；不经过消融规划器副本。只改变实际 `verify_cost` 更新系数 `verify_beta`：上升时 `v ← (1−β)v + β·tightness`，下降时沿用基线 `0.35β`。β 必须在 `[0, 1]`；诊断用的 `value_alpha/value_rho` 和旧 window 参数保持不变。EMA 模式不接受修改接受/释放阈值；阈值扫描仍通过消融入口的对应参数实现。

EMA 控制链路修复（`verify_control_version=ema_mpc_v1`）：普通接触 MPC 直接使用 tracker 的连续 `verify`，不再根据接触可达性 / `path_blocked` 覆盖为 0/1；接近路径仍使用原几何控制。β=0 冻结评分（初始化为 0），β=1 上升立即响应、下降仍使用 0.35 的更新率。`--cpo-variant direct` 和 `--ideal_contact_pose` 的目标函数不使用这项混合权重，不能用于此 EMA 扫描。

每条新轨迹记录 `verify_beta`、控制版本、`verify_mpc_steps`、实际权重的最小/最大/均值，以及 `verify_mpc_max_abs_error`。扫描器要求实际运行过接触 MPC，且实际权重等于 EMA 评分；否则报错。旧的二值覆盖版本结果可用于选取固定物体/seed，但不能作为修复后控制器的性能基线：汇总中标为 `historical_reference`、`baseline_comparable=false`，相对基线差值为空，绘图不使用这个历史点。不自动重测基线。

旧扫描得到的平坦成功率曲线不能用于说明 EMA 鲁棒性。请将新结果输出到独立目录；旧结果和 SVG 保留以便追溯。重新绘图时应使用新扫描目录。

`--ema-offsets` 为相对当前 `--verify_beta` 的绝对正负偏移，先基线再按半径递增运行，不会因为修改中心而继续扫描旧网格。也可用互斥的 `--ema-grid` 明确列出 beta 值。混合队列下始终补入基线，直接写入已有的 5 条基线结果，不启动子进程。默认 9 个网格点只有 8 个扰动点需要运行，共 40 次仿真；可以修改 `--verify_beta`，但必须提供对应中心值的已有基线记录。

`--threshold` 改变当前有效的 `accept_enter_tightness`，通过新参数 `--verify-accept-threshold` 传入 tracker：同一 patch、靠近 p_arm 时，`tightness < threshold` 才接受，阈值越大越宽松。已接受后的释放阈值固定为 `--verify-release-threshold 0.65`；网格需满足 `0 ≤ threshold ≤ release ≤ 1`。best sample 的直接接受分支沿用基线。这里扫描的不是成功判定阈值，也不是已经不参与接受判定的历史 window/cost 阈值。模型置信度的尺度参数 `--model_cost_error_threshold` 保持原值。

输出目录必须为空；省略 `--sensitivity-output` 时自动创建 `output/mpc_sensitivity/<模式>_<香港时间戳>`。文件包括：

- `grid.json`：网格、固定参数、已有结果来源、选定队列和各扰动点的执行命令；基线点的 `runs` 为空。
- `cohort.json`：固定的 5 个物体/seed、基线成功/失败标签、来源文件及记录下标。已有源文件保持原样，不再生成或运行 `selection/`。
- `cell_*/trials.jsonl`：该点完整 5 条结果，保留 `cohort_id` / `baseline_success`，以物体和 trial 联合标识轨迹；不同物体允许使用相同 seed。扰动点的 `cell_*/<cohort_id>/run.log`、`trials.jsonl`、`rollout_summary.json` 保存每条轨迹的原始输出，基线原始输出由 `cohort.json` 的来源路径定位，`cell_000/trials.jsonl` 保存复用记录的副本。
- `sensitivity.log`、`sensitivity_summary.csv` / `.json`：参数值、试验数、成功数、成功率、相对基线成功率差值（0–1 单位），以及相对所选历史轨迹标签的成功变失败 / 失败变成功条数；这些标签变化在基线版本不一致时不能作为单一参数的因果效应。`range` 模式维持原有 `cell_*/run.log` 等目录布局；自定义网格不含基线时差值为空。
- 加 `--plot` 后生成 `success_rate.svg` / `.png`：横轴为 EMA β 或接受阈值，纵轴为成功率。

先加 `--dry-run` 可以检查命令，不导入仿真器、不创建文件。绘图可在已有结果上单独执行，无需重新仿真：

```bash
python -m examples.mpc.franka.experiments.mpc_ablation.sensitivity \
  output/ablation/ema_mixed/sensitivity_summary.json
```

```bash
python examples/mpc/franka/ik2/test_mpc_ablation.py --a5 --headless --sync-planner \
  --obj elephant --trial-start 0 --trial_num 10 \
  --ranking-metrics-jsonl output/ablation/a5_trials.jsonl \
  --rollout-summary-json output/ablation/a5_summary.json

SCSP_PLANNER_ONLY=1 python -m pytest -q examples/mpc/franka/experiments/mpc_ablation/test_ablation.py
```

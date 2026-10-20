# Ranking counterfactual experiment

在 IsaacGym / acados 环境中，从仓库根目录运行：

```bash
python test_rs.py --flipping --rs-output outputs/rs_five_mixed6
```

默认寻找 5 条**当前同步仿真**轨迹：`stanford_bunny2` 成功和失败各一条，
`elephant`、`piggy_bank`、`rubber_duck` 成功各一条，共 4 success、1 failure。
每个物体/结果组合从 `--trial-start` 起尝试 `--trial_num` 个 seed（默认 0–4）。
历史 CSV 可用 `--rs-source-csv ...` 指定，用来筛选 seed；是否成功仍以当前
同步 rollout 为准。若没有足够符合条件的源轨迹，程序明确报错。

每条源轨迹随机采样 6 个不同控制时刻，共 30 对。候选时刻要求
`p_arm_world` 与 `best_contact_point_world` 相距至少 3 cm，且二元决策可用。
抽样在“至少包含 1 个已接触、1 个未接触时刻”的约束下均匀进行，不按
counterfactual 的结果筛选。已接触要求实测 patch 的 `dist <= 0`；有正间隙的
PhysX speculative patch 算未接触。可用 `--rs-min-contact-distance` 调整距离。

每个反事实分支都从同一 seed 重放完整前缀，核对物理观测、执行命令和排名
决策。采样时刻只将最终二元决策取反：选 `p_arm_world` 就改选
`best_contact_point_world`，反之亦然。保存该点的物体局部表面坐标和法向，
将 verify 强制写成相反选择（`stay/p_arm=0`、`switch/best=1`），冻结 verify
及目标选择，CPO/MPC 继续闭环规划。实测接触点距该局部表面点
不超过 1 cm 且 `dist <= 0` 后恢复正常 RS；若一直没接触，则保持到 trial
结束，不设置固定冻结步数。可用 `--rs-contact-tolerance` 调整接触容差。

接触求解器和 MPC 全程使用 acados，禁止 IPOPT 回退；初始化或 MPC 求解失败
会明确报错，不能当作任务失败样本计入 advantage。

源轨迹的原始后缀作为 factual rollout，每个样本独立重放取反分支，不会在
同一次 rollout 连续取反 6 次。终止奖励为 `J = gamma**T * success`，
`gamma=0.99`，`T` 包含剩余的所有实际控制步（包括接触转换和绕行）。
`normalized_advantage = (J_RS - J_cf) / max(J_RS, J_cf)`，两者都失败时为 0。
正值支持 RS，只有 RS 成功时为 +1，只有反事实成功时为 -1。

输出目录包含各源轨迹子目录及合并的 `states.jsonl`、`states.csv`、
`trials.jsonl`、`summary.json`、`table_vi.csv` 和 `manifest.json`。
`normalized_advantage` 是论文中的指标，`advantage_rs` 保留未归一化差值。
每个反事实 outcome 的 `contact_commitments` 记录目标局部坐标、冻结起始步、
是否接触成功以及解锁步。汇总保留零 advantage，并分别汇总两类接触状态和
成功/失败源轨迹。程序不自动修改论文表格。

单独调试一个物体和结果组合：

```bash
python test_rs.py --rs-cohort single --obj stanford_bunny2 --flipping \
  --rs-source-outcome failure --trial_num 5 --rs-output outputs/rs_bunny_failure
```

`--rs-resume` 在核对配置、源轨迹和采样时刻后复用已完成的配对。
`--rs-source-trajectories` 控制 single 模式需要的源轨迹数量；旧参数
`--rs-successful-trajectories` 作为别名保留。

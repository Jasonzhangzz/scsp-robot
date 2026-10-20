# ContactSDF 在 Franka 缓存任务上的单指适配

`ik2/test_mpc_isaac.py --contactsdf` 和 `ik2/test_mpc_ablation.py --contactsdf`
均在进入本地规划循环前分发到 `examples/mpc/franka/contactsdf_run.py`。
该分支只执行“官方接触检测 → 官方 `planOnce` → 首个指尖位移”。
官方三球示例没有本仓库的 verify、CSO、approach、分阶段 cost、多初值搜索，
这里也不使用这些策略，不创建 Lambda/soft-SCM。

## 官方来源与保持的算法

来源：[asu-iris/ContactSDF](https://github.com/asu-iris/ContactSDF/tree/3a21edf8e51788f72a7d77259af4dd584d5ec43f)，
固定 commit `3a21edf8e51788f72a7d77259af4dd584d5ec43f`。
`examples/mpc/franka/scm/vendor/contactsdf/PROVENANCE.json` 记录原文件 SHA256，
`upstream/` 保留接触检测、规划器、目标和参数的原文件。

- `lse_dynamics.py` 原样保留；不添加四元数归一化或替代接触响应。
- `ball_mpc_lse.py` 仅修改导入路径，保留原 NLP、全零首次初值、上一解及乘子直接复用。
- 检测器保留原 LSE SDF、梯度、接触中点、`phi = 0.5 × SDF`、0.015 m 筛选、地面保留一个接触点及雅可比公式。
- 保留 `h=0.1 s`、horizon 4、IPOPT 最多 100 次迭代、各控制分量 ±0.01 m，以及原重力、地面网格和状态边界。
- 目标保留指尖到物体中心的距离项、抓取闭合项、控制平方项和原终端位置/四元数项及权重。

## 用户确认的接口适配

这是官方算法在 Franka 单指任务上的适配实验，不是官方三球实验的原样复跑。

| 接口 | 适配 |
| --- | --- |
| 状态/速度/控制维度 | 三指 `16/15/9` 改为单指 `10/9/3`；检测器按维度读取指尖，目标中的指尖求和改为实际指尖数。单指下仍保留抓取闭合项。 |
| 新物体几何 | Franka 实际 actor-frame mesh 的三角面中心及面法线写成官方六列点云格式，然后调用官方检测器；不替换其 SDF 公式。两份点云输入使用相同的完整面集合。 |
| 坐标 | 平移原点设为初始物体 XY 和桌面高度，使官方地面处于 z=0；物体、指尖和目标同步平移，四元数和控制轴不变。 |
| 执行器 | 指尖位移交给 Franka 普通 Cartesian impedance 跟踪，保留关节姿态项与物理力矩限制；绕过本地 via、press、slew、floor、edge 和接触/姿态 guard。 |
| theta | `foam_brick` 使用官方 foambrick checkpoint，其余本次六类物体使用官方 cube checkpoint；未针对新物体学习或调参。这会影响迁移表现。 |
| 时间预算 | 保留缓存任务 50 s 仿真预算：`--max_rollout_length 2500` 对应 `2500 × 0.02 s`，在官方 0.1 s 控制周期下最多 500 次动作。 |
| 评价 | 沿用缓存任务终点阈值：位置误差 ≤0.02 m、四元数误差 `1−(q·q_goal)² ≤0.015`。不要求本地 flip/pose 阶段或 verify 状态；评价不会改变控制器。 |

由于官方检测器用点法线半空间的 LSE 表示几何，新物体的凹面也按该公式处理；不额外添加本地 mesh 碰撞检测或修正原公式。

```bash
python examples/mpc/franka/ik2/test_mpc_isaac.py --contactsdf \
  --obj elephant --flipping --headless --sync-planner \
  --no-headless-realtime --trial-start 2 --trial_num 1 \
  --max_rollout_length 2500 --solver-backend ipopt
```

## 固定 15 条缓存任务

从 35 条缓存任务（基线成功 25/35）中，按用户确认固定选择 7 类物体的 15 条任务，
该子集基线成功 12/15（80%）。选择发生在 ContactSDF 测试之前，不根据其结果重抽样。
这是按基线结果分层的比较子集，不能解释为全部缓存任务的无偏成功率估计。

纠正后的实验目录为 `output/benchmarks/contactsdf_official_single_tip_15_20261020/`：
`manifest.json` 保存固定案例与源 CSV 哈希，`source_snapshot/` 冻结实现，
`results.csv`、`summary.json`、`report.md` 由后台脚本在案例结束后自动生成。
脚本检查每条任务的初始物体位姿、目标、控制器标记与源文件哈希；所有 15 条有效完成后才报告总成功率。
每条失败轨迹仍计入分母；进程异常单独记为 error，不冒充完整实验。

此前 `output/benchmarks/contactsdf_isaac_15_20261020/` 中混用本地策略的结果已作废，
不得作为这一复现实验的成功率。

## Rotation 六物体测试

`output/benchmarks/contactsdf_official_rotation_6_20261020/` 从完整 rotation 缓存中固定随机
抽取六类物体，每类一条（抽样种子 20261020，不按成功与否筛选）。通过
`--contactsdf-task-json` 恢复缓存的物体初始状态、关节配置和原始目标；原缓存的目标采样
与当前脚本不同，历史目标重建后与原记录的终点位置、四元数误差分别交叉核验。
凹物体的原始碰撞分解文件按 SHA256 从独立归档恢复，避免当前分解版本替换旧任务几何。
这些都属于任务/几何接口，不改变官方检测、动力学、MPC 或目标公式。
保留该 rotation 缓存的 100 s 预算，即原 5000×0.02 s，官方周期下最多 1000 次动作。
后台脚本自动生成成功率、每周期规划耗时及终点误差的 mean±sample std；不自动覆盖论文表格。

该 rotation 队列随后按用户要求扩展为 15 条，目录为
`output/benchmarks/contactsdf_official_rotation_15_20261020/`。
保留原六类物体和六条完整结果，沿用同一冻结代码与 100 s 预算，再抽取九个未用的物体/seed 组合；
各物体共 2–3 条，不按基线或 ContactSDF 成败筛选。汇总中记录复用结果的文件哈希和原始路径，
后台仅执行新增九条，最终对全部十五条统计成功率与 mean±sample std。

## 验证

`tests/test_contactsdf_standalone.py` 核对原文件哈希、三指情况下接触检测输出与目标值/导数的逐项一致性，
并检查单指闭合项、原始 warm start 的传递、无 Lambda 和本地策略参数拒绝。
原生 Isaac 单步 smoke 已完成，IPOPT 返回 `Solve_Succeeded`；单步 smoke 不用于报告任务成功率。

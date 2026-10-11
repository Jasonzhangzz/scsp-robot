# 两阶段 object cost

`test_mpc_isaac.py` 对所有物体默认启用两阶段；MPPI、其它调用共享
lambda 类的程序仍采用原目标。位置目标和最终四元数目标不变。

阶段 `flip` 使用
`pos_coef * ||p-p_target||² + ori_coef * (1 - dot(R(q)*a_target, world_up)) / 2`，
保留位置权重，朝上误差替代完整旋转误差，世界 yaw 自由；阶段 `pose` 恢复
`pos_coef * ||p-p_target||² + ori_coef * (1-(q·q_target)²)`。
MPC 保留 10 倍终端系数，接触和动作正则不变。执行层另有下述撤压和站立保护。

目标朝上轴由实际 actor 坐标系中的目标四元数定义：
`a_target = R_target.T @ [0,0,1]`。因此 target pose 为局部 X 朝上时
优化 X，Z 朝上时优化 Z，也支持任意目标方向与正负轴。使用最终
target pose，standing transform 已由目标位姿/网格配置体现，不额外
硬编码网格轴。倾角、稳定/回退判定和执行保护均使用同一目标朝上轴。
lambda 在当前观测的物体局部坐标系求解，因此它使用
`R_current.T * world_up` 作参考方向。三个权重和两根轴通过运行时参数送入求解器，切换阶段不
重建求解器。acados 缓存区分新目标及符号 cost，避免复用旧二进制。

## 目标姿态与网格坐标系

已检查的七个 source STL 均以 +Z 朝上站立。piggy_bank、mug、teapot、
rubber_duck、foam_brick 的 XML 将 +90° Y 旋转烘焙进 canonical actor
网格，因此它们的 actor 局部 +X 朝上才是站立姿态；elephant 和 bunny
的 actor 则仍为 +Z 朝上。生成 target 和初始倾斜姿态时会补偿这次
网格旋转，世界 yaw 仍按原分布抽样。未烘焙的 raw 网格直接使用 source
站立方向，不重复应用 XML 旋转。

目标高度由目标旋转下的碰撞网格最低顶点计算，使最低点落在桌面上，
替代统一的“桌面 + 3 cm”。初始状态使用同一几何计算并留 2 mm 间隙。

[2026-10-10 目标姿态检查](../scm/outputs/target_pose_audit_20261010/REPORT.md)
包含七个物体的物理落稳验证。此前闭环报告使用了修复前的目标，
不代表本次修复后的控制表现。

## 开关和稳定条件

```bash
conda activate acc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python examples/mpc/franka/ik2/test_mpc_isaac.py --obj stanford_bunny2
```

`--two-stage-object-cost` 显式启用，`--no-two-stage-object-cost` 恢复
原 MPC cost。启用两阶段要求 `ori_coef > 0`。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--flip-enter-angle-deg` | 10 | 进入 pose 阶段的最大倾角，度 |
| `--flip-linear-speed` | 0.02 | 进入 pose 阶段的最大线速度，m/s |
| `--flip-angular-speed` | 0.25 | 进入 pose 阶段的最大角速度，rad/s |
| `--flip-stable-time` | 0.3 | 连续满足以上条件的仿真时间，秒 |
| `--flip-exit-angle-deg` | 25 | pose 阶段超过此倾角则计时回退，度 |
| `--flip-fallback-time` | 0.1 | 连续倾倒后返回 flip 阶段的仿真时间，秒 |

用 Isaac 同一物理帧的时间、姿态和真实速度判定，重复观测不累计时间，
缺失或非有限观测中断稳定计时。每个 trial 重新初始化阶段。

切换时重置排序/接触选择历史、收益归一化、停滞、verify/confidence、
失败接触屏蔽、via 路径和 MPC warm start；保留几何、动力学及已学习 SCM 参数。
跨阶段的预测与实际收益不比较。停滞分数在 flip 阶段除以 `pos_coef*0.02² + ori_coef`，
在 pose 阶段除以 `pos_coef*0.02² + ori_coef*0.015`。

最终任务成功要求当前处于 pose 阶段且满足原有位置 ≤0.02 m、四元数
误差 ≤0.015；稳定翻转和最终任务成功分别记录。

## 观测与结果

日志每 25 次规划输出倾角和速度，阶段切换立即记录。使用
`--ranking-metrics-jsonl path.jsonl` 保存逐 trial 结果：
`object_cost_stage`、`object_target_up_axis`（actor 局部坐标）、`object_tilt_deg`、`object_linear_speed`、
`object_angular_speed`、`first_stable_flip_time_s`、`object_cost_switches`、
`object_cost_fallbacks`、`stable_flip_success` 及 lambda/MPC 求解失败次数。
`pressure_releases` 与 `upright_action_rejections` 分别记录撤压次数和预测倾倒动作拒绝次数。
翻转时间从该 trial 的首个规划观测开始计算，不是墙钟时间。

默认运行是异步执行并启用在线 SCM 学习。关闭在线学习的闭环对照使用
`--sync-planner --no-online-scm --headless` 和相同 trial id；同步实验不能
代替默认异步模式的验证。

[复现实验脚本](../scm/outputs/two_stage_object_cost_20261009/run_validation.py)
包含 bunny trial 0–4 的同步成对对照、trial 0–2 的默认异步验证及其它
六个物体各一次同步检查，每次最多 5000 步。结果保存在同目录。

[闭环结果与限制](../scm/outputs/two_stage_object_cost_20261009/REPORT.md)
分别统计稳定翻转与完整位姿成功。两种模式共用 via 和已完成动作修复，
两阶段模式还包含撤压和防倾倒保护，因此该对照比较完整控制策略。

数值和求解器测试位于 `tests/test_two_stage_object_cost.py`，包括
NumPy/CasADi/Torch 一致性、局部坐标变换、稳定门槛、回退、无效观测、
反馈 cost、缓存重置和 acados/IPOPT 的实际运行时切换。

## Franka via 与翻转后的执行保护

此 MPC 入口使用固定目的地的抬升—横移—下降路径。跨越时先升到
物体网格最高点上方 25 mm，再沿直线移到目的地上方，最后下降。
跨越过程中不随候选排序反复改变目的地；末端每步前进仍受已有
`--via-max-step`、`--via-max-lead` 与 MPC 步长限制。空中及跨越时直接
跟随有界 via，不用物体收益驱动空中的绕行动作。其它共享入口保持
原 via 实现。

flip 阶段倾角进入朝上范围时，末端沿实际占据表面的外侧方向撤开
20 mm，然后等待真实速度和持续时间条件。撤开目标固定在世界坐标中，
不会逐帧跟随末端向外漂移。进入 pose 后再保持 0.3 秒撤压；撤压时
OSC 使用位置保持/撤开控制，清除旧按压位移的 slew 影响，避免撤开时仍保留旧按压。
所有 MPC 模式的动作结束后，OSC 均使用位置 PD 保持终点；
`_last_mpc_action` 仅供下一条动作的 slew 限制使用，不再重放上一条
位移的 `K*u` 按压力或非零速度参考。这样异步规划延迟不会延长按压。

pose 阶段接触动作的每个分量在求解前限制为 `0.25 / (600*sqrt(3))` m，
使预测动作落在实际 OSC 的 0.25 N 力上限内；flip 阶段对应 0.8 N。
不再先按大动作求解、再缩小执行动作。近物体任意表面时也应用近接触限幅。
实测倾角超过进入阈值且以 >1°/s 上升，或倾角 >2° 且以 >15°/s 上升时，立即撤压并
等待 0.3 秒；原 25°/0.1 秒回退规则仍生效。预测轨迹超过
`min(fallback_angle, max(enter_angle, current_angle+2°))` 的动作不执行。
这些保护不改变完整位姿 cost，位置与 yaw 仍在第二阶段优化。

绕行、撤压或预测拒绝覆盖 MPC 动作时，清除 warm start，不把旧 lambda
候选力的预测收益计入下一帧 confidence/dwell。执行保护测试见
`tests/test_franka_contact_motion.py`，覆盖有限步到达、目的地变化、撤压方向、
实测倾倒提前释放、真实 OSC 力限幅与旧按压 slew 清除。

执行端还在每个 2 ms PhysX 帧根据真实角速度计算目标朝上轴的倾角变化率，
快速倾倒时直接撤开，避免等待异步规划。撤开沿实际接触外法线施加
制动力，剔除向内分量和接触切向分量；撤开无需受按压力 0.25 N 的
限制。世界 yaw 不触发倾倒保护。`physics_pressure_releases` 记录执行端
的撤压次数，在线 SCM 记录实际撤开目标；执行保护覆盖动作时跳过
该帧候选力预测的 confidence/dwell 更新。

## 接触预测与执行的一致性（2026-10-10）

此 MPC 入口的 lambda 使用 MPC 的 20 ms 时间步长、物体 mobility 和
600 N/m 末端刚度。lambda 决策为末端驱动力，先经过末端与物体的
单边接触和摩擦响应，避免把无法实现的切向力直接作用于物体。
NumPy、CasADi、Torch 使用同一定义，力上限在 acados/IPOPT 中为运行时参数。

此入口将每组四条摩擦金字塔方向合并为一个法向反力和两个切向反力，
避免四次重复计算支撑力；切向力采用平滑的 Coulomb 圆盘限制。
该选择仅对本入口的 soft-scm 启用，旧 SCM 参数文件不能载入此模型。
桌面 gap 使用旋转后碰撞网格的最低点，lambda 与 MPC 接收同一 gap。
桌面支撑仍采用一个合力接触近似，尚未建模完整支撑足迹。

物体—末端相对 Jacobian 使用朝内法线，使末端向内运动对应 gap 减小。
PhysX 法向统一指向末端，不依赖刚体排列。采样点用于选择下一接触位置；
已经接触时，当前接触预测使用同一观测帧的实测点和法向，并随同步/异步
消息一起传递。只有实测 gap ≤0 才进入接触控制；尚未接触时保持几何接近。

进入 `pose` 阶段后，若存在实测接触，实测点最近的 mesh patch 会被锁存为执行 patch；相邻采样点的投影抖动不会再通过 top-k 引导覆盖该锁。只有接触消失或实际接触移动超过 `contact_switch_radius` 才允许换 patch。日志中的 `actual_contact_idx`、`patch_locked` 可用于核对实测点与执行点是否一致。

acados 缓存标识包含上述模型配置，生成过程用文件锁串行化，防止仿真和
规划进程并发生成同一个共享库。阶段切换和限幅更新不重建求解器。

本轮闭环结果与仍存在的收敛问题见
[接触修复验证报告](../scm/outputs/post_flip_contact_20261010/REPORT.md)。

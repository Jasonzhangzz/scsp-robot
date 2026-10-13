# elephant / stanford_bunny2 flipping 平推诊断

本次定位到三组可复现机制：CSO 的切向力罚项使它偏好沿表面法向加载；elephant 的总目标与接触保持逻辑可以把位置进展当作有效进展；执行层的速度阻尼会削弱乃至反转向上增量对应的竖直力。另确认规划与仿真的摩擦参数不一致。它们同时作用，不能用一个固定向上动作规则解决全部案例。

生产控制代码未修改；诊断脚本的探针、重算和绘图均位于本目录。

## 复现范围

- 用户配置：默认参数，仅指定物体；初始种子 0（NumPy 种子 100）。
- 使用 acc / Python 3.8、Isaac Gym CPU PhysX、acados，默认异步、实时物理。为便于记录使用 headless，并把诊断限制为 400 次规划。
- 两个默认异步原始轨迹都未稳定翻起：elephant 最终倾角 86.83°，bunny 101.42°。
- 同步伴随诊断用于确保“一个规划动作对应 20 ms 物理时间”；各 400 次规划，另采集各 300 次规划的逐物理帧 OSC 力。同步结果用于定位局部机制，不替代用户默认异步运行的总体成功率。
- 仅一个种子；并行采集与只读探针会影响墙钟时序。本文不把墙钟耗时作为性能基准，也不估计总体成功率。

## 1. CSO 正在强烈惩罚翻转所需的切向加载

`test_0902.py:64` 的默认 `friction_reg_coef=1.0`，`force_reg_coef=0.01`。`planning/mlqp_point.py:1103–1107` 的实际目标是：

```
object_cost(x_plus)
+ 1.0 * (ft1² + ft2²)
+ 0.01 * (fn² + ft1² + ft2²)
```

侧面接触的法向通常近似水平，向上加载需要一定切向分量。这个惩罚对切向力的成本明显高于法向力，而 upright 收益只按 20 ms 的单步预测计算。

在实测接触位置和法向上重新求解，保持其余输入不变：

| 对象 | 默认解的切向/法向力中位数 | 仅去掉切向罚项后的中位数 |
|---|---:|---:|
| elephant | 0.0106 | 0.6413 |
| stanford_bunny2 | 0.0209 | 0.6667 |

该比值只统计力范数 > 0.05 N 的有效加载快照：默认分别有 8/14、6/11 个快照。基线重算的物体代价收益与在线记录的最大差为 1.43e-9，因此比较的是同一个实测状态与同一个 NLP，不是更换初始状态的试验。

这是已验证的目标函数偏置。CSO 接触点排序、实测 patch 的力解，以及 MPC 接触冷启动时使用的 Lambda force 都会受到它影响。MPC 之后仍可改变动作，因此 CSO 力不等同于最终执行力。

数据：`lambda_replay.json`、`lambda_replay_summary.json`、`replay_lambda.py`。

## 2. 规划的摩擦模型与 PhysX 接触对不同

`params.py:402` 把 `mu_object_` 设为 0.5；`build_lambda_optimizer` 把它作为 `arm_friction`；`ContactIsaacCartesian.detect_once` 也把同一个 0.5 用于指尖和桌面 Jacobian。

实际仿真则设置桌面 0.5、物体 0.9、指尖 1.5。按 PhysX 默认的平均材料组合，指尖—物体为 1.2，物体—桌面为 0.7。代码未设置其他组合模式；本仓库 `mjwarp_rollout.py:125–135` 的 Isaac 场景转换也明确按这两个接触对写入 1.2 / 0.7。

[PhysX 4.1 材料文档](https://nvidiagameworks.github.io/PhysX/4.1/documentation/physxapi/files/classPxMaterial.html)规定默认组合模式为 eAVERAGE。[组合定义](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/struct_px_combine_mode.html)给出 `(a+b)/2`。

结果是规划的切向传力与桌面滑移条件和执行仿真不同。离线将两个接触对分别匹配到 1.2 / 0.7，并去掉切向罚项后，有效加载解的世界力仰角中位数变为 elephant 30.27°、bunny 26.67°；默认分别为 −1.44°、−13.45°。

此对照证明这些设置改变了规划的加载方向，尚未证明整体闭环成功率改善。仅调整指尖摩擦且保留罚项的收益有限，不能简单把单个 `mu_object_` 全局改为 1.2，否则桌面接触又会使用错误系数。

## 3. elephant 的位置收益会延长无效 patch 的保持

`ObjectCostStages` 在 flip 阶段使用 `[position, quaternion, upright] = [500, 0, 20]`。elephant 默认 `flip_position_radius=0`，保留完整 XYZ 位置代价；bunny 默认有 8 cm XY 区域，并在这个区域目标中不计 COM 高度代价。

`test_mppi_isaac.py:1563` 反馈给 dwell/confidence 的 `progress_cost` 是总物体代价，而不是单独的 upright 进展。`test_0902.py` 也会在实测 patch 仍有信任度时保留它。

elephant 同步轨迹中有 42 个接触周期同时满足：实际总目标下降、倾斜没有改善、confidence > 0.5、`accept_p_arm=True`。例如 step 72：

```
位置代价改善       +0.34093
upright 代价改善   −0.01419
总改善            +0.32674
confidence = 1.0，dwell = 0
```

这会把滑向位置目标的动作视作有益，继续留在该 patch。它是 elephant 的直接证据。

对 bunny 不能套用同样解释：在 12 个异步接触快照的有限方向搜索中，去掉位置权重没有改变最优候选；它已有 8 cm 区域。这个搜索仅比较有限动作族，不能证明位置项完全无影响。

数据：`objective_replay.json`、`replay_objective.py`、`*_sync_probe.jsonl`。

## 4. 向上的增量会被实际速度阻尼削弱

接触时，`test_mpc_isaac.py:1116` 调用的力律为 `F = 600*u - 40*v`，没有自由空间的速度参考。u 是位移增量，F 才是 OSC 施加的任务力；F 又不等同于物体最终收到的接触力。

逐物理帧记录中，elephant 的一个实际接触帧为：

```
u_z = 0.0035566 m
v_z = 0.121871 m/s
F_z = 600*0.0035566 - 40*0.121871 = −2.7409 N
```

探针记录的 F_z 正好是 −2.740875 N，且该帧 MPC action 仍处于 active 状态。这不是仅取动作到期后的力值造成的假象。

300 周期同步伴随诊断中：

| 物体 | 指尖/物体接触物理帧 | 实际 OSC 力接近水平（仰角 ±15°） | active 向上命令却产生低仰角/向下力 |
|---|---:|---:|---:|
| elephant | 1300 | 460 | 47/314 |
| stanford_bunny2 | 1544 | 1025 | 151/348 |

bunny 部分本应为负的 F_z 还被桌面高度保护截到 0，剩下横向分量。阻尼与保护规则用于控制与稳定，不能根据这项证据直接关闭。需要让规划的末端响应、实际阻尼及接触过程一致。

数据：`frame_force_analysis.json`、`*_sync_forces_probe.jsonl`、`probe.py`。图：`command_force_motion.png` / `.pdf`。图中无接触区间留空，不连接成连续力曲线。

## 5. 模型预测收益未完全落实到真实倾角

当前默认 CSO 是零初始速度、Q 先验阻抗的单步预测。MPC 接触加载时也会在物理质量响应和 Q mobility 响应之间切换；默认 horizon 为 5×20 ms=100 ms。它们不能自动保证预测出的短期 upright 收益在 PhysX 中成立。

在同一 20 ms 的同步诊断中，bunny step 76 的倾角由 119.019° 实际变为 119.571°，MPC 却预测变为 118.992°。elephant step 73 的实际倾角由 85.362° 变为 85.367°，预测为 85.019°。误差方向会影响排序和接触保持。

默认异步原始轨迹的观测间隔均值分别为 1.2975 / 1.2425 个物理策略周期；有些决策基于更早的接触与速度。同步仍复现低效滑动，因此延迟只是一项可能的放大因素，不能作为唯一原因。

## 修改优先级

1. 分开并对齐指尖—物体、物体—桌面的摩擦模型。
2. 重标定 CSO 的切向力罚项，避免默认解近乎只有法向加载。
3. 在 flip 阶段检查实际 upright 进展，避免 elephant 的位置改善一直支撑同一个无效 patch。
4. 对齐 MPC 末端响应与 OSC 力执行，再用多个种子进行完整闭环验证。

上述属于下一步修正方向。本轮完成原因检查，未把离线对照配置写入生产默认值，也未运行修改后的总体成功率实验。

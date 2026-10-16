# 接触动力学、代价、控制边界与 CA 姿态修复（2026-10-16）

## 实现

- **接触动量**：Wuji 的有限力 grasp response 只替换手接触产生的速度增量，
  从 SCM 动量平衡恢复其他接触反力，保留地面支撑，不重复加入测得的外力。
  `retain_other_contacts` 只由 Wuji 显式开启，其他调用者的默认路径不变。
- **地面支撑面**：使用最低面的最多四个真实网格点，保留支撑力矩；离桌后
  单边间隙自然关闭反力，不用接触力阈值决定何时允许离桌。
- **伺服动力学**：重力/偏置与 PD 响应一起隐式积分，再检查实际电机力限，
  避免在隐式更新后再用裸惯量重复施加重力。
- **代价**：物体位置系数 1、旋转系数 0.09、原始 MANO 点系数 0.01，
  参考姿态 wrist translation/rotation/fingers 系数分别为 1/0.09/9e-6，
  按 FA 的有效比例归一化。移除额外 transport、首次接触锁定、法向平衡目标；
  虚拟闭合系数单独设为 0.4，动作使用弱软正则。物理碰撞检查继续生效。
  虚拟闭合与 FA 的实际距离惩罚仍是不同物理量，没有声称完整 cost 等价。
- **精确评分**：原生 horizon 评分对测得手姿和虚拟伺服目标都重新计算 FK，
  不把大幅手指运动继续代入起点的材料点线性近似。
- **控制边界**：删除接触阶段 0.04 m/s、0.2 rad/s、0.5 rad/s 上限，删除
  示教速度派生的增量限幅、闭合偏置的 0.6 mm 步进上限和 transport 种子的
  0.05 m/s 上限。所有关节位置范围、控制位置范围、电机力限仍是物理边界。
  原始参考按示教时间访问每一帧，物体停滞不再冻结后续手姿。
- **IK**：默认 `--ik-backend ca`，在隔离子进程运行 DexForge 原始 CA 全流程，
  包含关键帧、整轨迹和 wrist mount。通过物体坐标中的 body FK 转换为执行器
  腕部坐标，不复制不同坐标系中的 Euler 角。CA 只提供姿态 prior；其 mask、
  优化锚点和可行性证书不替换 MPC 的原始 MANO best-contact targets。

`--ik-backend geometric` 保留此前 IK 作为显式对照。`--ca-result DIR` 和
`--ca-scene MOUNTED_MESH_XML` 可复用完整 CA 求解，检查原始输入数组与有效帧
掩码一致。`--sync-assets` 同步原始 CA、Wuji 资产、tools 和对应 ComFree 依赖。
CA 只修改输出目录中的私有场景副本。

## 验证

同一条 `20200709-subject-01/20200709_142802` SugarBox、全部 72 帧、Wuji 右手、
质量 0.2 kg、μ=1、原始指关节刚度与力限、2 ms MuJoCo、Lambda off。

本次实际重新运行 CA，数值过程未替换；只关闭视频和关键帧渲染。
CA FK 转换的最大 body 位置误差为 3.38e-16 m，旋转误差为 1.59e-15 rad。
最新输入检查和关节映射代码重新准备后，`wuji_q.npy` 与闭环使用的数组逐项相等。

此前第 239 周期冻结状态，同一 hold 命令，0.12 s 混合预测的中心 z 误差：

| | 修复前 | 修复后 |
| --- | ---: | ---: |
| 对原生 MuJoCo 的终端 z 误差 | 36.869 mm | 3.271 mm |

预测仍为固定几何的简化模型，不能把误差改善称作与原生动力学完全等价。

新闭环运行 180 个控制周期、5.4 s，并对保存的位置命令进行独立原生回放：

| 指标 | 修复前 2 ms 对照 | 修复后 |
| --- | ---: | ---: |
| 已访问示教帧 | 0–31 | 全部 0–71 |
| 最大最低点离桌高度 | 0.011 mm | 192.885 mm |
| 末端最低点离桌高度 | 未离桌 | 144.560 mm |
| 最长连续离桌、有受力手接触、无地面支撑 | 0 s | 4.02 s |
| 末端手向上合力 | 约 0.66 N | 1.9648 N |
| 末端地面向上力 | 约 1.29 N | 0 N |
| 末端重力归一化 wrench 残差 | 约 1.42 N | 0.00597 N |
| 原生回放对象位置最大误差 | — | 0 |
| MPC 求解失败 | — | 0 |

持握时长按 30 ms 控制采样统计：最低点高于 2 mm、地面竖直力小于 0.1 N，
且有实际受力手接触。此定义只用于离线统计，不进入控制器。

末端对象位置误差 **24.21 mm**、方向误差 **16.98°**。因此本次已经能抬起并
持续夹住物体，但没有达到程序的 10 mm / 5° 严格终点判据，
`object_task_completed` 仍为 false，不能称为完整轨迹精度已与 FA 相当。
只测试了这一条序列，未声称多序列或 Sharpa 泛化结果。

全部 180 周期验证：无 Lambda 调用；best-contact 点和 mask 与原始 MANO
逐项相等；与修复前语义点/法向/mask 完全一致；命令速度限幅全部关闭。
没有加入接触力、力闭合或承载证书门控。

相关 Wuji 测试 78 项通过，包含地面反力不被部分夹持覆盖、支撑面力矩与离桌
释放、不同 wrist mount 下 CA FK 一致性，以及无限速边界和新代价比例。
共享 grasp response 的 4 项回归也通过。旧 MANO pickle/绘图依赖有弃用警告。

## 产物与复现

- `outputs/wuji_contact_fix_20261016/`：新 CA 结果、私有 mounted 场景和 FK audit。
- `outputs/wuji_contact_fix_rollout_20261016/`：完整闭环命令、诊断与 metrics。
- `outputs/wuji_contact_fix_validation_20261016/`：修复后冻结状态预测对照。
- `outputs/contact_comparison_fixed_20261016/`：独立回放、接触力/GWS、对比曲线。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/zz/miniconda3/envs/pi0/bin/python examples/retarget/retarget.py \
  --sequence outputs/retarget_shared_20261016 \
  --physics-reference outputs/retarget_shared_20261016/dexforge_sdf_physics.xml \
  --ca-result outputs/wuji_contact_fix_20261016/contactaware \
  --ca-scene outputs/wuji_contact_fix_20261016/ca_sequence/scene/wuji_right/sugar_box_mesh.xml \
  --headless --no-demo-overlay --max-steps 180 \
  --output outputs/wuji_contact_fix_reproduction
```

省略两个 `--ca-*` 参数即重新执行 CA；默认 YAML 使用相同 CA/代价设置。

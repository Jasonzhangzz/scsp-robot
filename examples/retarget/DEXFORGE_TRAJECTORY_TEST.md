# 同一条 DexYCB 轨迹的接触实测（2026-10-16）

本报告记录修复前对照；后续运行已抬起并持续夹持物体，具体精度与限制见
[接触动力学与 CA 修复](CONTACT_DYNAMICS_CA_FIX.md)。

冻结状态的动力学对照、候选分项代价、控制速度和 IK 对照见
[失败原因分析](DEXFORGE_FAILURE_ANALYSIS.md)。

本次重新求解 `20200709-subject-01/20200709_142802`，两边均为 Wuji 右手。
DexForge 在 `8.147.67.185:1016` 的 `pi0` 环境中重新运行原始 CA、FA 入口，
通过已有的数据/手资产适配脚本启动；没有用旧结果替代新求解。
DexForge HEAD 为 `fa6d671948ef03d18514062748b94e445b63cd49`，原有底层
模拟器本地修改保留。新结果根目录为：

```
/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/grasp_sonic/results/
  scsp_contact_comparison_20261016/dexforge/20200709-subject-01/20200709_142802
```

## 输入与物理参数

两边使用同一份 72 帧 MANO 数组；输入 hash 相同，物体网格的顶点和面也完全相同。
未沿用之前只保留 69 个有效帧的缓存。物体坐标系中的点可以直接比较。

| 参数 | 本次对齐值 |
| --- | --- |
| 物体质量 | 0.2 kg |
| 物体主惯量 | (0.001, 0.001, 0.001) kg·m² |
| 物体质心 | (0, 0, 0) |
| 滑动、扭转、滚动摩擦参数 | (1.0, 0.005, 0.0001) |
| 重力 | (0, 0, −9.81) m/s²，物重 1.962 N |
| 手指 kp、阻尼、执行器力限 | 原始 Wuji 参数；两边编译模型一致 |
| 指刚度/指力限倍率 | 1 / 1 |
| 控制重规划周期 | 30 ms |
| FA 执行步长 | ComFree 10 ms |
| 原生 MuJoCo 对照 | 分别测试 10 ms、2 ms |

当前方法用 `--physics-reference` 从这次 FA 的 SDF 场景直接复制物体惯量、
接触参数和地面，没有手填一个近似物体。原生 MuJoCo 的凸包碰撞表示与
ComFree 的手侧 sphere cloud / 物侧 SDF 表示仍不同，不能称作相同求解器。
两边已有的地面稳定化造成初始物体位姿约 0.079 mm、0.153° 的差异，
故本报告使用物体局部坐标比较接触，而不假定世界坐标轨迹逐帧重合。

## 先检查同一组 DexForge 控制命令

先固定 FA 保存的位置目标，不运行当前 MPC。这样能区分控制算法差异与原生
执行环境差异。以下 MuJoCo 参数扫描在每个物理子步检查离桌和实际手接触：

| MuJoCo 步长 | 与当前场景一致的结构碰撞排除 | 最大最低点离桌 | 末端离桌 | 最长离桌且仍有手接触 |
| --- | --- | ---: | ---: | ---: |
| 2 ms | 否，官方 mesh 回放设置 | 0.147 mm | −0.144 mm | 0 s |
| 10 ms | 否 | 1.572 mm | −0.216 mm | 0 s |
| 2 ms | 是 | 143.738 mm | 132.734 mm | 1.416 s |
| 10 ms | 是 | 101.313 mm | −0.096 mm | 0.530 s |

这里“离桌且仍有手接触”指最低点高于 2 mm、地面接触力小于 0.1 N、存在
法向力大于 0.001 N 的手接触，仅用于离线统计，不进入控制器。

同一组 FA 控制在原生 ComFree 中最大离桌约 175.5 mm，末端约 168.5 mm，
在轨迹后段持续握持。这说明原始 mesh 回放失败不能直接归因于接触规划；
结构重叠碰撞和仿真步长会改变这一结论。排除的是关节安装结构的重叠，
没有改手/物体接触点或施加外力。

## 接触点与力闭合如何比较

`export_dexforge_contacts.py` 使用原生 ComFree 回放控制，导出碰撞点、法向、
切向和每条约束行产生的实际力。回放关节/物体位置与保存状态的最大误差为 0；
接触力合成与原生广义约束力的最大差约 5.44e−7 N。

`compare_contact_trajectories.py` 将三个轨迹的接触点转换到物体坐标，按原始
示教帧 ID 对齐，同时单独绘制执行时间。没有把停滞在第 31 帧的算法外推到
后面的示教帧。

所有轨迹重新计算同一种 GWS 余量：四边摩擦锥、μ=1.0、单位力射线、质心
为 wrench 原点、共同物体半径 0.10406158 m。正值表示原点位于六维 wrench
凸包内部。重复同一个接触点不会增大该指标。该余量没有包含电机能力和实际
力大小，不把正余量当成实际稳定握持的充分条件。

另外固定 FA 的**完全相同关节和物体姿态**，在 MuJoCo 中只查询碰撞几何：
同 body 的最近接触点平均相差约 1.62 mm，法向约 2.65°。例如示教约第 31 帧，
ComFree 有 5 个接触输出，MuJoCo 有 26 个接触点。MuJoCo 包含重复碰撞体和
接触流形的多点；因此不能直接把接触点数量或两个程序各自的 closure 值相除。

## 当前方法的变化及已完成对照

当前默认 `lambda_mode: "off"`，不实例化或求解 Lambda，不先判断 MANO 点的
接触 IK 可行性，也不据此删点。MANO 点作为持续软目标，MPC 在线评估附近
投影控制。实际接触的保持锚点与 MANO best-contact 目标分别记录；重新接近
的目标仍为示范点。

10 ms 对照已完成 600 个周期，即 18 s：全部周期没有 Lambda 调用，全部
示范 best-contact 点和掩码均与输入逐项一致。算法停在示教第 31 帧，最大
离桌约 0.014 mm，未完成夹持搬运。力闭合常常为正，仍不能代表实际承重。

同一阶段的首个约第 31 帧样本，当前 10 ms MPC 的 GWS 余量约 0.00423，
向上合力约 0.849 N；ComFree FA 约 0.0680、2.179 N。物重为 1.962 N。
当前 thumb 的物体局部接触 z 约 +24.9 mm，index 约 +2.8 mm；FA 分别约
+7.9 mm、+25.6 mm。两边都能在偏上区域接触，但实际相对分布和载荷不同。
这些是实际轨迹测量，不是把 MANO 区域改掉的建议。

2 ms 补测也已完成：使用全部 72 帧输入，运行 240 个控制周期（7.2 s），
仍停在第 31 帧，最大最低点离桌仅约 0.011 mm，未完成夹持搬运。全部周期
均未调用 Lambda，全部 best-contact 点及掩码与该帧 MANO 输入完全一致。

下面按示教第 31 帧附近分组，列出实际接触的中位数。当前方法在这一帧停留
141 个样本，两个 FA 轨迹各有 3 个样本，因此它表示各自在该阶段的实际状态，
不是相等执行时间或相等采样数量的平均成功率。

| 轨迹 | GWS 余量，μ=1 | 手向上合力 | 法向力总量 | 同 body 最近 MANO 点误差 |
| --- | ---: | ---: | ---: | ---: |
| 当前 MPC，MuJoCo 2 ms | 0.08763 | 0.982 N | 4.274 N | 19.21 mm |
| FA 控制，MuJoCo 2 ms | 0.07172 | 2.293 N | 6.482 N | 21.64 mm |
| FA 原生 ComFree | 0.06801 | 2.179 N | 5.442 N | 20.47 mm |

当前方法此阶段的几何余量并不低，位置误差也没有比 FA 更大，但手的向上合力
只有物重的大约一半；末周期仍有约 1.292 N 的地面向上支撑。它已经形成几何
上可闭合的点集合，却没有通过实际伺服建立并维持正确的承载 wrench。
这组数据不能支持“把原始接触点判不可行后丢弃即可修好”，也不能支持“正的
力闭合数值就说明已稳定夹持”。后续控制修复应以这些实际力和轨迹为依据。

默认配置已更新为 `lambda_mode: "off"`、`physics_profile: dexforge`、原始
指关节刚度/力限倍率 1、原生步长 2 ms；并已用默认 YAML 实际准备一帧，
确认生成场景为 0.2 kg、μ=1.0，无需显式传入临时物理 XML 才能生效。

数值与图表：

- `outputs/contact_comparison_dt01_20261016/`：完整 10 ms MPC 对照。
- `outputs/contact_comparison_20261016/`：2 ms 原生执行环境下的补充对照。
- 各目录 `samples.csv`：逐控制样本的 GWS、合力、地面力、离桌与区域误差。
- `contact_points.csv`、`aligned_contact_snapshots.csv`：实际点坐标、法向及力。
- `target_points.csv`：MANO 和 CA 目标点；`same_state_geometry.csv`：相同姿态
  下的碰撞查询差异。
- `trajectory_metrics.png`、`contact_positions.png`：轨迹曲线与物体局部接触位置。

## 复现当前方法

数据和参考物理 XML 来自本次新求解，已缓存于 `outputs/retarget_shared_20261016/`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/zz/miniconda3/envs/pi0/bin/python examples/retarget/retarget.py \
  --sequence outputs/retarget_shared_20261016 \
  --physics-reference outputs/retarget_shared_20261016/dexforge_sdf_physics.xml \
  --lambda-mode off --finger-stiffness-scale 1 --finger-force-scale 1 \
  --sim-dt .002 --headless --no-demo-overlay --max-steps 240 \
  --output outputs/wuji_dexforge_physics_dt002_20261016
```

三个 Wuji 测试文件共 75 项通过，另复查了 YAML `"off"` 不被解析为布尔值、
参数复制和 GWS 对重复接触的不变性。这些测试通过不代表物理夹持已成功。

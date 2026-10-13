> 已被 receiver_rotation_fix.md 的复核结论取代。此前人为弯曲初始化方案已撤回，捏合距离变小不能证明动作正确。以下内容仅保留诊断历史。

# cqq_v2 手套离线 retarget 诊断（2026-10-13）

检查服务器：8.147.67.185:1016。远端仓库：
`/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/ego_fast_retarget`。

输入 collection：
`/mnt/jfs/x2robot-prod/x2robot-data/collection/18/RP001_VR/XRRD11ZD2604070080/20261008170930-stack_cup_random_CQQ_rpAAAA`。

用户确认录制数据与 cqq_v2 标定使用同一副手套，实时测试在另一台机器上进行。
本次没有改动远端生产代码、YAML、录制数据或正在运行的批处理。

## 已复现：SDK 骨架 IK 对初始化状态敏感

`utils/wuji_retarget/emf_offline.py:170` 的 `solve_native()` 每条 episode、每只手新建
`WujiGlove.offline_pipeline()`（约第 190 行），直接从该 episode 首帧执行
`pipeline.hand_skeleton().compute(frame)`（约第 215 行）。之前的在线 IK 状态不在输入中。
随后将骨架位置交给新的 `RetargetSession.step()`。

在 episode `...@2026_10_08_17_19_04` 的全部 807 个原生 EMF 帧上，保持原始 EMF
和 cqq_v2 URDF 不变，只改变进入录制帧前的骨架 IK 状态，右手拇指—食指距离如下：

| 初始化方式 | 最小 mm | 中位 mm | 最大 mm |
|---|---:|---:|---:|
| 原样冷启动 | 100.78 | 114.66 | 141.95 |
| 先重复首帧 100 次 | 100.78 | 114.71 | 141.95 |
| 从该 URDF 的零位 FK 生成中立 EMF，初始化后再处理原始帧 | 26.89 | 34.29 | 62.18 |
| 非拇指 flex/PIP/DIP 取 0.3 rad 的 FK 初值 | 18.24 | 33.62 | 43.16 |

中立初始化 1、5、30 帧均得到相同的距离范围。合成初值只用于诊断，不进入输出。
使用真实录制帧的后半段预热或反向预热，也能改变同一输入段的输出。
这证明错误首先发生于 EMF → 人手骨架 IK，且存在不同收敛分支。
持续在线运行的 IK 有前序状态，而离线每条轨迹重新创建 pipeline；这是解释在线、离线
表现不同的一个有实验支持的原因。尚未获取另一台机器的 SDK 精确版本、代码及在线初值，
因此未完成在线与离线的逐帧等价性证明。

这些距离是人手骨架的数值，不是机器人实际接触距离。初值实验也不是完整修复：
捏合距离改善不能单独证明每个关节、每个手指姿态正确。后续应做多初值、原始 EMF 拟合
残差与连续性联合选择，并对照视频/在线结果验证。

## 已确认：新标定手指角度与旧 world 骨架被混用

`utils/wuji2_retarget/official_retarget.py:681` 从录制 JSON 加载 world 骨架。
约第 697 行调用 `emf_commands_for_frames()` 按 cqq_v2 重算 SDK 关节角。
约第 805–818 行仍用 `sequence.positions/rotations` 构造目标；约第 960 行 MCP
手腕求解仍使用这份录制骨架。`maybe_resize_glove_skeleton()` 当前直接返回输入，不重建它。

录制的 `wuji_glove_offline_manifest.json` 明确记录左右模型均为 `builtin_default`。
用内置模型重算的骨架与 H5 原有骨架最大位置差约 0.000015 mm，证实录制骨架来自
内置模型，而不是 cqq_v2。手指 q 来自 cqq_v2，手腕/MCP 对齐参考仍来自内置模型。
此混用影响模型一致性与整体手姿对齐；它不能单独解释局部捏合距离异常，
因为刚体手腕变换不会改变同一只手的拇指—食指距离。

修复时应输出并复用本次 EMF 解算得到的骨架位置与旋转，通过同帧录制的
VR/EMF TX world 变换及当前 URDF 的 wrist→TX 外参统一转换到 world，再用于
MCP 对齐、目标构造与 source 骨架发布。

## 排除项与取样结果

- YAML 和实际 batch 日志都指定 cqq_v2 的左右 URDF；离线代码检查 SDK 返回的
  `urdf_source=override` 及加载路径。并非 YAML 未传入或静默使用默认模型。
- 左 URDF SHA256：`96cfe0dd3770e61667dda41ce9a55b2cb408789a9acd283073544c196b7712ed`。
- 右 URDF SHA256：`180a3d54efe171142e323c68e5769d67ed945df8e0ddc9158a3e8ddbaf6d9056`。
- 录制与当前生产环境使用 wuji-sdk 2026.8.31；将已有 2026.9.22 wheel 解压到
  `/tmp` 隔离运行，骨架结果逐点完全相同，没有升级生产环境。
- 四条取样轨迹左右 SN 一致：左 `WG1JA06260620040`，右 `WG1KA06260701514`。
- 第一条已发布 JSON 去掉起始 ramp 后的首个右手 q 与独立重算的冷启动 SDK q 一致。
  异常已存在于发布之前，不能归因于发布时左右手调换或手指插值。
- 生产路径保留 SDK q，跳过 Laplacian 手指优化；`right_pinch_distance_weight` 等参数
  在该路径中被设为零。因此调整这些配置无法修复上游错误骨架。

| Episode 时刻 | cqq_v2 右手距离 min/median/max mm | 内置模型 min/median/max mm |
|---|---|---|
| 10-08 17:19:04 | 100.78 / 114.66 / 141.95 | 27.97 / 43.94 / 65.91 |
| 10-08 17:21:28 | 63.97 / 87.86 / 122.02 | 18.54 / 42.37 / 71.47 |
| 10-08 18:14:52 | 55.68 / 88.33 / 103.99 | 29.63 / 48.38 / 67.00 |
| 10-09 12:05:21 | 12.71 / 29.35 / 58.14 | 7.23 / 34.08 / 61.93 |

复现脚本 `check_ik_initialization.py` 可在远端 pi0 Python 环境执行，参数为
`--episode <第一条 episode 目录> --models <仓库>/hand_models/cqq_v2`。
脚本只读取输入、打印对比，不连接硬件，不写录制数据。

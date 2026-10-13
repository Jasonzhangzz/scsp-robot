# 右手 follow 差异的 SDK、URDF 和处理链路定位

## 已验证的结论

右手 follow_right_position 和 follow_right_ee_rotation_6D 的主要差异来自当前工作区新增的 **腕部世界位姿外参重建 + 使用新 SDK 世界骨架替换采集骨架**。关键改动尚未提交到 main。

右手 glove URDF 文件内容、实际 SDK 版本、右手 native SDK 骨架与手指关节角均相同。独立对照实验中，当前代码保持旧版采集骨架后，右手 follow 两项 195 帧全部逐元素完全一致，差值为 0。

本报告讨论的是该 episode、ex001_wuji / Hand1 / retarget_weight=normal；不把其他手型或 tips 权重分支的行为泛化到本次结果。

## SDK 实现与实际调用

两版均使用安装于 pi0 的 wuji-sdk==2026.8.31，实际调用链为：

```python
pipeline = WujiGlove.offline_pipeline(sn=recorded_sn, hand_side='right', urdf_path=right_glove_urdf)
skeleton = pipeline.hand_skeleton().compute(native_emf_frame)
session = RetargetSession.for_hand(HandModel.WujiHand, side=Handedness.Right)
q = session.step(skeleton_positions)
```

每个 native EMF 帧计算一次骨架和一次 session.step，之后关联原始相机帧。本次共有 785 个右手 native 帧，保留全部 195 个原始相机帧。当前 RX orientation 修正规则在此记录上 corrected_frames=0。

仓库 thirdparty/wuji-retargeting 中的 Retargeter / AdaptiveOptimizerAnalytical 最近有改动，但本次 normal SDK preserve 路径没有调用该优化器。当前官方 Retargeter 入口在 load_official_retargeter 中定义，额外优化由 retarget_weight=tips 分支触发；本次为 normal。因此该第三方优化器的权重或实现差异没有进入本次手指求解链。

SDK 模块路径：`/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/miniconda3/envs/pi0/lib/python3.11/site-packages/wuji_sdk/__init__.py`。历史原始配置显式要求同一 SDK 版本，运行时版本检查通过。

## 实际右手 glove URDF

- 历史原始配置路径：`/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/ego_fast_retarget/tests/retarget_1_comparison_20261013/historical_adde321/models/wuji_v3_20260920/right_hand.urdf`。
- 当前配置路径：`/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/ego_fast_retarget/hand_models/cqq/right_hand.urdf`。
- 两文件 SHA-256 均为 `c09676d29d209420111aa218962cc31b8b033bf7e1581ca4054411af9171b800`，内容逐字节相同。
- 重新使用未经适配的历史原始 model_config 运行历史 solve_native 后，runtime 确认 urdf_source=override，并实际加载上述历史路径；其右手局部骨架位置、旋转、手指 q 与当前 785 帧数组全部逐元素相同。
- 实际 episode 手套 SN 为 `WG1KA06260618504`。历史 model_config 中 source_right_glove_sn 是模型来源记录，不是这次 offline_pipeline 的设备 SN。两版都从 H5 读取真实设备 SN。

历史默认左手为 builtin_default；当前 YAML 左右手都指定 CQQ。为隔离代码差异，前面的完整轨迹比较和下列消融统一了左右手模型、initial_position=[0.37,0,0.245]、mcp anchor 和 20 Hz。另一次历史原始模型配置测试仅用于验证右手 SDK 与 URDF，没有把左手模型差异混进右手 native 结论。

## 具体发生变化的地方

历史 official_retarget 沿用采集 JSON 中的 hand_skeleton_world，SDK 重新求出的骨架主要用于手指 q。

当前新增链路：

1. `source_current/emf_offline.py:213`：从 glove URDF 的 emf_tx_base FK 取逆矩阵，得到 TX→校准 wrist。
2. `source_current/glove_ik.py:113`：构造新的世界腕位姿：

```text
T_world_calibrated_wrist = T_world_captured_wrist
                         @ T_captured_wrist_TX
                         @ inverse(T_calibrated_wrist_TX)
```

3. `source_current/emf_offline.py:265`：用新的世界腕位姿，把 SDK 的局部骨架转换为世界骨架。
4. `source_current/official_retarget.py:733`：调用 sequence_from_sdk；`source_current/sdk_retarget.py:137` 替换原采集骨架的位置、旋转和指尖。
5. 替换后的 wrist orientation 和 MCP 位置进入 MCP palm placement、world_to_base 对齐及机械臂目标生成，最终改变 follow 字段。

输出端 _controller_eef_from_base_palm 的源码在两版相同。因此 follow 字段的差异在输出序列化之前已经形成。

## 外参差异的数值证据

采集 TF 的右 wrist→TX：平移 `[0, -0.03, -0.045]` m，旋转为单位旋转。

相同的右手 glove URDF 的 wrist→TX：平移 `[0.0060980059, -0.0505083351, -0.0433681468]` m，旋转非单位矩阵。

两者组合出的 captured wrist→calibrated wrist：

- 平移 `[0.005126378000371866, 0.030840269582376144, -0.017773990792568684]` m，长度 **35.962713 mm**。
- 旋转角 **25.232302°**。

实际 195 帧的 source 右腕世界位置差都为 35.962713 mm，旋转角差都为 25.232302°，与上述矩阵推导一致。SDK 返回的局部 wrist 根关节始终位置 [0,0,0]、旋转单位四元数；该变化发生在新增的外部世界坐标重建中。

随后双手 world_to_base 对齐和 MCP 目标放置会改变最终 controller follow 的具体差值，所以最终平均 follow 误差为 35.81 mm、23.21°，并不要求与 source 腕位姿的静态差值相等。

## 对照实验：与历史同参数结果比较

统一同一原始 195 帧，去掉各自起始过渡段。

| 当前代码实验 | 右 follow 位置距离，平均 / 最大 mm | 右 follow 旋转角差，平均 / 最大 ° |
|---|---:|---:|
| 当前完整逻辑 | 35.809139621 / 44.158559930 | 23.212501175 / 23.681586868 |
| 保留 SDK 骨架，但使用采集腕位姿，不进行外参重建 | 4.162778162 / 8.137704453 | 0.008143099 / 0.008149393 |
| 保留旧版采集骨架，跳过 sequence_from_sdk 替换 | 0.000000000 / 0.000000000 | 0.000000000 / 0.000000000 |

第二项实验仅跳过 recorded_world_from_wrist 中的校准外参重锚定：源右腕位置回到历史值，源右腕旋转差只剩浮点舍入量；follow 剩余 4.16 mm 平均位置差来自使用新 SDK 骨架后 MCP 点及双手对齐的变化。

第三项实验仅把 sequence_from_sdk 替换为保留输入 sequence：右 follow 两项、world_to_base、左右腕 base 位姿全部逐元素一致。可确认新增的世界骨架替换链路足以解释本次 follow 差异。该实验的 arm_q 仍有最大 0.001197983 rad（约 0.06864°）残余，因此不声称整条机械臂关节数组完全一致；本次定位的 follow 字段确实精确归零。

所有诊断替换仅存在于独立 run_current_capture_wrist.py / run_current_legacy_bones.py 包装脚本中，没有编辑生产求解器。

## 文件

- sdk_urdf_audit.json：实际包版本、模型 hash、runtime 加载路径、TF/URDF 矩阵。
- historical_original_model_metadata.json / historical_original_model_comparison.json：历史原始模型配置的独立右手 SDK 复现。
- ablation_metrics.json：逐元素比较及消融统计。
- source_current/、source_historical/：本次源码快照。
- world_skeleton_uncommitted.patch：关键代码相对 main 的未提交修改。
- compare_ablations.py、run_current_capture_wrist.py、run_current_legacy_bones.py：复现实验脚本。

定位到行为差异不等于判定某一腕坐标定义物理上正确。若目标是复现历史 follow，应保留采集世界骨架/腕部目标；若目标是让新 glove 标定也改变全局腕位姿，则当前逻辑会产生这里测得的变化，需按期望的坐标定义决定是否保留。

# 原始 episode 的两版求解与右手 follow 字段对比

## 结论

当前代码和 main~2 均在 pi0 环境成功求解，原始 195 帧的机械臂 IK 均为 195/195。

**右手 follow 字段不完全一致，而且同一原始帧对齐后，195 帧全部不同。右手机械手 20 个手指关节角逐元素完全一致。**

| 对比项 | 平均差异 | 最大差异 |
|---|---:|---:|
| follow_right_position 的三维位置距离 | 35.80914 mm | 44.15856 mm |
| follow_right_ee_rotation_6D 的旋转角差 | 23.21250° | 23.68159° |
| right_hand_q / wuji_right_joint_position（动作段） | 0 rad | 0 rad |

位置距离范围为 30.60888–44.15856 mm；旋转角差范围为 22.89703–23.68159°。

follow_right_position 和 follow_right_ee_rotation_6D 是 right_arm_neutral 坐标系中的 right_arm_link6 目标位姿。它们不是手指关节角。6D 旋转按 [R00,R10,R20,R01,R11,R21] 恢复旋转矩阵，差异用相对旋转的 SO(3) 角度计算，避免直接把六个分量误当成角度。

## 输入、版本与参数

- 原始 episode：`/mnt/jfs/x2robot-prod/x2robot-data/collection/18/RP001_VR/XRRD11ZD2604070080/20260918180420-pick_doll_CQQ_rpAAAA/20260918180420-pick_doll_CQQ_rpAAAA@MASTER_SLAVE_MODE@2026_09_18_18_09_17`。
- 原始 JSON SHA-256：`22855f6088511fde395996732b6014998f7cf7c4a342446dbae50bcb30546730`。
- 历史版本：`adde3216860a76b70f3bfda462ab48e50ab9cba4`（main~2 / adde321）。历史源码从 Git 导出到独立目录。
- 当前版本：HEAD `0da78fb17418fb77fe408eb04391ba2452abcea4`，使用远端现有工作目录，包括已有未提交修改；修改快照见 current_working_tree.patch。
- Python：`/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/miniconda3/envs/pi0/bin/python`。
- 两版均为 ex001_wuji / WujiHand1，使用相同的 CQQ 左右手 URDF、mcp anchor、initial_position=[0.37,0,0.245]、20 Hz、move_from_start=true。
- 老版用 matched_model.json 传递同一手套模型。其 batch 不支持 YAML initial_position，也禁止转发 CLI 参数，所以包装脚本只把生成参数列表中的旧默认 [0.43,0,0.25] 替换为 [0.37,0,0.245]；未改求解数学逻辑。
- 当前 RX 修正规则对本 9 月 18 日记录的左右手均为 recorded_rx_orientation，corrected_frames=0。
- 机械臂 URDF、机械手 URDF 和 calibrated CUDA kernel 源文件在两个测试版本之间无差异。

## 对齐规则

当前最终回放共 250 帧，其中起始过渡 55 帧；历史共 249 帧，其中起始过渡 54 帧。分别去除各自过渡段后，用 source frame_indices 对齐同一原始 195 帧。

两版 frame_indices、时间戳和动作段 wuji_hand_q 全部逐元素完全一致。因此上述 follow 差异不是多一帧过渡造成的。

## 差异最大的帧

位置最大差异发生在原始帧 126（从 0 计数，采集开始后 4.239852 秒）：

- 当前位置：[0.24465180814320975, 0.2299182843343353, 0.0017671400368697299] m。
- 历史位置：[0.2764832742697798, 0.2153324435669574, 0.028674191685898992] m。
- 当前减历史 XYZ：[-31.831466126570064, 14.5858407673779, -26.907051649029263] mm。

旋转最大差异发生在原始帧 50（采集开始后 1.683911 秒），差异 23.681587°。

## 差异所在的处理阶段

两版 native EMF SDK 的左右手局部骨架、旋转和手指 q 均逐元素完全一致。两版右腕 base 位姿及 world_to_base 变换已经不同，后续机械臂 q 与 follow 目标也不同。输出 follow 的 _controller_eef_from_base_palm 函数源码相同。

代码检查发现，当前版本新增从原始 H5 重建 SDK 骨架世界坐标，并调用 sequence_from_sdk 替换用于腕部/MCP 对齐的骨架；旧版沿用采集 JSON 中已有的骨架和腕部目标。这是差异的候选来源；本次是结果对比，没有做逐项消融来量化每项代码变化的贡献。

## 保存的结果

- current_trajectory.json / historical_trajectory.json：最终发布的完整 JSON。
- current_raw.npz / historical_raw.npz：195 帧动作段的求解数组。
- current_final.npz / historical_final.npz：包含起始过渡的完整数组。
- right_follow_per_frame.csv：195 帧逐帧位置、XYZ 差值、位置距离和旋转角差。
- right_follow_comparison.json：右手 follow 详细统计。
- right_follow_comparison.png：轨迹及差异图。
- comparison_report.json：全部手臂、机械手、腕部和 native SDK 数组对比。
- run_current.py / run_historical.py、配置副本和 compare*.py：复现脚本。

远端结果根目录：`/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/ego_fast_retarget/tests/retarget_1_raw_comparison_20261013`。原始采集、生产配置和求解器源码未被改写。

## 后续定位已完成

通过 SDK/URDF 核对和独立消融，follow 差异已定位到当前未提交的世界骨架/腕部外参重建与 sequence_from_sdk 替换逻辑。详见 root_cause_report.md；原候选来源的描述由该验证结果补充。

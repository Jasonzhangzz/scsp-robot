# 旧版腕部处理恢复与删除后的验证

已直接修改远端生产工作目录 `/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/ego_fast_retarget`，恢复采集数据的世界骨架/腕部目标处理，并删除腕部外参重建、SDK 骨架替换及发布阶段相应的 source 骨架覆盖。

## 验证结果

同一原始 episode、pi0、ex001_wuji、同 CQQ 左右手模型、同 initial_position=[0.37,0,0.245]、mcp anchor、20 Hz，与此前 main~2=adde321 的同参数结果比较。

- 完整最终轨迹双方均为 **249 帧**，包括 54 帧起始过渡和 195 帧动作。
- 全部 249 帧的 `follow_right_position`、`follow_right_ee_rotation_6D` **逐元素完全一致，最大差值 0**。
- 左手 follow 两项和左右机械手关节角也在全部 249 帧逐元素一致。
- 动作段 world_to_base、左右腕位置和旋转逐元素一致。
- 机械臂关节角仍有最大 **0.001133970916 rad（0.064971748°）** 的差异。因此没有声称整份轨迹所有数组完全相同。本次删除条件按用户此前重点指定的右手 follow 字段验证。
- 两版动作段机械臂 IK 均为 195/195 成功。
- 15 项相关测试全部通过：EMF 原始 SDK 对照、相机帧关联、丢帧规则、Hand1 SDK 保留、原始 MANO 格式、RX orientation 修正及模型 hash 检查。
- git diff --check 通过；生产 Python 源文件没有对已删除函数/模块的残留引用。

## 删除内容

- official_retarget 和 replay/wuji_replay 两处 sequence_from_sdk 调用。
- SDK wrapper 的 sequence_from_sdk 函数及世界骨架结果传递。
- solve_native 和 align_camera 中 wrist/TX 外参及世界骨架重建逻辑。
- 无其他生产引用的 utils/wuji_retarget/glove_ik.py。
- batch 发布阶段以新 SDK 世界骨架覆盖 source_* 的逻辑及相关策略标签。
- 两项仅验证已删除功能的测试；保留并调整原有 RX/native SDK 集成测试。

SDK 手指求解、模型配置、RX orientation 修正规则及用户其他改动保留。

删除前先恢复骨架使用方式，并实测右手 follow 一致后才删除实现；删除后再次完整运行并确认上述结果。

## 保存位置

- 生产改动 patch：restore_legacy_changes.patch（相对于本次修改前的工作目录，不包含用户其他未提交改动）。
- 数值验证：restored_final_comparison.json。
- 完整轨迹：restored_final_trajectory.json、restored_final_final.npz。
- 运行日志：restored_final.log。
- 原始备份：远端 `/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/ego_fast_retarget/tests/retarget_1_raw_comparison_20261013/before_restore_legacy`。

未切换生产分支、未提交或推送 Git。

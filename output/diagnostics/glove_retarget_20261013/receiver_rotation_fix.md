# cqq_v2 实时与离线对照复核及修复

已部署于 8.147.67.185:1016 的 `/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/ego_fast_retarget`。
之前通过人为弯曲预热把手指捏紧的方案不正确，已完全撤回。MCP 刚体对齐确实不能改变局部手指捏合；本次修复发生在 EMF → glove hand IK 之前。

## 问题与证据

10.136.21.71 上容器实际名称为 `wuji-v3-demo-1`。其启动配置和复制出的 Python 3.12 二进制均为 wuji-sdk 2026.8.31，与离线 Python 3.11 环境版本一致。读取容器文件时未启动容器，也未连接或控制手套、机器人。

实际遥操作代码 `/home/xr/wuji-sdk/wuji-glove/examples/python/retargeting/3.teleop_one_hand.py` 切换至用户 `u_61eaaa7d75eb4a83bdc7d38de0ef1214`，从 `glove.hand_skeleton().subscribe()` 取得 MediaPipe 顺序的 21 个点，直接送入 `RetargetSession.step()`。`4.teleop_dual.py` 为每侧启动独立进程。没有额外手指朝向变换或人为初始化。该用户左右模型 SHA256 与 cqq_v2 完全一致：

- left：`96cfe0dd3770e61667dda41ce9a55b2cb408789a9acd283073544c196b7712ed`
- right：`180a3d54efe171142e323c68e5769d67ed945df8e0ddc9158a3e8ddbaf6d9056`

用容器真实 SDK 二进制的 offline_pipeline 解算同一段原始 807 帧 EMF，左右手骨架与生产离线 SDK 的冷启动结果逐点完全相同。并非两个环境的 SDK 二进制版本不同造成这次异常。

关键是录制输入与新标定的 RX 朝向约定不兼容。该目录中 10 月 8 日的 189 条轨迹，右手 thumb、index、middle 的接收器坐标系相对 10 月 12 日标定呈 local Y 180° 翻转；ring、pinky 和左手没有该差异。10 月 9 日的 210 条轨迹也没有该差异。全目录共 399 条已检查首帧及朝向变化范围。

对照右手成功标定原始记录：`/home/xr/.wuji/calibration/recordings/20261012T094943.011Z-wuji_studio-right-1396c1c1/full.mcap`。其输出模型哈希就是当前右手 URDF。标定 flat_open 与目标首帧的前三个 RX 相对朝向接近 Y 180°，后两个接近零。将前三个 RX 朝向 post-compose `[0,1,0,0]` 后，目标首帧的 URDF 线圈 FK 拟合位置残差从 thumb 16.36 mm / index 17.94 mm / middle 18.90 mm，降至约 0.07 / 2.78 / 3.62 mm，并得到自然放松的手指姿态。这里使用完整位姿拟合，不以更小的捏合距离为目标。

同一 SDK 重放明确标注的标定动作：pinch_index 指尖距离约 12.3 mm，flat_open 约 56.6 mm。造成采集端 RX 朝向约定改变的具体原因尚未确认；没有据此擅自修改固件或 SDK。

## 修复

新增 `utils/wuji_retarget/emf_orientation.py`，在 hand_skeleton.compute 前按显式配置转换 RX 朝向。新增 `hand_models/cqq_v2/emf_rx_corrections.json`，规则同时限定该 collection 的 10 月 8 日 episode 名、右手 SN 和 cqq_v2 URDF SHA256。转换为 `R_input * R_y(pi)`，只作用于右手前三个接收器，不改变位置、置信度、时间戳、源序号或原始 H5。相机精确匹配继续使用原始 EMF payload。

`batch_replay.yaml` 已增加 `glove_emf_rx_corrections` 指向该文件；参数经 batch → replay → SDK hand IK 传递。临时 staging 目录使用原始 episode JSON 名匹配规则，已测试。未匹配的左右手/日期/设备不转换；匹配但 URDF 校验不符时明确报错。

保留之前独立确认的模型一致性修复：手指 q、world 骨架、MCP 对齐及 source MANO21 发布共用本次标定模型解算的骨架。记录中的旧 world wrist 和捕获 TF 只用于恢复 world→TX 坐标变换。

SDK 初始化为 `sdk_native_initialization`，synthetic_frames=0。转换规则、哈希和命中情况写入 `replay_metadata.json` 的 hand_sdk_emf_metadata。

## 验证

- 17 项相关测试通过，涵盖 RX post-rotation、未匹配输入不变、模型哈希、原始输入保留、临时 staging、相机时间对齐及 SDK q 保留。
- 三个 10 月 8 日样本分别完整处理 807 / 685 / 665 个原生帧。首帧右手人手骨架 thumb-index 距离约 49.05 / 49.83 / 52.08 mm；轨迹最小约 9.77 / 4.98 / 10.57 mm。首帧放松和后续捏合都保留。
- 10 月 9 日对照样本的左右手骨架与 q 在启用配置前后逐点相同。
- 转换后的生产骨架与容器 SDK 二进制重放差异最大约 1.49e-8 m（浮点舍入量级）。
- 第一条完整 calibrated.wuji_replay 输出 399 帧，左右手和手臂 IK 均 399/399 成功。
- 第一条右手机器人 URDF tip-link FK：原始首帧 117.53 mm、轨迹 min/median/max 110.01/123.29/158.56 mm；本次修复首帧 65.09 mm、轨迹 14.21/29.39/71.16 mm。这是指尖参考点距离，不是网格表面接触间隙。
- 真实 retiming、54 帧起始 ramp 和数值发布合计 470 帧；发布 source 骨架与本次 IK world 骨架误差 0，source q 与原始 SDK 命令一致。该数值发布测试使用现有 calibration fixture 隔离 RGB 标定构建，没有生成 RGB 视频。

对照图：`receiver_rotation_comparison.png`。数值证据：`rx_validation.json`、`robot_rx_validation.json`。源码快照：`rx_fix_sources.tar`。历史人为预热结果不应继续当作验证依据。

远端备份：`.cache/glove_fix_backup_20261013/`、`.cache/glove_fix_rx_revision_20261013/pre_revision.tar`。
远端本次验证：`.cache/glove_fix_rx_validation_20261013/`；全轨迹日志：`.cache/glove_fix_rx_revision_20261013/validation.log`。

## 重跑

此前 PID 1400745 的批处理使用旧的已加载代码/配置，最后检查时该进程已结束。其旧输出需重新生成。重跑已有配置并覆盖旧输出；不要同时向同一 output 目录运行两个批处理。

```bash
cd /mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/ego_fast_retarget
python3 replay/calibrated/batch_replay.py \
  --config replay/calibrated/batch_replay.yaml \
  --devices cuda:0 cuda:1 cuda:2 \
  --overwrite
```

10 月 8 日右手日志应出现 `RX orientation=explicit_rx_frame_conversion`，左手和 10 月 9 日应为 `recorded_rx_orientation`，两侧初始化均为 `sdk_native_initialization`。需由用户生成 RGB 视频后继续核对真实动作、骨架及机器人接触；目前没有完成全量视频视觉验收。

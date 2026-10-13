> 已被 receiver_rotation_fix.md 的复核结论取代。此前人为弯曲初始化方案已撤回，捏合距离变小不能证明动作正确。以下内容仅保留诊断历史。

# cqq_v2 离线手套修复与验证

修改已部署到 `8.147.67.185:1016` 的
`/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/ego_fast_retarget`。
没有更换 wuji-sdk，没有修改 cqq_v2 URDF 或用户的 batch_replay.yaml。

修复内容：

1. 新增 `utils/wuji_retarget/glove_ik.py`。对显式指定的标定 URDF，使用该模型
   的 FK 构造轻度弯曲初始化（非拇指 flex/PIP/DIP 各 0.3 rad，左右手按屈曲方向取符号），
   预热人手 IK 三帧。合成帧不交给机器人 RetargetSession，不写入输出；
   原始 EMF 时间戳、原生帧数、相机匹配规则保持原样。
2. 从原生 H5 中与 EMF 同源时间戳的捕获 wrist world 位姿，以及捕获时的
   `wuji_glove_tf.h5` wrist→TX 外参恢复 world→TX；再应用当前 cqq_v2 的
   wrist→TX 外参，得到本次解算骨架的 world 位姿。
   旧骨架只提供捕获时的坐标变换，不提供当前手指骨骼几何。
3. `official_retarget.py`、`replay/wuji_replay.py` 的 MCP、手腕、工作空间
   对齐统一使用本次解算的标定骨架。
4. `batch_replay.py` 暂存输入增加录制 TF 文件；发布 source MANO-21 和
   source Wuji q 时使用同一次标定 EMF 解算，并保持原始锚点。模型路径、哈希
   和初始化策略保存到 `replay_metadata.json`。

MCP 是刚体对齐，不改变同一只手内部的拇指—食指距离。下面的捏合改善来自
人手 IK 初始分支变化与其后的机器人手指 retarget，测量完全在机器人 wrist 局部
坐标系进行，不依赖 MCP。

## 右手机器人指尖 FK 对比

相同原始数据、相同 cqq_v2 URDF、相同 WujiHand RetargetSession，单位 mm。
表中数字为最小 / 中位 / 最大距离。

| 轨迹时刻 | 修改前 | 修改后 | 原生帧数 |
|---|---|---|---:|
| 10-08 17:19:04 | 110.006 / 123.288 / 158.557 | 26.434 / 33.700 / 59.469 | 807 |
| 10-08 17:21:28 | 64.922 / 91.954 / 129.929 | 25.616 / 35.137 / 59.313 | 685 |
| 10-08 18:14:52 | 81.424 / 98.925 / 120.486 | 13.223 / 26.027 / 58.569 | 665 |
| 10-09 12:05:21 | 14.162 / 31.852 / 72.106 | 14.178 / 31.752 / 72.228 | 310 |

三个异常样本明显收紧，原本正常的样本数值基本一致。
这些值是 URDF 指尖参考点间距，不是表面接触间隙；没有强行将手指距离压到零。
是否与杯子接触、动作是否符合视频，仍由用户生成数据后的可视化验证。

## 已完成验证

- 14 项相关测试通过：4 项 SDK 保留策略测试、7 项 EMF/时间匹配测试、
  3 项新增标定骨架/坐标变换/实际录制回归测试。
- 原生 807 帧保持原始时间戳；全部 399 个相机帧匹配成功，没有新增丢帧。
- 世界坐标转换保持局部捏合距离，最大数值差约 4e-9 m。
- 第一条完整 `calibrated.wuji_replay` 跑通，输出 399 帧；手臂 IK 399/399 成功。
  轨迹求解时间约 9.2 s（不含环境导入与远端 CPFS 延迟）。

备份位于远端仓库 `.cache/glove_fix_backup_20261013/`。
验证产物位于 `.cache/glove_fix_validation_20261013/`，不在正式数据输出目录。
本地 `fix_sources.tar` 保留修改后的源文件和测试；`pinch_comparison.json`
及两个 `robot_pinch_*.npy` 保留数值对比。

## 用户重跑

在远端按原来的配置运行。已有输出需使用 `--overwrite` 重新生成，或将 YAML 的
output 改为新的对比目录。不要使用 `--skip-existing` 验证旧产物。

```bash
cd /mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/ego_fast_retarget
python3 replay/calibrated/batch_replay.py \
  --config replay/calibrated/batch_replay.yaml \
  --devices cuda:0 cuda:1 cuda:2 \
  --overwrite
```

新日志中应出现 `glove IK initialization=calibrated_fk_flexed_seed`。

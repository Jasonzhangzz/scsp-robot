# DyWA Franka evaluation

统一入口：

```bash
python examples/mpc/franka/ik2/test_mpc_ablation.py --dywa --headless \
  --dywa-objects foam_brick stanford_bunny2 elephant piggy_bank teapot mug rubber_duck \
  --trial-start 0 --trial_num 3 --max_rollout_length 1000 \
  --dywa-output output/benchmarks/dywa_new_run
```

`--dywa` 在加载本地 MPC、ContactSDF、STOCS 之前分发到本目录。
单物体可使用 `--obj mug`；`--dywa-dry-run` 只生成输入，不能计为轨迹。
输出目录须为新目录，防止覆盖或混入历史结果。

## 代码与权重边界

- `upstream/` 提取自 [jiangranlv/DyWA](https://github.com/jiangranlv/DyWA)，
  固定 commit `52eeee070c873f1c65ed62989ba4169551ab3106`。
  `upstream/PROVENANCE.json` 保存 313 个原始 Python/YAML/CUDA/C++ 文件的 SHA256。
  `upstream/PATCHES.json` 和 `history_buffer.patch` 单独记录用户授权的唯一内部修复：
  `HistoryEncoder.put_history()` 中增加 `buffer = self.history`。
  审计检查该文件撤销这一行后必须与原始 SHA256 一致；其余 312 个文件不变。
- `WEIGHTS.json` 固定官方发布的 student、teacher、环境归一化统计与 CORN 编码器文件的
  下载 revision 和 SHA256。四份文件必须全部吻合，不接受随机或部分权重替代。
- `inputs.py` 只准备物体 mesh/质量/摩擦、初始/目标状态、桌面坐标和球形末端输入。
  初始与目标同步平移，保留位移、离桌高度和四元数；任务放不下就报错。
  保留凹物体的碰撞分解。球心对齐 DyWA 原 `panda_tool`，保留其 body/joint/frame
  排列供原控制器使用；这是球形末端适配实验，并非原夹爪硬件的复跑。
- policy 使用官方 `StudentAgentRMA`、感知/归一化包装器、动作分布、动作映射和
  Franka 数值 IK/控制器。不截取前三维、不交给本地 OSC/MPC，不安装旧版
  FlashAttention/PyTorch3D 替代运算。官方 `dt=0.0125`、`action_period=8`，每步 0.1 s。
- 官方 `agent_type=student` 访问不存在的 `actor_net`；使用其原 anneal 分支，
  `alpha0=alpha1=1`，保证包括第一步在内全部选择学生输出，保留官方随机采样。
  teacher 仅提供原包装器的辅助观察/损失信息，不选择执行动作。
- 评估 reset 仅初始化指定任务与学生历史，不做训练用随机动作预热。
  学生保持 `eval()`，推理使用 `no_grad()`。初始化不同 batch 的历史是临时输入状态；
  只重新初始化这类 memory buffer，其余张量严格加载并逐个核对，缺少任何 policy 权重即报错。
  rollout 前后核对参数/归一化 buffer 摘要及磁盘权重哈希，历史 buffer 的正常更新除外。

## 输出

每个 `物体/trial_NNN/` 保存 `spec.json`、`runtime.log` 和启动检查 `preflight.json`。
实际进入 rollout 后才写 `trajectory.csv`，每步记录物体 XYZW 位姿及位置/四元数误差；
成功完成调用后写 `summary.json` 和 `audit.json`。
总表默认是输出目录的 `results.csv`，也可通过 `--dywa-csv PATH` 指定。

`status=completed` 代表完整运行，可成功或任务失败；`error` 是启动/执行异常，
其 success/steps 留空；`dry_run` 只有输入。只有全部请求完成时才输出总成功率，
异常不冒充失败轨迹。误差阈值沿用本仓库：位置 ≤0.02 m、四元数误差 ≤0.015。
`max_rollout_length` 在此是官方 0.1 s 动作步数，1000 步对应最多 100 s。
保留官方内部成功半径、角度阈值与 timeout（默认 300 步），原环境 done 可以提前结束；
外部 2 cm/0.015 判据只负责评分，不改变 policy、控制或原任务的停止条件。

## 环境

```bash
python -m planning.dywa.bootstrap
# 在适配 Isaac Gym 的 Python 环境中安装原生依赖，并构建原 CUDA IK：
PATH="$CONDA_PREFIX/bin:$PATH" CUDA_HOME=/path/to/cuda \
  python -m planning.dywa.bootstrap --build-kinematics
```

源码代码已包含于本目录；第三方仓库仍用于原机器人 mesh 等大资产，默认 `thirdparty/DyWA`。
`DYWA_ROOT` 可指定资产目录，`DYWA_PYTHON` 指定兼容环境，`--dywa-ckpt-dir` 指定权重目录。
依赖包括 Isaac Gym、PyTorch、FlashAttention 1.0.4、原生 PyTorch3D、nvdiffrast、
tensorboard、Hydra/OmegaConf 及官方仓库环境中的依赖。先导入 Isaac Gym，再导入 PyTorch。导入适配还显式绑定官方的 envs namespace，
并恢复 Isaac Gym 使用的旧 np.float 名称（其原含义就是 Python float）；不替换数值运算。

## 本机核验（2026-10-21）

7 物体 × seed 0/1/2 的输入检查全部通过。
`output/benchmarks/dywa_official_7x3_20261021/results.csv` 记录 21 次真实启动尝试，
全部在 CUDA 设备检查阶段异常，**完成轨迹 0 条，不能计算成功率**。
本机 RTX 5060 Ti 是 sm_120，而 Isaac Gym 的 Python 3.8 / PyTorch 2.3.1+cu121
不包含所需 CUDA kernel。支持该 GPU 的 Python 3.11 环境没有匹配的 Isaac Gym 二进制。

此前复现了官方 `HistoryEncoder.put_history()` 中 `buffer` 未定义的 NameError。
用户随后明确授权只补 `buffer = self.history`，现已修复；没有修改权重或其余内部运算。
CPU 上严格加载真实学生网络后，reset 与连续 3 次 forward 均输出有限数值，
形状为 `[1, 2, 20]`（20 维动作的均值及 log std）；训练张量摘要与四份权重哈希均不变。
证据保存在 `output/diagnostics/dywa_history_fix_20261021/cpu_contract.json`。
修复后统一入口的 GPU 启动复查仍在 CUDA 设备检查失败；这项修复不解决 sm_120 兼容性。
CPU 合成输入核验不能当成完整环境轨迹或成功率结果，原 21 次失败启动记录保留不变。

```bash
python -m pytest tests/test_dywa_interface.py tests/test_dywa_official.py -q
```

`planning/dywa_interface.py` 与旧 `experiments/ik2/test_dywa_isaac.py` 仅保留兼容转发。
`legacy_ops.py` 只供旧 ICP 数值回归测试使用，正式运行不会导入它。

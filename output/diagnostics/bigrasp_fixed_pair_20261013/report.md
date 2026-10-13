# 固定接触点对抬升测试（2026-10-13）

两组测试均使用 elephant、物体质量 0.01 kg、MPC horizon 8、控制间隔
0.01 s、600 个周期、acados SQP-RTI（失败时回退 IPOPT）。初始化点对
`[0, 20]` 及左右臂分配全程固定，Lambda 每周期只重算这对点的力。

| 指标 | 默认倾角 0.35 rad | 诊断倾角 0.9 rad |
|---|---:|---:|
| 点对切换次数 | 0 | 0 |
| 双侧物理接触周期 | 0 | 456 |
| 双侧在指定点附近承力周期 | 0 | 0 |
| manipulation 周期 | 0 | 0 |
| 最终高度变化 | -1.54 mm | +4.35 mm |
| 最终水平漂移 | 0.92 mm | 15.27 mm |
| 目标高度变化 | +60 mm | +60 mm |
| 成功 | 否 | 否 |

默认姿态最终状态中出现 `left_link7/right_link7` 碰撞，左、右末端位置
误差约 25.93/14.90 mm。该碰撞未包含在当前简化 MPC 的预测状态中。
诊断倾角能产生双侧接触，但最终末端误差仍为 59.93/35.98 mm；接触点
附近承力判据（距离不超过 15 mm、法向力不小于 0.034335 N）从未双侧
同时成立。因此，本次结果不能用于确认双臂在 best_contact_points
承力后的抬升能力，也不能把任意双侧接触算作成功抓取。默认倾角未修改。

冻结路径绕过在线候选搜索和 verify_cost 切换。可选的
`--allow-contact-switch` 使用两臂各自的接近距离收益 veto，并要求
共同的 Lambda 物体预测收益不下降；它还不是候选点对的整臂 MPC
仿真验证。需要首先解决固定点对的接触建立问题。

复现默认测试：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/zz/miniconda3/envs/acc/bin/python \
  examples/mpc/franka/bigrasp/bigrasp.py \
  --mpc --no-visualize --mppi-device cpu --planner-horizon 8 \
  --control-steps 600 --freeze-contact-pair \
  --scene-output envs/xmls/_generated_bigrasp_online_review.xml \
  --diagnostics-output output/diagnostics/bigrasp_fixed_pair_20261013/reproduced.json
```

第二组增加 `--fingertip-max-pitch 0.9`。两组原始结果在
`trial.json`、`tilted_trial.json`，控制日志在同名 `.log` 文件，汇总指标
在 `summary.json`。原始 JSON 还保存了最终 qpos/qvel 和测试参数，便于
重建物理接触。高度与水平漂移均以初始化物体位姿为基准。

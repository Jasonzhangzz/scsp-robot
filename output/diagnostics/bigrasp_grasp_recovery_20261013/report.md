# 空中求解失败修复验证

本次修改分开了实际双侧承力和指定点对到位，加入夹持确认/释放滞回、
空中恢复夹持状态，并将默认阻塞 IPOPT 回退改为有界保持动作。初始化
的点对及左右臂分配仍固定。

第一次 400 周期测试复现了 `HPIPM_QP=NAN_SOL`，保存了
`trial_solver_failure.npz`。该现场所有 QP 输入矩阵均有限。把动态等式
消元后，用 HiGHS 检查同一 QP 的线性约束得到“不可行”；仅去掉 XY
状态硬边界后可行。加入 XY 松弛变量后，最小必要松弛约 **0.0176 mm**。
这说明该现场需要允许有限制动位移，而非要求已漂移物体立即停止。

将 XY 边界改为强二次惩罚后，使用同一状态、接触几何、质量矩阵、
外力和热启动重放，acados 返回 `ACADOS_SUCCESS`。重放最大预测 XY
超出软走廊约 **0.0834 mm**。原始重放结果见
`replay_soft_corridor.json`。

最后运行 600 周期 elephant 仿真，配置为物体 0.01 kg、horizon 8、
控制间隔 0.01 s、末端倾角上限 0.9 rad、Warp CPU：

- 600 周期全部使用 acados，无 IPOPT、无求解失败回退。
- 规划耗时中位数 **9.69 ms**，P95 **11.73 ms**，最大 **15.51 ms**。
- 210 周期处于 `grasp_recovery`，390 周期处于 `approach`。
- 抬升测试仍未成功：最大高度增加约 **11.66 mm**，最终约 **3.23 mm**，
  目标为 60 mm。不能将数值求解修复等同于完成指定点对的稳定抬升。
- 最大预测软走廊超出量约 **4.95 mm**；软走廊不提供硬位移保证。

日志与完整遥测在 `soft_trial.log`、`soft_trial.json`，汇总在
`summary.json`。回归测试另覆盖未到指定点但有双侧承力、短暂力下降、
微小间隙变化、失败时保持压力、热启动保留，以及已有向外动量时的
可解性。native acados 测试组 49 项通过。

复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/zz/miniconda3/envs/acc/bin/python \
  examples/mpc/franka/bigrasp/bigrasp.py \
  --mpc --no-visualize --mppi-device cpu --planner-horizon 8 \
  --control-steps 600 --fingertip-max-pitch 0.9 \
  --scene-output envs/xmls/_generated_bigrasp_contact_latency.xml \
  --diagnostics-output output/diagnostics/bigrasp_grasp_recovery_20261013/reproduced.json
```

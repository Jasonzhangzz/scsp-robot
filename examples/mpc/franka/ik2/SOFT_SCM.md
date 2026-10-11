# soft-SCM 接入与 Isaac MPC 的动作单位

`test_mpc_isaac.py` 默认在 CSO 接触排序和显式 MPC 中使用 soft-SCM。
默认弹窗；只有传入 `--headless` 才关闭可视化。同步和异步规划均经
`ExplicitMPCParams` 传递同一组接触参数。

```bash
conda activate acc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python examples/mpc/franka/ik2/test_mpc_isaac.py
```

可选参数：

- `--contact-model soft-scm`（默认）或 `--contact-model comfree`。
- `--soft-scm-response-time 0.05`：接触响应时间，秒。
- `--soft-scm-damping-ratio 1.0`：阻尼比。
- `--soft-scm-regularization 1e-8`：正的对角 Delassus 正则项。
- `--table-clearance 0.012`（默认）：给 10 mm 指尖球心保留 2 mm 的桌面
  裕量，避免 OSC 追踪延迟时穿入桌面；需要贴桌接触时可显式传入更小值。
- `--sync-planner`：真正执行“规划一次、推进 10 个 2 ms 物理步”，用于
  排除异步反馈延迟；默认仍为异步和可视化。

在线辨识默认开启，可用 `--no-online-scm` 关闭；参数、数据采集和成对测试见
[ONLINE_SCM.md](ONLINE_SCM.md)。初始默认值仍只是规划起点。

这些默认值是 20–50 ms 规划步长的起点，尚未经机器人闭环标定。
2 ms 仿真基准选出的 `tau=.008, zeta=.25` 可通过参数指定，但不能视为
已经验证过的 CSO 最优参数。旧 `--model_param` 只控制 ComFree 刚度，
不会再改变 soft-SCM 的响应时间或阻尼。

## 接触更新

```text
D = diag(J Q^-1 J.T) + r I
x = h/tau
a = (2*zeta*x + x*x)/(1 + 2*zeta*x + x*x)
b = x*x/(1 + 2*zeta*x + x*x)
p = max(-D^-1 (a J v_free + b phi/h), 0)
v_next = v_free + Q^-1 J.T p
```

零 Jacobian 填充行的权重为零。CasADi/IPOPT、acados、NumPy 后处理和
Torch 路径使用相同接触律；acados 缓存区分模型和参数，避免加载旧动力学。
这里“soft”指有限时间恢复；接触激活仍用硬 `max`，梯度为分段梯度。

## 慢速问题：准动态位移不能当作全动态力

Isaac 参数中的 `Q_robot=robot_stiff_` 是准动态阻抗，而 `cmd` 是末端位移。
该路径显式设置 `contact_dynamics='quasi-dynamic'`：

```text
v_free = curr_v + Q^-1 [external; K_robot cmd] / h
```

这保留 main 分支的无接触位移语义：`q_robot_next=q_robot+cmd`。
此前把该 Q 当作质量，使用 `v_free=h Q^-1 force`，会使位移变成
`h^2 cmd`。在 `h=.02` 时，5 mm 输入仅预测 2 micrometers，增益小 2500 倍。
修正只恢复输入单位和自由运动比例；接触项仍是 soft-SCM。

`ExplicitModel` 仍支持 `contact_dynamics='full-dynamic'`（其默认），用于
真正提供物理质量和外力的调用方。该模式保留 `v_free=curr_v+h M^-1 force`；
不能在两种模式间复用参数而不核对单位。

## 验证记录（2026-10-09）

acc 环境 Python 3.8.20、CasADi 3.6.7、Torch 2.3.1+cu121：

- 87 项相关回归通过，包括隐式标量行参考、稠密有效质量、零填充、
  CasADi/Torch/NumPy 一致性、力梯度和准动态位移雅可比。
- CSO 与显式 MPC 的 IPOPT、acados、Torch CPU 独立求解检查通过。
- 完整入口 `--help` 验证了新参数；保留默认可视化。
- Isaac 默认异步 MPC 完成显式 headless 的 1 trial / 3 步短运行，末端
  动作约 2.2 mm，CSO 排序平均约 2.54 ms。只验证运行和动作恢复，
  不用于评价任务成功率（3 步不足以完成任务）。
- 原有三项 Warp/ComFree 原生测试在 acc 中不兼容：动态模块加载、
  Python 3.8 缺少 `functools.cache`、旧 Warp 的 `vec5` 注解解析。
  它们不属于本次接入路径；保留失败记录，未修改全局依赖。
- acc 的 Torch CUDA 内核不支持当前 GPU；Torch 求解检查明确使用 CPU，
  不能由此宣称 Torch GPU 后端通过。

使用实际 IK2 参数、同一无接触状态与目标、3 步 acados：错误的全动态
解释给出动作范数约 `0.984 micrometers`，修正后约 `2.034 mm`。
这是同状态规划检查，不等于整个机器人任务速度提升了相同比例。

远端 fetch 两次因 TLS 握手失败；与本地保存的 `origin/main`（`36a4a46`，
2026-09-30）比较，自由运动位移和修正后结果一致。未覆盖或回退用户工作区。
测试日志保存于 `/tmp/scsp_soft_scm_acc_*` 和 `/tmp/scsp_soft_scm_ik2_gain_check.log`。

后续翻转诊断见 [FLIP_DIAGNOSIS.md](FLIP_DIAGNOSIS.md)。单步动力学误差
较小不等于准动态 CSO 的接触排序或机器人闭环成功率更高。
python /home/zz/scsp-robot/examples/mpc/franka/scm/compare_prediction.py \
  --engines mjwarp scm \
  --shapes elephant stanford_bunny2 teapot foam_brick rubber_duck mug \
  --modes separation sliding sticking \
  --samples-per-mode 100 \
  --preview-seconds 1 \
  --output-dir outputs/prediction_objects_new

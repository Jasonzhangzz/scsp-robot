# Isaac 翻转诊断（2026-10-09）

已确认执行层的额外桌面高度限制会改变 MPC 动作方向，足以使一个原本
可成功的翻转案例失败。soft-SCM 的单步仿真精度也不能直接转化为当前
准动态 CSO 的准确性。尚未验证整体成功率已恢复。

## 直接因果证据：下压动作被改成上抬

Franka 指尖是半径 10 mm 的球。此前执行器默认把指尖中心参考高度限制
在桌面以上 12 mm，而 MPC 的指尖状态下界为 `table_height - .01`。
两者没有使用相同约束。

当指尖中心在桌面以上 10 mm 时，MPC 的 `u_z=-2 mm` 被执行器改成
`u_z=+2 mm`。这里的位移还进入接触 PD 的 `K*u`，所以这项裁剪会改变
下压力的方向，不能仅理解为对位置做无害的修正。实际非穿透由 PhysX
球体与桌面碰撞求解器处理。

在 acc 环境、foam_brick、trial 0、相同初始状态、候选点和控制周期下：

| CSO / MPC | 执行动作处理 | 首次达到成功阈值 |
|---|---|---|
| main / main | 当前执行器，12 mm 裕量 | 600 步内未达到 |
| main / main | main 的动作执行逻辑 | 约 196 步 |
| soft / soft | 当前执行器，12 mm 裕量 | 600 步内未达到 |
| soft / soft | main 的动作执行逻辑 | 约 99 步 |
| soft / soft | 当前执行器，额外裕量设为 0 | 约 99 步 |

最后两项的全部状态和 MPC 动作数组**逐元素完全相同**。该案例在固定
20 ms 执行周期下的差异来自高度裁剪，而非新的动作排队机制。
额外裕量为 10 mm 或 9.5 mm 的试验也没有达到阈值；只将 12 mm 改成
球半径并不足够，因为参考位移同时负责产生接触下压力。

默认额外裕量已改为 0；`--table-clearance` 仍可显式开启额外限制。
默认值为 0 时仍保留桌面平面处的参考高度下界。

## 接触与异步反馈时序

发现并修复了两个独立问题：

1. OSC 的物理接触缓存只在 `draw=True` 时失效。headless 或少绘图帧会
   继续用旧接触状态选择阻尼与力上限。现在每次 PhysX 推进后失效，同一
   物理帧内仍复用一次测量。发现此问题之前的无绘图消融结果已作废。
2. 异步入口原先按绘图帧发布观测、按墙钟时间刷新接触 Jacobian，可能
   将新位姿和旧接触数据一起交给规划器。现在每 10 个 2 ms 物理步发布
   一次观测，并在该帧重新计算接触数据；绘图周期独立。

模型预测步长与单个动作轨迹均为 20 ms，但这不保证异步反馈的延迟
为零。固定周期消融在 trial 0 成功，而完整异步 headless 入口的
600 次规划测试仍失败（最终位置误差 79.06 mm、姿态代价 0.53909）。
因此不能宣称上述修复已恢复默认异步模式的高成功率。

原有 `--sync-planner` 实际仍启动异步进程。现在该开关通过本进程的
`SynchronousMPCBus` 复用原有规划策略，严格执行“规划一次，推进
20 ms 物理时间”，并在同一帧取位姿与接触。

完整入口同步验证：trial 0 在 **98 步**完成，最终位置误差 **19.47 mm**、
姿态代价 **0.01225**；CSO 排序平均 **5.39 ms**。与临时消融工具相比，
生产入口在达到阈值时直接结束，不再多执行一个动作周期。

另修复 `end_trial` / `stop` 命令可能被 latest-only 队列丢弃的问题。
普通动作继续使用最新值，回合结束命令使用有界的可靠入队。修复后的
异步 2 trial × 30 步生命周期测试正常结束；该短测试不评价翻转成功率。

## 为什么“预测更准”不能直接迁移到 CSO

此前仿真基准使用约 2 ms 步长、物理质量以及携带的速度状态。当前
CSO 使用 50 ms、`Q=diag(50 I, .05 I)` 的历史阻抗、零初始速度及历史
impulse-like 接触变量；MPC 使用 20 ms、阻抗矩阵和位移输入。
默认 `tau=.05, zeta=1` 尚未进行机器人闭环标定。这不是同一个精度比较。

还发现 soft-SCM 在此配置下存在可复现的接触过度恢复：

```text
W = J Q^-1 J.T
D = diag(W) + r I
p = max(-D^-1 (a J v_free + b phi/h), 0)
v_next = v_free + Q^-1 J.T p
```

摩擦锥的四条相关行分别计算反力后叠加，对角近似没有处理其耦合。
在 CSO 的 `h=.05, tau=.05, zeta=1` 下，`a=.75`。将同一桌面 Jacobian
和 `phi=0` 输入两个模型，世界法向输入为 `-10 mm/s`：

| 保存的状态 | main 法向输出 | soft-SCM 法向输出 |
|---|---:|---:|
| step 0 | -9.72 mm/s | +6.84 mm/s |
| step 30 | -9.72 mm/s | +10.96 mm/s |
| step 100 | -9.72 mm/s | +11.86 mm/s |
| step 200 | -9.72 mm/s | +12.02 mm/s |

这说明当前的软响应系数并不保证耦合接触下不会反转或放大速度。相同
状态下，两种模型的接触点排序也会改变。该检查不是整体仿真误差统计，
也不是在证明 main 的物理响应正确；main 在这条 CSO 路径实际使用
`K_contact*h*I` 的先验权重，而非此前基准测试的对角 Delassus SCM。

只把 soft-CSO 行权重减为 1/4 的探索试验，在 trial 0 改善了结果，
但 trial 1–3 未达到成功阈值；不能据此作为通用修正，未修改生产模型。
后续模型改进应处理相关约束的重复恢复，并在当前阻抗、步长及输入语义
下验证接触排序，不能直接套用 2 ms 仿真基准的参数。

## 验证、数据与复现

acc 中 40 项 OSC、接触时序、同步执行和队列相关回归通过；soft-SCM
的 22 项模型回归在独立进程中通过。两类测试混合并以 Isaac 辅助模块
优先导入时出现过原生库崩溃，日志保留；没有据此改动环境依赖。
源码编译、差异空白检查和消融工具 `--help` 检查通过。

比较的是本地缓存的 `origin/main` **36a4a46**（2026-09-30），未成功
获取最新远端。消融保持当前几何、成本、候选点及控制器，不等于完整
分支比较。33 次探索运行覆盖 trial 0–3，预算至多 600 步；未在原始
默认 5000 步、100 trial 下测成功率。本次四姿态短预算对照不能支持
“soft-SCM 成功率显著低于 main”的统计结论。

[运行汇总](../scm/outputs/flip_diagnosis_20261009/summary.csv)、
[协议与限制](../scm/outputs/flip_diagnosis_20261009/manifest.json)、
[trial 0 轨迹图](../scm/outputs/flip_diagnosis_20261009/trial0_execution.png)、
[逐步数据与消融脚本](../scm/outputs/flip_diagnosis_20261009/) 已保存。
轨迹图的横轴是物理时间；姿态指标为 `1-(q dot q_target)^2`，不是角度。
成功阈值为位置误差 `<.02 m` 且姿态代价 `<.015`。
工具环境没有 DISPLAY，实际仿真验证显式使用 headless；默认仍开启弹窗。

推荐先进行可视化的固定周期验证：

```bash
conda activate acc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python examples/mpc/franka/ik2/test_mpc_isaac.py --sync-planner --trial_num 1
```

显式复现高度裁剪对照（去掉 `--headless` 会开启弹窗）：

```bash
python examples/mpc/franka/scm/outputs/flip_diagnosis_20261009/audit_runner.py \
  --audit-cso soft --audit-mpc soft --audit-out /tmp/flip_clearance12 \
  --table-clearance 0.012 --max_rollout_length 600 --headless
```

将 `--table-clearance` 改为 `0`，或使用 `--audit-execution main` 比较
旧执行逻辑。模型对照用 `--audit-cso main` / `--audit-mpc main`；基线
源码和出处随数据保存。复现还依赖当前工作区的物理配置和规划代码。

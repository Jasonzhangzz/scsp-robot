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

在线辨识默认关闭，添加 `--adaptive` 启用；参数、数据采集和成对测试见
[ONLINE_SCM.md](ONLINE_SCM.md)。初始默认值仍只是规划起点。

这些默认值是 20–50 ms 规划步长的起点，尚未经机器人闭环标定。
2 ms 仿真基准选出的 `tau=.008, zeta=.25` 可通过参数指定，但不能视为
已经验证过的 CSO 最优参数。旧 `--model_param` 只控制 ComFree 刚度，
不会再改变 soft-SCM 的响应时间或阻尼。

## bunny 接触与回落修复（2026-10-11）

两阶段 Cartesian MPC 默认携带 19 维状态：物体位姿、指尖位置、
物体世界线速度、物体系角速度和指尖世界速度。仿真接口仍使用原来
10 维 pose；额外速度来自真实 Isaac 刚体观测。`--no-mpc-carry-velocity`
可用于与旧准动态模型作对照。在线 SCM、ComFree 和自定义模型仍使用
原有路径；本节结果使用 `--no-adaptive --contact-model soft-scm`。

- 原来的 `Q` 是阻抗，不能当作质量来积分回落。新模型在无指尖支撑
  时使用仿真物体质量、惯量和重力，逐步携带速度；有压力时保留经验证
  的准动态接触 mobility。两者用有界 quintic 压力权重衔接。这是约化
  混合模型，不是完整的刚体冲击模型，仍需闭环标定。
- 指尖预测积分执行器实际采用的 `K * increment - D * velocity`。
  默认有效质量 2.5 kg、阻尼 40，可用 `--mpc-tip-effective-mass` 和
  `--mpc-tip-damping` 调整。第一步指令边界同时包含执行器的 slew limit，
  避免规划按 5 mm 预测、执行却被限速。终端增加速度项以降低翻起后回落。
- acados 每个 horizon 阶段从碰撞凸包更新桌面支撑点和物体系力臂。
  初始阶段保持实测接触几何；IPOPT/Torch 仍使用支撑 patch 运输。
  RTI 仍只求解一次，没有增加求解后非线性校验或动作拒绝。力上限由
  指令投影保证；真实求解失败仍报告并回退。QP 不等式容差保留原来
  经验证的 1e-8，放宽到 1e-5 的闭环探索未带来可靠改善，未保留。
- 稀疏样本编号变化不再替换实际接触点/法向，也不重置同一物理 patch
  的反馈。已接触且 via 被挡时仍交给 MPC 优化接触和物体目标；不能
  用 approach 指令覆盖它，让保持接触的收益落在 objective 之外。
- `pred_dcost` 与下一观测的 `act_dcost` 按实际发送动作、同一阶段目标
  和固定尺度比较，脱离接触的回落也进入模型误差反馈。`verify_cost`
  保留为诊断 EMA；真实接触的执行权重不再由它反复开关。接触停滞
  按实际 object cost 进展计分，而不是是否已到达目标采样点。
- bunny2 的 flip 默认使用目标 XY 半径 8 cm 区域，允许翻转时质心升高；
  elephant 和其他未标定物体保留原 XYZ 位置项。可通过
  `--flip-position-radius` 显式覆盖，0 表示原位置项。pose 阶段恢复完整
  位置/朝向目标，并增加超过 10° 的倾角惩罚以降低重新推倒的收益。

实测 baseline 中，同一物理 patch 移动 1.96 mm、法向变化 1.75°，最近
采样编号却沿网格跳了 56.6 mm。脱离支撑后五步，实际倾角从 44.21°
增至 56.82°，旧预测仅为 45.4°。新记录重放测试要求回落角误差小于
3°，接触时指尖预测误差小于 0.2 mm，并检查 Torch/CasADi 动力学一致性。
固定记录位于 `tests/fixtures/bunny_contact_transition.npz`。

真实 Isaac/OSC 同步回归使用相同 seed 0/1/2、每次最多 1600 步、70 个
采样点、原生 acados、90° 随机侧躺。bunny 修改前 0/3 达到稳定翻转；
当前 2/3 完成整个位置/朝向任务：seed 0 在 17.12 秒稳定翻转、26.30 秒
完成任务，最终位置误差 9 mm、四元数误差 0.01461；seed 1 在 6.28 秒
稳定翻转并完成任务。seed 2 最终倾角 72.03°，仍未解决，不能声称
所有初始方向都能成功。排序 P95 为 13.59–15.13 ms；seed 0/1/2 的
MPC 失败计数为 7/24/0，仍有真实 QP 失败与 IPOPT 回退。

elephant 使用原 XYZ flip 位置项的速度状态模型：seed 0 在 30.42 秒
达到稳定翻转，1600 步内未完成最终位姿；seed 1 在 16 秒稳定翻转，
28.78 秒完成位姿。MPC 失败计数分别为 8/50，回退延迟仍明显，不能
宣称恢复 developer 的速度。其余 seed 未取得完整结果，不计入成功率。
指标见 `/tmp/scsp_elephant_fix_v19.jsonl` 和
`/tmp/scsp_elephant_fix_final_rest.jsonl`。

184 项相关回归通过，覆盖接触几何、观测传输、阶段目标、执行器限幅、
critic/dwell 和实际记录重放。恢复 QP 容差后另跑 20 项相关用例通过。
日志为 `/tmp/scsp_fix_verified_tests.log`、`/tmp/scsp_fix_verified_force.log`。
未宣称全仓测试通过。

默认异步 seed 0 的 1600 步检查未成功（最终倾角 86.82°）。规划超时
期间执行器持续推进，单步预测与后续观测时间跨度可能不同；本轮同步
结果不能当作异步成功率。复现上述模型效果请显式使用 `--sync-planner`。

原始指标：`/tmp/scsp_bunny_fix_v19.jsonl` 和
`/tmp/scsp_bunny_fix_final_rest.jsonl`。以下 elephant 接触回归章节是本次
速度状态修复之前的历史记录，不能与当前结果混为一组成功率。

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

## 翻转过程中的接触预测

两阶段 MPC 在预测窗口内运输观测到的物体局部接触点：桌面法向保持世界
坐标，指尖法向随物体旋转，转动力臂和法向间隙逐步更新。正间隙未能在
当前步闭合时，不传递指尖压力；桌面支撑和重力继续作用，回落会增加沿途
物体 cost 和终端 cost。接触追踪目标也随预测物体姿态移动，自由接近的
via 仍是世界坐标路径。该近似只运输本帧的接触 patch；新接触由下一帧
PhysX 观测刷新，不在 horizon 内运行完整碰撞查询。

正常规划路径每帧只执行一次 acados RTI，使用动力学 Jacobian 和目标函数的解析 Hessian，
不生成接触动力学的二阶导数。用当前状态和接触几何重建 warm start；输出
指令投影到原有力上限，并报告实际输出指令的预测轨迹。取消非线性残差
校验触发的多轮 RTI/IPOPT 回退；CSO 也不再用 seed/零压力的额外 cost 比较
替换 RTI 指令，不再自动创建独立的 stay/switch rollout
来改写 CSO 换点排序。acados 自身报错时仍保留原有后端回退。

接触执行也不再用“实测 patch 的单步收益低于另一采样点”触发强制撤离。
采样点的收益假定已经到达该点，不能据此跳过当前接触的 MPC horizon；
换点仍通过排序目标和原有 via 执行。实测 patch 与附近目标一致时，使用
实测法向和力臂持续求解，单步收益只作诊断。
翻转阶段恢复 developer 的可信 patch 连续性和近端排序 anchor；原有
实测进展 confidence 可以释放停滞 patch，pose 阶段仍允许重新排序。

`verify_cost` 在 MPC 请求层保留 critic 的 EMA，不因 `if_contact` 或
lift/drop 状态清零。PhysX margin 内、一个有界指令可闭合的实测 patch
即使尚无接触冲量，也可以交给 MPC；其正间隙原样传入动力学，不能提前
传递支撑力。没有实测 patch 或距离不可达时仍执行接近；已接触时 via 被阻挡也通过 MPC 优化。

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

## elephant 接触回归验证（2026-10-11）

已 fetch 并对照 `origin/developer` 的 `cc20b03`。本轮闭环对照的
“修改前”是已有动力学修正的工作区，未运行整份远端分支，不能当作
完整分支成功率比较。以下使用真实 Isaac/OSC、70 个采样点、原生
acados、固定 SCM 参数及默认 90° 随机侧躺，未直接修改物体位姿。

- 同步 trial 0 修改前 600 步后仍倾斜 85.37°；414 个有接触规划帧中
  347 帧执行了 approach，MPC 只实际执行 67 次。
- 最终同步 trial 0/1 各运行 1600 步，分别在 8.38/17.68 秒达到稳定
  翻转，最终倾角 1.07/1.08°。trial 2 在同预算内未翻起（85.51°）。
- 默认异步 trial 0、1000 步，在 8.06 秒达到稳定翻转，最终倾角
  1.01°。该运行没有相邻帧超过 0.3 的 verify 跳变，换点帧总耗时
  P95 为 17.09 ms。以上运行均无 MPC fallback。
- 213 项相关回归通过，覆盖观测传输、正间隙接触、critic 连续性、
  可信 patch 连续性、接触脱离后的回落、原生力限制和后端一致性。
  另行收集旧 `test_blocked_escape.py` 失败：它引用当前工作区原已缺失
  的 `_track_best_via`，不是此次修改引入；没有宣称全仓测试通过。

稳定翻转与整个位姿任务分别计分：上述成功翻转回合尚未完成整个目标
位置/朝向任务，第三个方向也仍有接触排序或预测误差。因此本轮证明
接触校验导致的执行中断已改善，不能据此声称恢复远端的整体成功率。
降低力上限到 0.8 N、取消位置代价或切向正则的单回合探索未解决失败
方向，均未改成默认值。

原始日志与指标位于 `/tmp/scsp_elephant_before.*`、
`/tmp/scsp_elephant_continuity_final.*`、`/tmp/scsp_elephant_async.*`，
摘要为 `/tmp/scsp_elephant_regression_summary.json`，测试日志为
`/tmp/scsp_elephant_final_tests.log`。

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

## 接触采样与凹物体碰撞（2026-10-12）

`--collision-geometry auto` 对 mug、rubber_duck 使用 CoACD 多凸体，
其余物体保持原有单凸包。每个凸体作为同一刚体的独立 URDF collision
元素加载；planner 使用同一组凸体，不能对拼接后的 STL 再取一个凸包。
分解依据 authored body-frame visual mesh，缓存键包含网格、分解配置和
CoACD 版本；构建锁及最后发布的 manifest 避免并行初始化读到半成品。

`--contact-sampling auto` 对 mug、duck 使用按面积生成候选点后的 FPS，
包括大三角面内部。候选点必须位于凸体并集的外露表面，且实际 1 cm
指尖球不能被其他凸体阻挡；内部切分封口面不参与采样。foam 等物体
保留已有密集网格的顶点采样。曲率来自原网格的 incident faces，面内
采样只比较 2 mm 邻域内的相邻网格边，不比较稀疏 FPS 点的 8 个近邻。
连续表面投影使用最近三角面及其真实接触法向，避免吸附到稀疏稳定顶点。
多凸体的接近路径先到接触点外侧的 35 mm standoff，再沿接触法向进入。
到达后释放旧目的地的 latch，接触已到达时直接跟随物体上的 patch；
避免垂直下降先撞到 duck 的头/背，也避免换点后停在旧中间目标。

安装依赖 `coacd`、`rtree`。可用 `--collision-geometry hull` 和
`--contact-sampling vertices` 显式对照旧几何与采样。
每条 trial JSONL 记录实际 collision geometry、凸体数、采样方式和点数。
本轮诊断、消融及回归日志在 `outputs/mpc_geometry_fix_20261012/`；
单 seed 验证不能当作五条的成功率统计。

python /home/zz/scsp-robot/examples/mpc/franka/scm/compare_prediction.py \
  --engines mjwarp scm \
  --shapes elephant stanford_bunny2 teapot foam_brick rubber_duck mug \
  --modes separation sliding sticking \
  --samples-per-mode 100 \
  --preview-seconds 1 \
  --output-dir outputs/prediction_objects_new

# SCM 在线辨识

借鉴的是 ContactSDF 的 on-MPC 状态预测误差学习流程，独立实现，没有复制上游代码：

- [论文 V-B](https://arxiv.org/html/2408.09612v2#S5.SS2)：缓冲执行数据，交替执行 MPC 和参数更新。
- [learning/cases/cube/learn.py](https://github.com/asu-iris/ContactSDF/blob/main/learning/cases/cube/learn.py)：保存 q、u、phi、Jacobian、实际 q_plus，批量更新 theta 后再送回 MPC。
- [learning/core/learn_lse.py](https://github.com/asu-iris/ContactSDF/blob/main/learning/core/learn_lse.py)：CasADi 的预测损失梯度和 Adam。
- [dynamics/lse_dynamics.py](https://github.com/asu-iris/ContactSDF/blob/main/dynamics/lse_dynamics.py)：学习平移/三个转动方向的度量、机器人刚度、重力质量、摩擦与 dSDF 平滑度。

ContactSDF 同时采用 C-SDF 的平滑几何查询。这里迁移了动力学参数学习流程；没有将当前几何查询替换为 C-SDF。高曲率导致的接触点跳变仍可能使 CSO 的候选预测不稳定。

## 参数和更新

机器人刚度 K_robot 和重力质量 m_g 固定。学习六个对数比例：平移阻抗、物体坐标系三个转动阻抗、恢复时间 tau、摩擦 mu。阻尼比 zeta 和 Delassus 正则 r 固定。

```text
theta = log([s_translation, s_rx, s_ry, s_rz, s_tau, s_mu])
S = diag([s_translation^-1/2 (three entries), s_rx^-1/2,
          s_ry^-1/2, s_rz^-1/2, 1 (robot entries)])
Q_inverse(theta) = S Q_inverse_prior S
tau(theta) = tau_prior * s_tau
J(theta) = J_normal + s_mu * J_friction
mu(theta) = mu_prior * s_mu
```

四个摩擦金字塔行的均值给出 J_normal，减去均值得到原来的摩擦部分。初始 theta=0 与原来的 soft-SCM 相同。Q_inverse 的合同缩放保持正定；Q 的基准随当前配置而定。这是**准动态等效阻抗辨识**，不声称六个参数是真实质量、惯量或唯一物理参数。

每个有效样本的损失为：

```text
||p_pred - p_actual||^2 / (0.005 m)^2
+ (1 - dot(q_pred, q_actual)^2) / ((0.05 rad)^2/4)
```

四元数归一化且正负号等价。相较上游代码的四次小角度损失，这里直接使用角度平方的局部近似，以免小转动梯度过弱。只监督物体位姿；不把机械臂当作自由质量拟合。

初始累计 32 个有效样本之后，每增加 32 个有效样本做一次更新。回放缓冲区最多 512 个样本，更新使用最近 64 个；每第五个作为留出，其余训练。六步 Adam、对数空间先验正则、每次更新各参数最大对数变化 0.08。只有留出预测损失下降才接受。阻抗与 tau 的比例限制为 [0.1,10]，mu 为 [0.2,2]。留出数据来自同一条轨迹，具有时间相关性；它不能替代独立测试轨迹。

MPC 的参数输入变为 `[sigma; theta]`。CSO 的阻抗、恢复系数、环境摩擦响应与机械臂接触摩擦锥同时更新；最大力幅值约束仍由原有优化问题提供。CasADi/IPOPT、acados 与 Torch 使用同一组参数。运行时更新不会重新生成/编译求解器；首次运行会生成新的 acados 缓存。

## 执行数据的时间和坐标系

每 10 个 2 ms PhysX 步记录一个 20 ms 窗口，窗口内记录真正经过裁剪、限速、动作调度之后的 OSC 位置目标。u 是窗口末目标相对窗口初实际指尖位置的位移，同时保存目标路径均值供诊断。它拟合的是当前 OSC 执行后的等效响应，不是原始规划命令的理想执行。

观测带最近 32 个完整窗口，因此异步 IPC 丢掉中间观测也不会把不同时间段的动作与状态拼起来。样本使用 trial/id 去重，不跨 episode 拼接。只接收起止都存在指尖接触、时长正确、法向变化小于约 37 度、切向接触力臂跳变小于 1 cm 的窗口；接触进入/退出及大几何跳变会被计数并排除。不是所有物理帧都会进入辨识。

显式模型使用世界坐标的平移速度、物体坐标的角速度；PhysX 接触 Jacobian 的转动列已经转换到物体坐标。CSO 的线速度和角速度均在物体坐标。两者的对比都包括这一坐标系修正。

## 运行

使用现有 acc 环境；默认可视化弹窗和异步规划，在线辨识默认关闭，添加 `--adaptive` 才启用：

```bash
conda activate acc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python examples/mpc/franka/ik2/test_mpc_isaac.py --obj elephant --adaptive
```

默认运行不创建在线辨识器，也不采集辨识数据。`--no-adaptive` 可显式关闭参数更新；旧参数 `--online-scm` / `--no-online-scm` 保留为兼容别名。固定模型也保存数据：加 `--scm-record-data`，该参数不会启用辨识更新。指定输出目录：`--scm-learning-dir PATH`。改变更新间隔：`--scm-learning-min-samples`、`--scm-learning-update-every`。

每个物体/试验输出 `learning.jsonl`、`transitions.npz` 和 `parameters.json`。NPZ 保存初末状态、实际输入、几何、学习前对新样本的预测、同状态的固定先验预测、当时参数和模型配置。每个试验结束保存；中途强行终止仍可保留已写入的更新 JSONL。单试验从先验重新开始，避免不同随机化物理条件间混用参数。

复用同一物体和相同模型配置的参数：

```bash
python examples/mpc/franka/ik2/test_mpc_isaac.py --obj elephant \
  --scm-parameters PATH/elephant_trial_000/transitions.npz
```

检查点加载会校验物体目录、时间步、基准 Q、mu、tau、zeta、机器人刚度、重力质量与动力学约定。该命令冻结已学参数；添加 `--adaptive` 可继续更新。

成对测试入口（默认弹窗，下面显式关闭）：

```bash
python examples/mpc/franka/scm/benchmark_online_scm.py \
  --obj elephant --trials 3 --steps 600 --headless
```

此入口默认同步规划，以保持相同物理步数；`--async-planner` 可切换。它保留两组完整日志、状态数据、JSON/CSV 汇总和预测误差 PNG/PDF。任务的 `final_quat_err` 是原有 `1-dot^2` 成功指标，不是弧度；预测统计另外使用 `2*acos(abs(dot))` 的弧度误差。各试验轨迹不同，因此判断辨识本身的预测改善应看 online 数据上**同状态/同输入**的固定先验与更新模型对比。

## 验证

acc 中数值/梯度、参数约束、留出预测改善、去重和几何过滤、检查点加载、三种后端一致性与角速度坐标转换测试通过。实际结果与命令见 `../scm/outputs/online_scm_elephant_20261009/final/`，分析仅包含完整结束的三个配对试验。

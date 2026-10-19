# 采样接触规划

对应 [dairlib `sampling_based_c3_public` 的 sampling C3](https://github.com/DAIRLab/dairlib/tree/sampling_based_c3_public/examples/sampling_c3) 和论文 [Approximating Global Contact-Implicit MPC via Sampling and Local Complementarity](https://arxiv.org/abs/2505.13350)。

每个控制周期在物体周围采样指尖位置。每个采样点有自己的局部接触模型，再比较预测代价：当前点仍然更好就执行接触计划，否则先沿直线把指尖送到那个采样点。这一段无接触运动就是 dairlib 里的 reposition。

这里不链接 Drake、Gurobi、LCM，也不跑 dairlib 的仿真器。`SamplingController` 的独立示例保留三维点推模型；Isaac 的 `run_isaac.py --method c3` 使用 `rigid.py` / `rigid_sampling.py` 的刚体姿态模型，并通过 `scip_miqp.py` 使用 SCIP 求解原版的 MIQP 投影。

Isaac 模型的状态与原版 jack 顺序一致，共 19 维：指尖位置、物体四元数（wxyz）、物体位置、指尖线速度、世界系角速度、质心线速度。它使用当前 Isaac actor 的质量、中心惯量、质心和接触摩擦，碰撞几何使用同一组凸分解网格（未分解时使用 Isaac 的凸包）。目标四元数进入符号不变的姿态代价。

接触方程移植自 dairlib `solvers/lcs_factory.cc` 的 Anitescu 分支：每个接触有四条摩擦射线，`Jc = Jn + mu*Jt`；互补间隙为 `phi/dt + Jc*v_next`。模型包含指尖/物体、物体/桌面、指尖/桌面接触，以及重力和接触力矩。每个候选求解 C3 后，以满足互补条件的前向仿真重新评分；旋转后的接触几何重新查询。该实现沿用原版的方程和采样结构，没有调用原版 `SamplingC3Controller` 二进制。

优化阶段按 jack 的 `[0,1,3]` 接触裁剪，使用一个指尖/物体接触和最多三个桌面支撑点（通常 16 个二元变量）；指尖/桌面由动力学 QP 的未来结点高度约束处理。评分阶段使用更多支撑点、邻近凸部件和指尖/桌面接触。这能避免把额外的接触二元变量无谓加入每个 MIQP。

每条摩擦射线有一个二元变量，使用 indicator 约束强制 `lambda=0` 或 `gap=0`，同时二者非负；目标仍为二次投影距离。没有以凸 QP 或互补惩罚替代 MIQP。SCIP 的二次目标使用等价 epigraph 表示，并在选定整数模式后解连续 QP 去除数值容差。物理前向仿真的接触 LCP 本身是凸 QP，与用于规划的 MIQP 投影分开。

Isaac 与 SCIP 的原生库在本环境中存在进程内兼容问题，因此 SCIP 在常驻 spawn 进程中运行，最多并行求解四个采样点。默认每个 MIQP 投影限制 0.1 s、相对 MIP gap 目标 0.001；达到时间上限时只接受通过互补约束检查的可行整数解，并在日志中记录实际状态和 gap。若尚无可行整数解，延长同一 MIQP 的预算后重试；仍失败的候选会被拒绝。全部失败时明确报错，不改用凸投影。

| 计划 | 实现 | 代替 dairlib 里的什么 |
| --- | --- | --- |
| `c3` | `csrc/c3_core.cpp`，本机 g++ 编成扩展 | C3 的 ADMM：动力学 QP，再做 `λ ≥ 0` 且间隙 `≥ 0` 的凸投影 |
| `miqp` | CasADi QRQP | `C3MIQP` 的互补投影。每个接触要么分离（力为 0），要么维持接触（间隙为 0），不再用 Gurobi |
| `mpc` | acados SQP | Drake 里交给 SNOPT 的接触隐式非线性规划。间隙随指尖方向变化，穿透和“有间隙还出力”用光滑残差惩罚 |

从仓库根目录：

```bash
python examples/mpc/franka/experiments/sampling/build_c3.py
python -m pip install -r examples/mpc/franka/experiments/sampling/requirements.txt
python -m pytest examples/mpc/franka/experiments/sampling -q -k 'not acados'
```

`c3` 第一次求解时如果还没有扩展模块，会自己编译。acados 使用本机已有的源码树，生成代码缓存在 `~/.cache/scsp-robot/acados/`。

```python
from examples.mpc.franka.experiments.sampling import SamplingController

controller = SamplingController(method="c3")  # "miqp" 或 "mpc"
result = controller.plan(
    ee_position=[-0.05, 0.0, 0.0],
    object_position=[0.0, 0.0, 0.0],
    object_goal=[0.08, 0.0, 0.0],
)
result["ee_command"]      # 下一步指尖位置
result["force_command"]   # 下一步指尖力；reposition 时为 0
result["mode"]            # "c3"、"miqp"、"mpc" 或 "reposition"
```

状态排列是指尖位置、物体位置、指尖速度、物体速度，各 3 维。物体目标也可以只给平面上的 `xy`，高度沿用当前物体高度。

Isaac 试验是单独入口，不经过 `test_mpc_isaac.py` 或 `test_mpc_ablation.py`，也不改它们的步长和代价：

```bash
python examples/mpc/franka/experiments/sampling/run_isaac.py --headless --method c3 --trial_num 1 --max_rollout_length 4
```

Isaac 刚体入口的 `--method c3` 和 `--method miqp` 均选择 SCIP C3MIQP。独立 `SamplingController` 的 `miqp` / `mpc` 点推示例仍可使用；Isaac 入口不把这些点推模型当作翻转模型。

姿态规划步长为 0.05 s（原版 `planning_dt_pose`），执行周期为 0.02 s。每拍只执行下一结点的 0.4 倍，不再把 0.1 s 的整结点压缩进 0.02 s。重定位速度 0.20 m/s，对应每拍最多 4 mm。

```bash
python examples/mpc/franka/experiments/sampling/run_isaac.py --viewer --method c3 \
  --obj elephant --samples 2 --max_rollout_length 400 \
  --rollout-trace-jsonl /tmp/sampling_flip.jsonl
```

日志包含物体位置/姿态、位置误差、姿态误差（弧度）、候选代价、MIQP 状态/gap 和求解失败原因。控制台姿态误差使用度。`--sampling-admm-iterations` 默认 3，与原版配置相同；`--sampling-projection-time-limit` 可增加单次 MIQP 求解时间。当前实现不能以 50 Hz 实时求解；同步运行时每拍耗时会超过 20 ms 的仿真时间。接入姿态模型不代表对任意物体和初始状态保证翻转成功，应以闭环日志中的实际姿态误差验证。

执行接口同时传递位置结点和优化的执行器力前馈，合力仍受原有 OSC 限力和几何保护约束。自由重定位不把 via 误标成接触点，也不使用压力释放伺服覆盖轨迹跟踪。目标生成使用 jack 的 0.15 m / 2 rad lookahead，并保留接近 180 度时的旋转轴滞回。

姿态权重使用原版角误差平方的四元数 Hessian，并按最小特征值进行 PSD 正则化。候选评分采用 `cost_type=3` 的 PD 前向仿真（Kp=100、Kd=0.5），执行采用最终动力学 QP 的名义位置轨迹和执行器力（对应原版 `GetStateSolution` / `GetInputSolution`）。C3 姿态模式使用 85% 改善切换门限；12 拍配置代价下降不足 1.8% 时重新选择接触。重定位目的地另加较强的锁定门限，减少限时 MIQP 代价噪声造成的反复改道。

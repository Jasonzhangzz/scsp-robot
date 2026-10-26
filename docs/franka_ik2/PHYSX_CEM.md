# Lambda 的 GPU 并行 PhysX CEM

用 `--lambda-predictor physx` 启用原生 PhysX 预测；省略时保留 SCM 基线。
接触点排序和单点接触力优化都走这个后端，接口仍是
`choose_contact_points` / `optimize_control_input`。MPC 自身的模型和求解器由原来的参数控制。

```bash
conda activate acc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python examples/mpc/franka/ik2/test_mpc_isaac.py \
  --headless --sync-planner --num-envs 1 --obj foam_brick \
  --sample_num 70 --lambda-predictor physx
```

`planning/contact_force_physx_cem.py` 是进程代理；worker 首先导入 Isaac Gym，
通过 `contact_force_physx_native.py` 调用 GPU PhysX。CEM 采样、目标计算、精英选择
复用 Warp GPU 内核，状态和外力通过 GPU 张量与 PhysX 共享；不执行 SCM 预测内核。
这也使现有 Python 3.8 / PyTorch 2.3.1 环境能够使用 RTX 5060 Ti：
PyTorch 只包装张量描述符，不执行不支持 sm_120 的 CUDA 算子。
可用 `SCSP_ISAACGYM_PYTHON` 指定另一个安装了 Isaac Gym 的 worker Python。

每个“表面候选 × 种群样本”独占一个物体和支撑平面环境，另保留一个临时接触点槽。
加载相同凸碰撞网格/分解块、质量、体坐标系对角惯量和材料；检查实际形状数和质量属性。
默认每个预测窗口是 20 ms，拆成 10 次 2 ms 原生步；
`--lambda-physx-dt` 必须整除规划窗口。每代先运行一次分离空载步以清除上次接触历史，
再完整重置位姿与速度。2 m 网格及独立 collision group 隔离环境，评分前减去环境原点；
布局坐标使用 PhysX float32，故可能有微米级误差。

优化变量仍是接触坐标系的三维力，保留 L1 摩擦锥、总力上限、线/角速度增量上限和现有
位姿/力/切向力代价。该力及其力矩作为窗口内恒定参考坐标系外部扳手施加在物体上；
`tau_o` 包含重力，PhysX 场景自身重力为零，避免重复加载。
环境接触由 PhysX 原生碰撞检测和 TGS 求解，`J_tilde/env_phi` 不进入物理预测。
共享策略每轮把真实观测中的平面变换至 Lambda 的物体参考坐标系，支持倾斜平面。

当前预测场景是物体＋局部支撑平面（1 m 方形静态板，中心取原点在平面上的投影）；
适用于仓库的小物体和短时域桌面任务。机械臂、真实指尖的运动与碰撞、有限桌边、其他障碍物
没有加入这个 Lambda 预测场景。分解网格按各凸块加载，不执行额外 VHACD。

测量完整 Lambda 调用耗时：

```bash
python examples/mpc/franka/experiments/ik2/benchmark_physx_cem_latency.py \
  --benchmark-output outputs/my_physx_cem_benchmark \
  --benchmark-repeats 50 --benchmark-warmup 10 \
  --lambda-predictor physx --obj foam_brick --sample_num 70 \
  --headless --trial-start 0
```

输出目录必须不存在。把 `physx` 改为 `scm` 并指定另一个目录可测量匹配基线。
这是固定初态的优化器基准，计入 GPU 同步和 IPC，初始化另记。
本机结果：[耗时报告](../../outputs/lambda_physx_cem_20261026/README.md)。
70 个采样点、49 个有效候选、24 样本×12 代，预热后 50 次完整 Lambda 平均
159.816 ms，P95 163.783 ms；排序平均 90.662 ms，单点力优化 69.154 ms。
同配置 SCM 平均 8.055 ms。当前 PhysX 配置约 6.26 Hz，无法达到 50 Hz。

原生回归检查：

```bash
SCSP_TEST_PHYSX_CEM=1 python -m pytest -q tests/test_physx_cem.py
```

测试启动独立 worker，因此测试主进程不需要保持 Isaac Gym 的导入顺序。

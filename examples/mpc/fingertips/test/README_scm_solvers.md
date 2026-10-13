# SCM 求解器分支

`test_scm_mujoco.py` 可用 `--solver`（别名 `--solver-backend`）选择
`acados`、`ipopt`、`cma-es` 或 `cem`，默认仍为 `acados`。

| 分支 | 接触点上的力优化 |
| --- | --- |
| `acados` | 原有 ACADOS SQP-RTI；保留现有 IPOPT fallback |
| `ipopt` | CasADi Opti / IPOPT，直接求解原有非线性约束问题 |
| `cma-es` | pycma 全协方差 CMA-ES，使用 ask/tell 更新采样分布 |
| `cem` | EvoTorch 库的 CEM，通过其 `cem` / `cem_ask` / `cem_tell` 接口执行 |

三个新增分支使用同一个 coupled quasi-dynamic soft-SCM 预测图、位姿代价和力正则项。
候选接触点、接触选择策略和 MuJoCo 点力执行器继续共用。
采样分支直接从 IPOPT 目标图导出 CasADi Function，不调用 IPOPT 或 ACADOS；
每个样本都满足法向力上限、摩擦锥和总力上限，再返回本次搜索中代价最小的可行力。
达到采样预算只表示返回可行最优样本，不表示已证明全局最优。

CMA-ES 使用 `cma`，CEM 使用 `evotorch`，均已加入 requirements.txt 可选依赖：

```bash
conda run --no-capture-output -n pi0 python -m pip install 'cma>=4,<5' 'evotorch>=0.6,<0.7'
```

从仓库根目录运行，修改 `--solver` 即可切换：

```bash
SCSP_PLANNER_ONLY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  conda run --no-capture-output -n pi0 python \
  examples/mpc/fingertips/test/test_scm_mujoco.py \
  --solver ipopt --obj foam_brick --steps 5000 \
  --force-output output/scm_solvers/ipopt.json
```

`--solver cma-es` 和 `--solver cem` 使用同样命令。
无 `--viewer` 时无窗口运行；加上 `--viewer` 可查看仿真。
`--execution` 默认 `wrench`，由 MuJoCo 物理推进实际施力结果。
JSON 和同名 CSV 保存逐步轨迹、请求/实际求解后端、规划耗时与成功状态。
达到步数上限仍未成功会保存结果并返回退出码 1；成功并停稳返回 0。
短程 smoke test 未到目标时的退出码 1 与求解异常不同。
`--qp` / `--lcp` 使用各自现有 SLSQP 基线，此处 `--solver` 只选择 SCM 力优化分支。

采样参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--population-size` | 24 | 每个候选接触点、每代的样本数，至少 4 |
| `--population-iterations` | 12 | 每个接触点最多运行的代数，至少 1 |
| `--solver-seed` | 0 | 加到 trial seed 的随机种子偏移 |
| `--cem-elite-fraction` | 0.25 | CEM 精英比例，范围 (0, 1]，至少取 2 个样本 |

两个采样分支每个接触点单独初始化分布，并先评估撤力和纯法向力种子。
CEM 的采样、精英选择与分布更新由 EvoTorch 执行，标准差下限为 0.02。
仅在接口层处理力约束映射和 SCM 目标计算，没有自行实现 CMA-ES 或 CEM 算法。
随机种子按 trial 分离；不修改场景的全局 NumPy 随机状态。
EvoTorch 的 CPU Torch RNG 使用 `fork_rng` 隔离，执行后恢复调用者状态。
增大代数/种群会增加每轮完整接触规划的耗时。

回归验证：

```bash
SCSP_PLANNER_ONLY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  conda run --no-capture-output -n pi0 python -m pytest -q \
  tests/test_scm_force_solvers.py tests/test_runtime_compat.py \
  tests/test_direct_contact_force.py tests/test_grouped_contact_response.py \
  tests/test_shared_cartesian_contact.py tests/test_franka_contact_limits.py \
  tests/test_franka_scene_friction.py
```

新增测试覆盖约束、二次目标优化、种子复现、全局 RNG 不被修改、
三个分支不构造 ACADOS 的真实 MuJoCo 运行，以及在线 SCM 参数变化后的预测/代价一致性。

CMA-ES 接口参见 [pycma 官方文档](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html)。
CEM 接口参见 [EvoTorch 官方文档](https://docs.evotorch.ai/latest/reference/evotorch/algorithms/functional/)。

## 本次验证结果

2026-10-13，在 pi0 环境使用 CasADi 3.7.2、MuJoCo 3.11.0、
pycma 4.5.0、EvoTorch 0.6.1。完整回归测试 **67 项通过**。
以下均为 `foam_brick`、种子 0、真实 `wrench` 执行，成功后满足无外力保持与停稳判据：

| 库求解器 | 起点 | 成功步数 | 最终位置误差 (m) | 最终姿态误差 |
| --- | --- | --- | --- | --- |
| CasADi / IPOPT | 直立 | 285 | 0.017044 | 0.000137 |
| pycma / CMA-ES | 默认随机倾斜 | 461 | 0.005642 | 0.013217 |
| pycma / CMA-ES | 直立 | 58 | 0.015879 | 0.014528 |
| EvoTorch / CEM | 默认随机倾斜 | 489 | 0.016218 | 0.000187 |
| EvoTorch / CEM | 直立 | 282 | 0.016319 | 0.001129 |

直立测试附加 `--init_tilt_min_deg 0 --init_tilt_deg 0 --steps 1000`。
默认倾斜测试使用 `--steps 5000`。这些成功任务均未切换求解后端，
未出现 MuJoCo 不稳定，逐步力满足 3 N 上限和摩擦金字塔约束。
CMA-ES 与 CEM 默认倾斜测试的初始物体位姿相同。
额外的 IPOPT 默认倾斜长测约 20 分钟后主动停止，没有完整结果，
不计为成功或算法失败；本次 IPOPT 的停稳验证范围是直立起点。

轨迹 JSON/CSV、日志及 `summary.json` 保存于
`output/diagnostics/scm_solvers_20261013/`。
其中 `legacy_numpy_cem/` 是已经移除的旧手写 CEM 记录，不能作为当前库分支的验证。
本次仅覆盖上述物体和种子，不能推断其他物体/种子的任务成功率。

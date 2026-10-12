# MPC headless 成功率统计

`test_mpc_isaac.py` 是统一入口，默认物理后端为 Isaac Gym、异步规划。
`--headless --num-envs 5` 创建五个独立 Isaac Gym 进程及独立规划器，保持
原有 CPU PhysX 配置。所有进程使用 spawn，trial 编号和规划历史互相隔离。
默认异步 viewer/headless 共用从第一条指令开始的实时物理时钟，保留 sleep。
原实现没有 viewer 时跳过实时等待，物理推进会超前于 20 ms 预测；修复后
可用 `--no-headless-realtime` 复现旧的无节拍 headless 行为。
逐步诊断默认在两种显示模式下均关闭，可用 `--rollout-verbose` 打开。
默认 flip 和 pose 阶段均将执行器的桌面高度、桌边指令裁剪写入 MPC
约束，避免 MPC 预测向下/向外推压而执行器实际抬升/截断。对照旧 flip
行为可加 `--no-flip-executor-constraints`；pose 的独立开关保持不变。
需要快速同步评估时，显式添加 `--sync-planner`：每次动作
执行 10 个 2 ms 物理帧，关闭渲染、实时等待与 sleep。
两种模式的成功条件均复用 `run_mpc_planner()`；异步模式可能跳过观测，
受求解器和物理引擎的墙钟速度影响，不能用同步结果代替它的成功率。

## acados 修复

新版 acados（0.5.6 起）通过 `ocp.code_gen_options.json_file` 指定生成路径，
`AcadosOcpSolver.generate()` 不再接受 `json_file=`。旧缓存构建函数因此
抛出 TypeError，让 CSO 和 MPC 都回退至 IPOPT。`planning/acados_env.py`
现在兼容两种 API，并在构建锁内检查新版 OCP hash；不匹配的缓存通过
项目的编译流程重建，保留大文件的编译优化。

默认 `--solver-backend auto` 在本机 acados 可用时选择 acados。
原有求解失败时的 IPOPT 回退仍保留。每个 trial 记录配置后端、实际最后
一步后端、初始化错误、acados 求解次数和回退次数。`solver_backend=approach`
表示最后一步处于无需 MPC 求解的接近阶段；用 `planner_solver_backend`
和 `lambda_solver_backend` 确认所加载的求解器。

## 并行 Isaac Gym 评估

使用已配置好 Isaac Gym 的 Python 环境：

```bash
python examples/mpc/franka/ik2/test_mpc_isaac.py \
  --headless --async-planner --num-envs 5 --solver-backend acados --obj foam_brick \
  --trial_num 100 --max_rollout_length 600 \
  --ranking-metrics-jsonl outputs/isaac_trials.jsonl \
  --rollout-summary-json outputs/isaac_summary.json
```

按连续 trial 编号把工作分配到独立环境；每个 worker 依次运行自己分配的
trial，主进程统一保存 JSONL 和汇总，失败／中止／缺失记录不能生成完整
成功率。每个规划进程使用一个 BLAS/OpenMP 线程。

Python 3.8 的 Isaac 环境若无法安装 CoACD，可先在有 CoACD 的环境导出
同一份凸分解，再在 Isaac 中读取；源网格与全部凸体均校验 SHA256。

```bash
conda run -n pi0 python examples/mpc/franka/ik2/export_collision_manifest.py \
  --obj mug --output outputs/mug_collision_manifest.json
conda run -n acc python examples/mpc/franka/ik2/test_mpc_isaac.py \
  --obj mug --headless --num-envs 5 --trial_num 5 \
  --collision-manifest outputs/mug_collision_manifest.json
```

surface sampling 的三角形距离查询需要 `rtree`；本机 `acc` 已补齐 1.3.0。
这不会改变物体几何或成功阈值。

## MJWarp 历史实验分支（不用于当前 Isaac 基线统计）

使用安装了 MuJoCo、mujoco_warp、Warp、CasADi 与 acados 的 Python 环境。
本机 `pi0` 环境已验证可用，无需安装 Isaac Gym：

```bash
conda run -n pi0 python examples/mpc/franka/ik2/test_mpc_isaac.py \
  --physics-backend mjwarp --headless --async-planner --num-envs 8 \
  --solver-backend acados --obj foam_brick \
  --trial-start 0 --trial_num 1000 --max_rollout_length 600 \
  --ranking-metrics-jsonl outputs/mjwarp_trials.jsonl \
  --rollout-summary-json outputs/mjwarp_summary.json
```

- MJWarp 在一个 GPU batch 中推进多个独立 Franka 环境；没有 viewer。
  上述异步命令使用 latest-only 队列，每 2 ms 物理帧接收指令、每 20 ms
  发布观测，保留实时 pacing 和启动 sleep；GPU 物理慢于实时速度时不会
  额外等待。将 `--async-planner` 改为 `--sync-planner` 可运行无 sleep 的
  固定周期评估。
- 每个环境对应一个 CPU 规划进程，拥有独立 CSO、MPC warm start 和阶段历史。
  CPU 使用每进程一个 BLAS/OpenMP 线程；并行数应结合 CPU 核数与显存选择。
- 执行复用现有 Franka OSC、动作调度、力限制与桌面边缘保护逻辑。
  CPU MuJoCo 只计算每帧运动学／接触几何，物理积分由 MJWarp 完成。
  这部分 CPU 工作及每帧 GPU 状态回读仍会限制吞吐，并非全 GPU 规划。
- 按 trial 编号分配固定随机种子，支持不整除的 trial 数、单环境独立重置。
  同步模式每个动作后等待对应观测；异步模式使用与 developer 相同的
  latest-only 总线，结束 trial 和统计记录使用可靠通道。
- 主进程统一写入 JSONL，校验 trial 编号完整且不重复，再输出总体成功率、
  墙钟耗时、吞吐和回退次数。JSONL 为追加模式，summary JSON 为覆盖模式。
- 数值异常、碰撞／约束缓存溢出或规划进程异常会终止评估，避免产生有效
  成功率。缓存容量可通过 `--mjwarp-nconmax` / `--mjwarp-njmax` 调整。
  默认容量为每环境 512 contacts / 2048 constraints，覆盖 mug 等多凸体
  的接触；旧的 512 constraints 在完整 mug rollout 中实测不足。

MJWarp 默认 `--mjwarp-scene isaac` 使用 Panda MJCF 的关节变换，惯性、
质量和质心改用实际 Isaac 资产导出的属性（URDF 没有惯性标签）。
滑动摩擦显式使用 PhysX 默认平均组合规则。使用 URDF 速度上限对每帧
关节速度和位置增量做投影，避免轻惯量末端的数值发散；该投影是
PhysX 速度约束的近似，不能复现其接触与关节约束联合冲量求解。
`--mjwarp-scene legacy` 保留原 MJCF 惯性与 MuJoCo 默认摩擦，便于审计旧结果。
结果记录 scene、异步标志、速度投影次数以及场景 hash。
修改 Isaac 的 URDF／碰撞网格后，应在 Isaac 环境运行
`python examples/mpc/franka/ik2/export_isaac_dynamics.py` 重新导出。
两个物理引擎仍存在接触、速度限制和积分差异，成功率应分别报告。
对齐参数的 MJWarp 场景尚未通过 Isaac 成功率对应验证，不能用于宣称
复现 developer 基线。正式基线对照应使用 Isaac 的原异步入口。
MJWarp 分支专用于 Franka OSC + MPC，不支持 `--floating-fingertip`
或 `--sampling-c3`。

默认关闭逐步诊断；需要排查时加 `--rollout-verbose`。
首次执行可能需要生成 acados 和编译 Warp kernel；汇总墙钟时间包含启动。

## 已验证范围

- acados 原生构建、加载、冷／热缓存与实际 RTI 求解。
- Isaac：`acc` 环境真实 headless 1 trial / 2 步运行，CSO/MPC 均加载
  acados，初始化错误和 IPOPT 回退均为零。验证时使用现有 `pi0/bin/ninja`
  并将 `acc/lib` 加入共享库搜索路径。
- MJWarp：2 环境、3 trials（含非零起始编号和后续重置），2 环境各 80 步
  接触 rollout，以及 rubber_duck 的 64 个碰撞凸体短运行。
- GPU 测试校验 20 ms 动作时间、单环境重置隔离和已结束环境冻结，
  同时将 `time.sleep` 替换成抛错函数验证无 sleep 的物理执行路径。

这些短运行用于检查执行与统计逻辑，不构成完整任务成功率实验。
实测同步 Isaac 的 foam_brick seed 0 全程成功轨迹及 bunny seed 0 的
前 250 周期，在真实 viewer 与 headless 下状态和动作逐元素完全一致。
这只验证同步模式下关闭渲染不改变执行，不保证异步墙钟调度逐帧一致。

## 每物体五条的汇总

使用全新的输出文件运行 seeds 0–4，然后执行：
`python examples/mpc/franka/ik2/summarize_rollouts.py outputs/你的结果目录`。
目录中应有七个物体各自的 `<object>.jsonl`，支持 Isaac 和 MJWarp，禁止混合
不同物理后端。工具检查完整 seeds、重复记录、
中止标志和 planning 周期数，输出统计表、逐 trial CSV、成功 seed 和
完整 batch 复现命令。成功耗时使用实际仿真时钟（异步不能用 steps × 20 ms），
同时单独报告成功墙钟时间，并区分曾稳定翻起和最终任务成功；最终
`success` 仍要求位置误差 ≤2 cm、四元数误差 ≤0.015，且进入 pose 阶段。
planning 为 CSO＋接近控制＋MPC callback 的
墙钟耗时，排除物理、IPC、渲染。std 为样本标准差（ddof=1），planning
统计合并全部 trial 的周期；没有成功案例时耗时为空。

随机种子能复现初始样本，异步调度不保证逐帧确定性。
改变 batch 大小会改变数值路径，复现时保留完整 seed 集合、num-envs、
scene、调度模式及软件环境，而不只是单独运行成功 seed。

当前原生 Isaac 五条 bunny 检查：开启节拍等待后，曾稳定翻起 5/5，最终
任务成功 3/5；同 seeds、相同 70 个候选点、未改变代价和成功阈值的无节拍
对照为 2/5 曾稳定翻起、0/5 最终成功。默认采样未包含高曲率／不稳定法线
点，35／70／140 点审计也未显示高曲率过采样。剩余失败需区分翻起后偏离
目标位置／偏航与再次倒下。五条是诊断样本，不能据此保证总体成功率。
真实异步 viewer/headless 单 seed 重复仍有轨迹差异；严格轨迹核对请使用
同步模式。详细数据见 `outputs/bunny_isaac_diagnosis_20261012`，原生 Isaac
七物体各五条的统计见 `outputs/isaac_five_trials_20261012`。

## 启动缓存与接触切换修复

启动阶段现在仅构建所选求解器；备用 IPOPT 在首次发生回退时才创建。
同进程的规划参数／接触优化器复用已有实例，仿真进程只构建物理参数。
`[mpc-startup]` 和 `[acados]` 消息在默认安静模式下也会显示构建进度。

`acc` 使用的 CasADi 3.6 存在额外的缓存问题：相同数学模型的
`StringSerializer` 输出会因 SX 节点共享关系不同而变化，新版 acados
的原始 hash 因此每次都不匹配。缓存比较现在将所有符号表达式统一为
一个执行 CSE 的 Function，再比较完整 OCP 配置；动力学、代价、维度、
约束、求解设置与编译环境仍参与校验。JSON 与共享库的时间顺序检查
仍保留，防止读取中断编译遗留的旧二进制。接触求解器缓存也区分
CasADi 版本，避免 `acc` / `pi0` 互相覆盖。

接触切换时清空不再适用的 MPC warm start 和内部对偶／QP 状态。
CSO 的各个候选点分别重置内部状态，并用该候选力的实际预测状态
初始化末端状态。移除了由保守法向力上限和摩擦金字塔已保证的冗余
力球约束，允许的接触力集合没有改变。

捕获的 MINSTEP 案例还存在真实不可行性：末端已经具有向下速度，
有限力和动作平滑约束无法使后续状态立即满足净空硬边界。速度状态
MPC 现在仅将末端高度边界设为有高代价的恢复软约束（线性权重 1000、
二次权重 1e6），初始实测状态、力上限及执行器桌面制动保持原有约束。
从自由空间进入较低接触力上限时，动作平滑窗口以力上限内的前一动作
为中心，避免平滑窗口与力球没有交集。逐 trial 的
`mpc_floor_recovery_max_m` 和 `mpc_contact_resets` 提供对应诊断。

验证结果：

- Isaac `acc` 相同命令的规划器就绪时间从约 27.7 秒降到 0.73 秒，
  没有重编译；150 步、72 次 MPC 求解和 7467 次 CSO 求解均无失败或回退。
- MJWarp 250 步、15 次候选接触点切换均无 MPC／CSO 失败或回退。
- 原始失败状态已保存为 `tests/fixtures/mpc_switch_floor_recovery.json`，
  回归测试覆盖冷／热初值、力上限及净空恢复。

## bunny2 冷启动与 developer 对照

bunny2 首次启动原本需要编译约 15 MB 的动力学雅可比 C 文件；
仅禁用大型文件的编译优化仍要约 29 秒。现在用一个解析方向导数
子函数组成雅可比，生成文件约 2.6 MB，冷编译约 8 秒。
这没有减少接触迭代或改变动力学：回归测试对比原表达式的预测状态、
所有雅可比列，并验证捕获的 MINSTEP 状态仍满足力限制与高度恢复。
总启动日志现在分别记录物理初始化、规划器就绪和首次指令时间。

本机 `acc` 的 bunny2 异步入口实测：首次指令从约 32 秒降到
冷缓存约 13 秒、热缓存约 6 秒。总时间包含 Isaac/PyTorch 导入和
进程启动，不能与单独的 `[mpc-startup]` 时间混用。
不同 CasADi 版本产生的不兼容图缓存会自动重建，避免
`Unknown type21` 将缓存读取问题变成 IPOPT 回退。

与远端 developer 对照时应使用同一物理后端、trial 编号、默认参数
及 5000 步预算，并将两边都固定到同步 20 ms 的执行方式。
旧分支的 acados 加载器不兼容本机 API，因此对照工具只在内存中
替换该加载器，保留旧控制器、约束和候选点初始化逻辑。
不能用异步 viewer 或 MJWarp 的成功率替代这个 Isaac 对照结果。

```bash
git fetch origin developer
git worktree add --detach /tmp/developer-reference/scsp-robot origin/developer
python examples/mpc/franka/ik2/compare_developer_rollouts.py \
  --reference-root /tmp/developer-reference/scsp-robot \
  --objects foam_brick stanford_bunny2 elephant \
  --trials 1 --steps 5000 --workers 2 \
  --output-dir outputs/developer_comparison_run1
```

工具同时保存初始物理状态、目标、模型参数、候选点和实际初始观测，
对照它们是否一致，并检查完整 trial 编号和中止状态。
输出目录需为新目录，防止将不同运行追加到同一个分母。
`mug` / `rubber_duck` 的默认凸分解几何要求仿真环境安装 `coacd`。
少量成功案例的对应性检查用于排查回归，不能估计总体成功率。
- `pi0` 的 120 项相关测试、`acc` 的 11 项缓存／接触切换测试通过。

旧的展开式首次生成新配置约需 30 秒；当前 compact JVP 的 bunny2 冷编译
约 8 秒。之后等价配置应直接使用缓存。
上述时间是本机短运行结果，不是完整任务成功率或所有物体的性能保证。

翻转后接触点诊断可在原启动命令上加：

```bash
--rollout-trace-jsonl /tmp/bunny-contact-trace \
--rollout-planner-snapshots /tmp/bunny-contact-snapshots
```

trace 的 `contact_predictions` 区分 λ 假设已经到达的最佳候选、λ 实测接触、
MPC 实际指令首步与 horizon 终点；每项都有 world-frame 位姿、position 改善
(mm) 和各成本项的改善。`observed_position_reduction_mm` 是相邻规划观测之间
的实测改善；异步模式可能跨多个执行命令，不能视为单条命令的因果效果。
快照按 object/trial 命名，保存第一个实测 pose 接触请求。详细 bunny 对照见
`outputs/bunny_pose_contact_diagnosis_20261012/REPORT.md`。

Pose 阶段的执行约束对齐现默认启用（`--pose-executor-constraints`）：MPC
纳入执行端的桌面端点高度下界和当前桌沿方向限制，避免预测依赖执行端会
裁掉的 push 分量。`--no-pose-executor-constraints` 用于对照此前版本。
这项改动不启用主动撤离接触的策略，且不作用于 flip 阶段。
诊断与实验限制见 `outputs/postflip_push_diagnosis_20261012/REPORT.md`。

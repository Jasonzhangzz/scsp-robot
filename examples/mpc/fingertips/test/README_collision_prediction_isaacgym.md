# PhysX 原生碰撞预测基准

本入口对固定的 MJWarp 实验初态比较 SCM、ComFree 与 IsaacGym 内置 PhysX。
它复用来源中 `selected=True` 的查询、初始位姿与速度、质心外力，不重新采样。
完整来源包含 8000 条查询；PhysX 会重新判定接触类别与起点接触点数，因此不保证重新分类后每个物体、每种类别仍有 200 条。
原来的 MJWarp 入口及论文表格均不改变。

## 时间与模型

| 方法 | 步进 | 接触信息与动力学 |
|---|---|---|
| PhysX 参考 | 0.002 s × 10 | IsaacGym 原生 collision 与 PhysX 接触求解 |
| SCM | 0.02 s × 1 | PhysX 起点 manifold；现有三分量平滑 SCM |
| ComFree | 0.02 s × 1 | 相同 PhysX 起点 manifold；固定上游版本的约束装配与 GPU 求解 |

这是在外部求解器边界接入原生碰撞记录的混合基准，不是把 PhysX 内部求解器替换为 SCM／ComFree。
SCM／ComFree 保留当前自由运动计算与位姿积分；禁止从 MJWarp 读取碰撞点，也不使用 PhysX 未来的接触点或接触力预测。
ComFree 关闭内部 collision detection，注入同一份已保存的初始 manifold。
参考中球心的外力在十个子步均保持相同，因此总外部冲量为 `F × 0.02 s`；这不是把同一完整冲量重复十次。

主进程使用 `pi0` 环境，独立 worker 使用 `acc` 环境加载 IsaacGym，避免动态库和 Python 版本冲突。
采用 **CPU pipeline + GPU PhysX**，默认同物体 16 个独立环境并行；CPU pipeline 是状态与原生逐点接触读取接口，不代表 PhysX 在 CPU 求解。
独立批次重新创建参考场景，避免接触历史缓存；`prepare_sim()` 后再次完整写回初态。
所有环境原点固定为零，使用独立 collision group 隔离，避免网格布局平移引入浮点精度变化；运行时检查跨环境接触。
默认 TGS、8 次位置迭代、1 次速度迭代、contact offset 1 mm、rest offset 0、恢复系数 0、摩擦 0.5，关闭额外滚动与扭转摩擦。实际设置和软件版本保存在 manifest。
GPU 接触缓冲区固定为 8,388,608，额外缓冲倍数为 16，防止复杂凸分解模型的 broadphase 漏接触。原生 `invalid parameter`、容量不足及可能丢失交互的诊断均作为失败处理；大批次可显式拆分重试，保存请求／实际 world 数与原始原因，不静默换引擎。

目标、小球、地面使用相同几何、质量、质心、惯性、摩擦和重力。
复杂物体直接加载现有 CoACD 凸块，关闭 VHACD 与自动惯性重算，校验实际形状数量及质量属性；失败明确记录，不回退为单凸包或 MuJoCo 碰撞。
圆柱用固定 96 段凸网格表示，记录其相对于解析圆柱的离散化误差。
PhysX 对各凸块的内部 cooking 可能简化顶点；三个 Isaac 基准方法共享 cooking 后的原生 manifold，不声称它与 MJWarp 的解析圆柱及原始凸网格逐顶点一致。
SDK 接触记录不提供的凸块身份保持 unavailable，不能据此声称测量了参与接触的凸块数。

## 运行

在仓库根目录执行，输出目录必须不存在，避免覆盖已保存实验。路径可按实际位置调整。

```bash
/home/zz/miniconda3/envs/pi0/bin/python examples/mpc/fingertips/test/benchmark_collision_prediction_isaacgym.py run \
  --input-dir outputs/collision_prediction_200_per_mode_20261026_job/data \
  --output-dir outputs/collision_prediction_isaacgym \
  --reference-batch-size 16 --comfree-batch-size 16
```

`--objects sphere cube cylinder foam_brick` 筛选物体；`--limit-per-object 8` 限制每物体的来源查询数。
同一物体的查询可以 GPU 并行，不同物体顺序处理。SCM 先计算，ComFree 再批量复用完全相同的初态和 PhysX manifold。
批大小 1 用于核对串行一致性。源文件哈希随查询保存，不能用来源 MJWarp 接触标签替代 PhysX 的重新分类。

先执行小规模 smoke test，默认三个基本体与 foam_brick：

```bash
/home/zz/miniconda3/envs/pi0/bin/python examples/mpc/fingertips/test/benchmark_collision_prediction_isaacgym.py smoke-test \
  --input-dir outputs/collision_prediction_200_per_mode_20261026_job/data \
  --output-dir outputs/collision_prediction_isaacgym_smoke --limit-per-object 8
```

添加 `--objects all` 检查全部资产。固定子集可能缺少某些 regime/count/geometry，这些缺口作为覆盖率输出，不冒充达到配额或物理上不可实现。
`smoke-test` 自动执行实际 GPU 串行／批量、顺序重排、A-B-A 重置、起点几何及五档自由飞行冲量审计，结果写入 `audit.json`。主检查与完整 11 帧接触数序列的严格检查分别报告。
合成方向、冲量及接触字段验证场景不计入主实验统计。
通过小规模检查后可用 `nohup ... > job.log 2>&1 &` 后台运行；根据同机小规模耗时估算 ETA，并保存启动指令，不需要持续监视。

离线重新生成报告不需要 IsaacGym 或 GPU：

```bash
/home/zz/miniconda3/envs/pi0/bin/python examples/mpc/fingertips/test/benchmark_collision_prediction_isaacgym.py report \
  --input-dir outputs/collision_prediction_isaacgym
```

精确回放使用已保存的查询 NPZ，输出到新目录：

```bash
/home/zz/miniconda3/envs/pi0/bin/python examples/mpc/fingertips/test/benchmark_collision_prediction_isaacgym.py replay \
  --query outputs/collision_prediction_isaacgym/sphere/queries/QUERY_ID.npz \
  --output-dir outputs/collision_prediction_isaacgym_replay
```

只对已完成的相同查询进行 ComFree 批大小统计：

```bash
/home/zz/miniconda3/envs/pi0/bin/python examples/mpc/fingertips/test/benchmark_collision_prediction_isaacgym.py comfree-batch \
  --input-dir outputs/collision_prediction_isaacgym \
  --output-dir outputs/collision_prediction_isaacgym_comfree_batches --batch-sizes 1 16 64
```

此命令复用已保存初态、起点 manifold 与 PhysX 终态，不重新采样，不重算参考或 SCM。
批量 API 的纯物理耗时与含分配、上传、诊断的总耗时分别记录，不能把前者解释为完整实验加速比。
输出 `throughput.csv` 的批次耗时 mean/std/RMSE/median/p95、吞吐量，以及无失败时相对 batch=1 的纯物理加速比；失败批次不报告清洁吞吐量或加速比。
`source_queries.csv` 为保存查询、起点 manifold、qpos/qvel/wrench 分别记录 SHA-256，`batch_parity.csv` 对比各批大小与来源保存的 ComFree 预测，`batch_verification.json` 检查相同查询身份与模型资产。

## 接触与终点诊断

起点 manifold 来自独立的原生碰撞探测副本：零速度、零重力、零外力，运行一次 2 ms 探测，丢弃其动力学结果和载荷。
两种预测器只读取该起点几何，含 contact offset 内的候选点及带符号间隙。
这份候选 manifold 与统计中 `gap <= 0` 的实际接触点计数区分保存。

PhysX 参考保存 0–20 ms 的 **11 帧状态**与十次真实求解接触记录。
**20 ms 终点的额外接触诊断在独立副本上求解 20–22 ms，其诊断力不属于 0–20 ms，且不会改变参考的 20 ms 终态。**
该时间语义同时写入 manifest 与接触记录，避免将额外求解误算成 12 帧参考运动。

按物体–地面、物体–小球交互对判定加载接触，法向载荷阈值 `1e-4 N`：

- 由无加载接触转为有加载接触记 Onset（存储键 `contact_establishment`）；反向变化记 Separation。
- 同一区间两种事件均发生记 `mixed_transition`，保留两个事件标记。
- 无转换且有加载接触时，任一点切向相对速度大于 `1e-3 m/s` 记 Sliding，否则 Sticking。
- 全程无加载接触归 Separation，同时保存 `free_flight=True`。

这些是窗口层面的 contact regimes，不穷举所有逐点模式组合。
起点接触点数只计目标与地面、小球的实际几何接触点（`gap <= 0`），分 0、1、2、≥3；排除球–地面，摩擦约束行不额外计数。
另存两种交互分项、区间最大值和终点计数。
名义几何配置标签保持来源值；实际 floor manifold 的仿射秩用 0.1 mm 容差区分 none/point/line/face，两者不能混称。

## 指标、失败与交付

只评估目标物体在 20 ms 的终态。绝对指标与原基准相同：

- 位置：`||p_pred - p_ref||`，展示 mm。
- 旋转：`2 acos(|q_pred · q_ref|)`，四元数先归一化，展示度，正负号不影响结果。
- 辅助位姿：`sqrt((position_m/L)^2 + rotation_rad^2)`，`L` 为固定体坐标包围盒对角线；这是自定义综合指标。
- 另存 `1-(q_pred · q_ref)^2` 以及世界坐标线速度、角速度误差。

相对位置／旋转／位姿误差除以参考在 20 ms 内的对应运动量，速度误差除以参考终态速度幅值。
分母下限分别为 `1e-6 m`、`1e-4 rad`、`1e-4`、`1e-4 m/s`、`1e-3 rad/s`；不足时为 undefined，不能记零。
相对误差为倍数而非百分数。静止附近的运动归一化比值可能很大，主要展示绝对位置和旋转误差。
PhysX 相对自身误差为零仅作为一致性检查；失败和非有限预测保留原因并从有效误差集合排除。

输出包含：

- 配置、版本及源查询哈希 manifest；模型 XML、加载资产与碰撞形状／质量属性验证。
- 每查询输入与各方法终态 NPZ、11 帧参考状态、起点 manifold、原始及统一坐标接触记录。
- 每物体 `predictions.csv`、`samples.jsonl`、`reference_contacts.jsonl.gz`；根目录绝对／相对误差 `results.log`。
- `summary.csv/.md`：物体×PhysX regime×起点接触数组，保留 mixed、失败和空组；mean、population std、RMSE、median、p95、总数、有效数、失败数、定义／未定义数。
- `stratified_summary.csv`：外力、名义初速度、seed、名义几何分层；`geometry_summary.csv`：物体×regime×名义几何。
- `case_counts.csv/.md`：各边缘及联合组的查询数，valid-reference 少于 100／200 标记；无新增目标配额，target/shortfall 为空。
- `summary.log`：可机器读取的 JSONL 汇总；`coverage.json`：实际模式、接触数和实测几何覆盖，缺口与数据完整性错误分开。
- 位置／旋转误差分布与各自接触数热图 PNG/PDF，报告实现及输入 CSV 哈希 `report_provenance.json`。

报告可重复生成且不改写原始查询、结果或 manifest。读不到有效查询的接触记录属于完整性问题；固定样本缺少某个分组属于覆盖缺口，不能混为同一种失败。

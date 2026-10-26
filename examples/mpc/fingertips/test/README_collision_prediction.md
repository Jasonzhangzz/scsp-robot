# 小球碰撞的 20 ms 预测基准

入口为 `benchmark_collision_prediction.py`。这是本仓库新增的共享状态预测实验，
不运行 MPC，不把目标跟踪误差当作动力学预测误差。

| 方法 | 推进 | 实现 |
|---|---|---|
| SCM | 0.02 s × 1 | `scm_box`，CPU float64，三分量平滑 Coulomb box，3 轮 GS |
| ComFree | 0.02 s × 1 | 上游 GPU 引擎，stiffness=0.1、damping=0.001 |
| MJWarp | 0.002 s × 10 | 上游内置 GPU 引擎，作为数值参考 |

SCM 使用温度 `1e-4 N s`、正则项 `1e-8`，不启用额外 MuJoCo 软接触对齐。
MJWarp 行的误差为零，仅作一致性检查。SCM 是该接触律在双自由刚体上的动力学评估；
本实验不比较三种方法的 CPU/GPU 耗时，也不将数值参考误差解释为真实物理误差。
额外 smoke test 可单独测量 MJWarp 的串行与多 world GPU 吞吐量。

## 运行

使用已准备好固定依赖的 `pi0` Python 和 NVIDIA CUDA GPU：

```bash
cd /home/zz/scsp-robot
PYTHON=/home/zz/miniconda3/envs/pi0/bin/python
ENTRY=examples/mpc/fingertips/test/benchmark_collision_prediction.py

# 离线准备七个复杂物体，命中内容哈希缓存时直接复用。
$PYTHON "$ENTRY" prepare-assets

# 基本体和一个复杂物体的小规模检查。
$PYTHON "$ENTRY" run --objects sphere cube cylinder elephant \
  --seeds 42 --samples-per-mode 2 --headless \
  --output-dir outputs/collision_prediction_smoke

# 完整实验：10物体 × 5 seeds × 4种实测状态 × 20条 = 目标4000条主查询。
$PYTHON "$ENTRY" run --headless --output-dir outputs/collision_prediction_full

# 原生 GPU 并行：每个物体的16条独立查询共享模型，状态和外力独立。
$PYTHON "$ENTRY" run --headless --reference-batch-size 16 \
  --output-dir outputs/collision_prediction_full_parallel

# 每个物体×实测mode共200条：5 seeds，每个seed每种mode收集40条，总目标8000条。
nohup $PYTHON "$ENTRY" run --headless --reference-batch-size 16 \
  --samples-per-mode 40 --output-dir outputs/collision_prediction_200_per_mode \
  > outputs/collision_prediction_200_per_mode.log 2>&1 < /dev/null &

# 独立的额外 smoke：10物体，每种模式6条主查询，额外核验实际几何与串并行结果。
$PYTHON "$ENTRY" smoke-test --headless \
  --output-dir outputs/collision_prediction_extra_smoke

# 对已保存数据核验 mode/count/geometry 覆盖；加 --verify-parallel 实测 GPU 并行。
$PYTHON "$ENTRY" smoke-check --input-dir outputs/collision_prediction_extra_smoke \
  --verify-parallel --batch-size 16

# 无需 GPU 的离线重新统计。
$PYTHON "$ENTRY" report --input-dir outputs/collision_prediction_full

# 从 queries/ 中选择一条完整初始状态，重新运行三种方法。
$PYTHON "$ENTRY" replay \
  --query outputs/collision_prediction_full/sphere/queries/sphere_s42_000000.npz
# 添加 --viewer 循环慢放参考的11帧，不影响统计。
```

运行拒绝覆盖已有输出目录。`run` 始终无窗口；`--headless` 可显式标明批处理意图。
可使用 `--objects`、`--seeds`、`--samples-per-mode`、`--max-attempt-multiplier`、
`--asset-cache`、`--coacd-threshold`、`--reference-batch-size`、`--nconmax`、`--njmax`、`--device` 调整实验。
其余物理和接触律默认值集中在 `collision_prediction_core.Settings`。
GPU 包固定为本仓库 `examples/mpc/franka/scm/requirements.txt` 中的版本；
CoACD 使用 1.0.11，trimesh 使用当前环境，实际版本全部记录在 manifest。

SCM 批次完成后，可用完全相同的保存样本单独测试 ComFree 原生 GPU 并行：

```bash
$PYTHON examples/mpc/fingertips/test/benchmark_comfree_saved_batch.py \
  --input-dir outputs/collision_prediction_200_per_mode_20261026_job/data \
  --output-dir outputs/comfree_saved_batch \
  --batch-sizes 1 16 64
```

该入口只读取四种主要mode的已选查询，不重新采样、不重新计算SCM或MJWarp。
每个batch size使用相同的8000条完整初态和外力，输出相对同一MJWarp终点的绝对/相对误差，
以及相对已保存ComFree单world预测的差异。`source_queries.csv`保存来源路径和逐文件SHA256，
`summary.csv`按物体×mode×起点count统计，`geometry_summary.csv`按几何统计，
`throughput.csv`记录物理推进耗时、逐batch时间分布、吞吐量、真实多world数和重试数。
预热不计入物理计时；另列包含预热、分配和上传的rollout总耗时，误差计算不在该总耗时内。
出现原生告警的batch拆分重试；有重试时不报告吞吐加速比。逐样本结果和汇总分别写入
`results.log`、`summary.log`，每个物体/批大小的预测终态另存NPZ。

参考来源：[上游模型推进示例](https://github.com/asu-iris/Complementarity-Free-Dexterous-Manipulation/blob/main/examples/sim/run_comp_prediction.py)、
[上游位姿误差函数](https://github.com/asu-iris/Complementarity-Free-Dexterous-Manipulation/blob/main/utils/metrics.py)、
[ComFree 固定版本](https://github.com/asu-iris/comfree_warp/tree/ba8b9969f5f7859d51b1f738f458f8d2f49be080)、
[CoACD](https://github.com/SarahWeiii/CoACD)。上游推进示例仅打印时间，未定义本实验的预测误差。

## 物理与采样

地面上有目标物体和一个真实动力学小球，小球半径、质量分别为 0.01 m、0.01 kg。
基本体质量为 0.01 kg，球半径 0.03 m，立方体边长 0.06 m，圆柱半径 0.025 m、长度 0.06 m。
复杂物体复用 `test_0902.py` 对应 XML 的几何变换与完整编译惯性。
各接触统一 `condim=3`、摩擦系数 0.5，无扭转/滚动摩擦，重力为 9.81 m/s²。

外力幅值为 `{0, .05, .1, .2, .5}` N，作用于小球质心，在整个20 ms内恒定。
物体名义线速度级别为 `{0, .05, .2, .5}` m/s，角速度覆盖0–3 rad/s；
不同采样配方使用不同撞击方向、偏心位置和小球速度。
球面、方块的顶点/棱边/平面、侧卧/直立圆柱用于覆盖点/线/面接触。
几何类别与接触点数独立记录，不从几何名称推定实际接触数。

复杂物体先由参考引擎落定。持续接触的候选采用顶部加载，并在查询前落定；
部分滑动候选从该加载状态赋予两刚体相同平移速度，其余采用侧面作用。
建立接触候选混合顶部与侧面撞击。准备阶段不计入20 ms预测，也不参与误差。
每次实际初速度、名义速度级别、小球初速度和外力均保存。

CoACD 的阈值为真实尺度1 mm，seed=42、自动预处理，其余参数固定并保存。
每个分解组件单独重建凸包以修复绕序或近共面三角形；不会对整个物体退回单凸包。
组件属于同一刚体，并显式设置原物体质量、质心和惯性，显示网格不参与碰撞。
输出目录包含独立的模型与网格副本；回放校验其哈希。

## 标签与分组

标签只来自 MJWarp 的原生求解接触力。保存10次子步的求解起点接触，
再在20 ms终点执行一次不积分的 `forward`，共11个接触快照。
相对速度在各快照的求解位姿处计算，使用该时刻进入求解器的速度。
CPU `mj_forward` 只用于状态运动学和各模型自身计算，不替代 GPU 接触力。
原生摩擦棱锥的各行由原子操作独立分配，行地址未必连续。本入口逐行读取
`contact.efc_address`，再从对应 world 的 `efc.force` 解码；不沿用旧辅助函数的连续切片。
该修正仅作用于新增实验，不修改旧基准默认行为。

- `contact_establishment`：物体–球或物体–地面的加载接触从无到有。
- `separation`：加载接触消失；全程无加载接触另标 `free_flight=True`。
- `sticking` / `sliding`：区间无接触切换；有加载接触时，任一点切向速度超过
  `1e-3 m/s` 即为 sliding，否则为 sticking。
- 同一窗口既建立又失去接触为 `mixed_transition`，保留两种事件标记。

加载阈值为法向力 `1e-4 N`，事件定义在刚体交互对层面，
不把同一交互对上接触流形点身份的变化直接当作接触建立。
原始接触点的生灭及数量仍可从逐子步记录分析。

主表按 **t=0** 的目标物体全部有效几何接触点数分组：`0, 1, 2, >=3`。
有效点要求原生 `efc_address>=0` 且 `gap<=0`；不要求该点有正载荷。
不计小球–地面接触，不把摩擦棱锥行当作独立接触点，不对 CoACD 组件接缝点去重。
另存物体–地面/小球分项数、区间最大数及终点数。

每个物体、seed按**实测类别**填充配额。所有尝试均保留，超额查询 `selected=False`；
混合事件单列，`selected=True`。主表的四种状态只使用已选配额，混合事件使用所有混合样本。
最多对每种请求配方尝试 `samples_per_mode × max_attempt_multiplier` 次，缺口明确报告。
不能从 requested_mode 推断实际类别，也不保证每个类别覆盖全部力/速度组合。

## 指标与文件

比较目标物体20 ms终点的状态。位置误差为欧氏距离（mm），
旋转误差为归一化四元数的最短测地角（度），另存 `1-(qᵀq*)²`。
综合指标为 `sqrt((position_error_m/L)^2 + orientation_error_rad^2)`，
`L` 是固定的物体体坐标包围盒对角线。线/角速度误差在统一世界坐标计算。
这些是有对应刚体坐标系的姿态误差，不做球体/圆柱等几何对称性消除。
不同步长的积分误差也包含在内。例如自由落体采用半隐式 Euler 时，
20 ms一步与2 ms十步之间的位置差约为1.7658 mm，即使接触律完全没有作用。
回放显示相对保存状态的最大偏差；GPU浮点归约可能产生微小数值差异。

| 文件 | 内容 |
|---|---|
| `manifest.json` | 参数、依赖/上游版本、统计定义、完成状态与缺口 |
| `implementation/` | 实际运行及生成报告的源码快照，哈希记录在manifest |
| `<object>/model.xml`, `assets/`, `object.json` | 独立模型、网格、惯性和哈希 |
| `<object>/queries/*.npz` | 完整输入、参考11帧、三方法终点、样本元数据，无pickle |
| `<object>/samples.jsonl` | 所有尝试的实测/请求标签、配额选择与失败原因 |
| `<object>/reference_contacts.jsonl.gz` | 逐子步原生几何、力、速度、物体交互对 |
| `<object>/predictions.csv` | 各方法逐样本误差、分组、状态及失败原因 |
| `summary.csv/.md` | 物体×模式×起点接触数的误差统计 |
| `stratified_summary.csv` | 按外力、名义速度级别和seed分层 |
| `results.log` | JSONL逐样本绝对误差、相对误差、分母、分组及失败原因 |
| `summary.log` | JSONL逐组数量及绝对/相对误差的统计量 |
| `case_counts.csv/.md` | mode、mode×count、mode×geometry、三者联合的数量，含空格与少于100/200条标记 |
| `<object>/case_counts_seed_<seed>.json` | 每批落盘的该seed实测mode及联合分组的当前样本数 |
| `coverage.json` | 各物体、seed的配额、尝试数、实测类别计数和缺口 |
| `<object>/reference_batches.json` | GPU batch、拆分和串行重试的执行记录 |
| `smoke_test.json/.md` | 实测模式/接触数/接触几何覆盖及可选串并行验证 |
| `error_distributions.png/.pdf` | 已选查询的误差分布，不显示异常值散点 |
| `contact_count_heatmap.png/.pdf` | 按模式和起点接触数的平均综合误差 |

统计包含 mean、总体std、RMSE、median、p95、有效数和失败数；空组统计为null。
`object=all` 是按已选样本池化，不是物体宏平均。失败预测不填零，不参与误差均值，
但保留在 `n_total/n_failed` 中；参考失败单独标记，输入可生成时仍保存。
相对误差为无量纲比值（0.1代表10%）：位置、旋转及综合位姿误差除以MJWarp
从初态到20 ms终态的对应运动量，速度误差除以参考终态的世界坐标速度模长。
不用相对世界原点的位置/四元数作为位姿分母。分母低于1 µm、0.1 mrad、
综合位姿1e-4、线速度1e-4 m/s、角速度1e-3 rad/s时记为null/NA，保留实际分母。
这类有效预测的相对误差未定义，单列`n_undefined`；不当成零误差或预测失败。
每个物体×实测mode可设置采样配额，count与geometry分组报告实际数量；
空格的物理可实现性未经验证，不将空格直接称为“不可能”。
逐查询捕获原生 CUDA `printf` 告警：参考求解出现碰撞迭代告警时标记
`invalid_reference`，预测求解出现告警时标记该方法失败。准备阶段告警另存，
不自动否定之后独立重置且无告警的查询；不会修改上游碰撞算法或放宽判据。

## GPU 并行与额外 smoke test

`--reference-batch-size 1` 保持串行候选顺序与旧入口使用方式。大于1时使用
MJWarp 原生 `nworld`：同一物体的不同初始状态、速度和外力批量上传到 GPU，
10个时间子步仍按时间顺序执行。不同物体分别编译和组 batch，SCM 及 ComFree 的
预测和 CPU 接触诊断仍逐查询执行。因此纯物理吞吐量不能解释为整条流水线加速比。
batch 大小会影响候选配方填充顺序、浮点归约顺序及最终选中样本；使用相同 batch 配置比较实验。

若大 batch 出现无法归属具体 world 的原生 CCD 告警或缓冲区错误，则递归拆分重试；
小 batch 仍失败时逐查询串行重试，原生告警仍会使该次参考无效。
`reference_execution`、失败原因和 batch 日志保留真实路径，串行回退不算作 GPU 并行成功。
NPZ 记录实际 GPU batch 的同伴样本 ID；`replay` 自动重建同一批输入，报告实际执行路径
和相对保存状态的差值。GPU 接触排序与浮点归约可能使重放无法逐位相同。

额外 smoke 使用独立 seed=20261026，目标240条主查询。检查每个物体的四种实测模式、
总体 `0/1/2/>=3` 接触数及混合事件；不强求不可实现的物体×模式×接触数全组合。
基本体几何按原生地面接触位置的仿射维数确认，SVD阈值0.1 mm：球面点接触、
方块顶点/棱边/平面和圆柱侧卧/直立。复杂物体检查凸分量数量及实际参与接触的多个 geom。

串并行验证为每个物体选择16条包含不同模式、接触数和几何的已保存输入，
比较两个刚体全部11帧，要求模式和起点点数完全一致。数值验收容差为位置20 µm、
旋转1 mrad、线速度0.002 m/s、角速度0.05 rad/s；JSON同时保留更严格的
5 µm、0.1 mrad、0.0005 m/s、0.005 rad/s检查结果及未舍入的实际差值。
参考重放告警的查询单列失败原因、不计为匹配；每个物体至少一半查询及至少两个真实
GPU world 通过有效对照才算通过。纯物理计时预热后重复3次取median，出现告警则不报告加速比。

验证命令：

```bash
SCSP_COLLISION_GPU_TEST=1 /home/zz/miniconda3/envs/pi0/bin/python \
  -m pytest -q tests/test_collision_prediction.py
```

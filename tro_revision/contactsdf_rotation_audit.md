# ContactSDF rotation 核对记录（2026-10-22）

更新：按用户指定，TABLE IV 的 ContactSDF Planning Time 已改为上游日志求解计时：Rotation **5.30±1.37 ms**（15,000 次），Flipping **5.22±2.22 ms**（7,500 次）。均为 pooled mean ± sample SD，包含解提取，排除检测及求解前参数准备。用 `python tro_revision/update_contactsdf_solver_timing.py` 复现；逐案例日志哈希及统计位于 `output/diagnostics/contactsdf_solver_timing_20261022/statistics.json`。下文 637.80 ms 是先前填表采用的整周期统计，保留作审计依据；它不是当前表格单元格。

结论：核心 MPC 和接触动力学沿用了官方源码，但本地单指 Franka 实验并非原论文实验的完全复现。637.80 ms 的主因是几何检测，不能归因于 IPOPT 慢，也不能直接与论文的 Allegro MPC running frequency 对比。0/15 是当前适配与参数迁移协议下的结果，不能作为 ContactSDF 在原任务上的复现结论。

## TABLE IV 更新

文件：`tro_revision/experiment.tex`，`tab:all_data` 的 Rotation / ContactSDF 列。PDF 为 `tro_revision/root.pdf`，第 13 页。

| 指标 | 填入值 | 原始统计与口径 |
|---|---|---|
| Success Rate | 0.00 | 0/15；15 条有效完成，执行/校验错误为 0 |
| Execution Time (s) | -- | 仅统计成功案例，无成功样本 |
| Planning Time (ms) | 5.30 ± 1.37 | 15,000 次日志求解计时，样本标准差；整周期另为 637.80±473.67 ms |
| Obj. Pos. Error (m) | 0.06 ± 0.03 | 原值 0.0560975 ± 0.0256707，15 个终点 |
| Obj. Quat. Error | 0.64 ± 0.34 | 原值 0.6381634 ± 0.3378454，15 个终点 |

来源：`output/benchmarks/contactsdf_official_rotation_15_20261020/summary.json`、`results.csv` 及每个案例的 `trials.jsonl` 和 `run.log`。保留原六条案例并增加九条；原实验最多五个案例进程并发，耗时不是隔离环境下的单任务基准。表下注明单指适配、参数未经重训，以及上游日志求解计时口径。

## 官方仓库逐项核验

重新克隆 [asu-iris/ContactSDF](https://github.com/asu-iris/ContactSDF)，HEAD 为 `3a21edf8e51788f72a7d77259af4dd584d5ec43f`，与 vendored provenance 相同。14 个保留的上游文件均与远端 SHA256 一致；相关六个活动实现文件也与实验冻结快照一致。

| 项目 | 核验结果 |
|---|---|
| `dynamics/lse_dynamics.py` | 活动版本与远端逐字节相同。D-SDF 显式速度投影、摩擦锥四方向线性化、积分均继承上游；没有内层 QP/LCP 求解器。 |
| `planning/ball_mpc_lse.py` | 仅相对导入不同；CasADi `nlpsol(..., 'ipopt', ...)`、NLP、多重射击、初始化和 primal/dual warm start 均相同。 |
| Solver 设置 | horizon=4，IPOPT max_iter=100，动作边界 ±0.01，沿用默认容差/线性求解器设置；没有换成其他后端。 |
| Solver 二进制/依赖 | 本机核验环境 CasADi 3.6.7、MuJoCo 3.2.3；官方 README 推荐 3.6.5、3.1.6。历史日志没有完整二进制版本记录，不能证明历史运行的 IPOPT/线性代数库与作者完全相同。 |
| `ball_lse_contact_detection.py` | 相对导入、移除 simulator 类型标注、三指硬编码改为按实际指数量 reshape；其余算法未改。三指输出一致性测试通过。详见审计目录 `detector.diff`。 |
| Cost | 三指配置的目标值及导数与原代码数值一致；实际执行改为单指，已改变优化问题。没有增加 CSO/verify/approach/multistart。 |
| Theta | foam_brick 使用官方 foam 参数，其余五类物体使用 cube 参数；均未在这些 Franka 物体任务上学习。负的 mass_sqrt 项同样来自官方数组，源码平方后使用，并非负质量。 |
| 实际系统 | 原三球位置控制 / MuJoCo → 单点 Franka 阻抗控制 / Isaac；状态 16→10，控制 9→3。执行器、物体和任务已改变。 |

因此，准确表述是“使用官方检测、D-SDF 和 IPOPT MPC 的单指迁移实验”；不能表述为代码、solver 环境、物理系统及实验协议全部一致。

## 频率差异：原始日志与分段计时

从每条 `run.log` 提取 1,000 个 `mpc solving time`，共 15,000 个，与记录中的规划次数逐条核对：

| 项目 | 结果 |
|---|---:|
| 检测+规划整周期 | 637.802 ± 473.672 ms |
| 上游日志中的 MPC 求解计时 | 5.300 ± 1.366 ms |
| 整周期减去日志求解计时的均值 | 632.502 ms，约 99.17% |
| 整周期均值的倒数 | 1.568 Hz |
| 日志求解均值的倒数 | 188.67 Hz |

上游日志计时包括求解调用和解提取，不能称为纯 IPOPT 内核时间。上述倒数是计算吞吐估计，**不是**逐周期频率的均值，更不是实测闭环 wall-clock 频率。后者还包括仿真、执行器、读状态和记录等开销；`mean(1/t)` 也不等于 `1/mean(t)`。

原记录中共有 1 次非成功 MPC 状态 / 15,000 次规划；与“执行/校验错误为 0”是不同口径。不能用大面积 solver failure 解释 0/15。

检测复杂度来自本地 `mesh_pointcloud()`：将**每个三角面**的中心和法向全部写入 `_subsampled.txt`，没有实际降采样。官方三球发布文件实际是 cube 26、foambrick 56、stick 531 行；本地是 1,752–17,036 行。检测器对地面网格点逐个做 SDF 和梯度查询，导致大量重复平面计算。

另在同一固定初始状态下，单进程每类测 6 次、舍弃首次预热后取均值（仅诊断，不替代历史结果）：

| 物体 | 平面数 | 地面查询点 | 检测 ms | 规划 ms |
|---|---:|---:|---:|---:|
| elephant | 8,792 | 900 | 623.28 | 6.23 |
| foam_brick | 7,956 | 400 | 239.77 | 9.99 |
| mug | 7,314 | 900 | 474.44 | 5.32 |
| piggy_bank | 17,036 | 900 | 1,196.55 | 5.72 |
| stanford_bunny2 | 4,566 | 900 | 328.00 | 5.55 |
| teapot | 1,752 | 900 | 160.85 | 5.19 |

本机 CPU 为 i7-14700KF；论文使用 i9-13900K。诊断用单线程环境。硬件与原实验的并发差异仍影响时间，但分段计时已确认检测是主要瓶颈。

## 与论文和官方执行代码对照

论文版本：[arXiv:2408.09612v2](https://arxiv.org/html/2408.09612v2)，核验下载 PDF 与 HTML。

| 论文位置 | 数据/设置 | 对本地结果的含义 |
|---|---|---|
| Table II，三球 | cube / foambrick / stick 的 MPC solving cost 分别为 31.77±9.13 / 24.95±8.00 / 52.81±5.99 ms | 是求解耗时；不应把本地 637.80 ms 的检测+规划直接放在同一口径比较。单指 NLP 更小，5.30 ms 本身不矛盾。 |
| Table IV，Allegro 仿真 | 40.6±17.1 / 40.3±12.9 / 46.9±15.3 Hz | 是不同机器人、模型和几何查询协议；不等同于本地三球分支的单指适配。 |
| Table VI，Allegro 硬件 | CCW/CW 为 47.2±12.8 / 49.5±11.3 Hz | 不同硬件控制流水线；不能拿 h 的倒数替代。 |
| 模型步长 | 三球模型 h=0.1 s；硬件同样 h=0.1 s | 模型预测步长与 MPC 计算频率是两回事。 |

**本地执行节拍还与官方发布的三球代码不同**：官方 `frame_skip_=10`；直接加载官方 cube XML 得到 MuJoCo `model.opt.timestep=0.002`，因此每次 `stepPositionOnce` 推进 **0.02 s**。本地 `contactsdf_run.py` 使用 `frames=round(h/sim_dt)`，每次推进 **0.1 s**。官方发布代码本身就没有强制“控制间隔=h”；本地把两者绑定，是实质性的执行协议变化。模型的 h 相同不代表闭环执行节拍相同。

本地 quaternion error 是 `1-(q·q_target)^2`，没有单位；论文终点 orientation error 用 rad。不可直接比较数值。对每个终点单独换算 `angle=2 asin(sqrt(error))` 后，本地为约 **1.915±0.846 rad**；没有将均值误差直接进行非线性换算。

## 影响正确性和迁移表现的问题

1. **凸体假设不成立（mug）**：论文 C-SDF 使用凸体支撑平面交集。本地把凹表面所有面直接交给同一检测器；不是按各凸分块分别检测再组合。固定抽样的支撑平面检查中，mug 其它表面点落在部分平面外侧至少 **0.095118 m**。其它五类抽样未发现违例不等于严格证明全部平面有效。
2. **单指 closure 改变行为**：上游代码实际使用 `sum(v/(||v||²+1e-4))`，不是论文文字中的单位方向和。三指可以相互抵消，单指没有这种抵消机制；实际代价为 `0.1*r²/(r²+1e-4)²`，在 r>0.01 m 时会随距离增大而下降，可能鼓励远离物体。这是公式迁移的行为风险，不能称为只有输入维度变化、行为不变。
3. **参数未针对任务学习**：官方按物体任务学习 theta，本地跨物体和执行器迁移，无法保证预测模型与实际质量、接触摩擦及刚度一致。
4. **代码不等于论文公式逐项严格一致**：发布的 D-SDF 在外层用 `fmax(0,LSE)`、归一化含 `+1e-2`；论文写的是外层平滑 LSE 和理想行范数。检测器还保留上游的 half-SDF、只保留一个地面接触点、使用齐次坐标 1 变换法向（会引入物体平移项）等实现细节。本地没有修正这些，保证了代码继承，但不构成对论文每个公式严格一致的证明。

本轮只填表、核验和诊断，未改控制器、旧轨迹或原始统计。若要评价 ContactSDF 方法在新任务上的公平表现，应另开新批次：先保证凸几何接口有效、压缩冗余支撑平面、矢量化查询，再明确单指 cost、执行节拍及学习协议；新批次同时记录检测、NLP、完整规划与实际执行频率。不能用诊断计时覆盖旧实验的 637.80 ms。

## 验证与复现

- `python tro_revision/audit_contactsdf_rotation.py`：重新核对远端 checkout、14 个原始文件、冻结实现、15 条记录、15,000 次日志、汇总统计；`--apply` 更新表格。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SCSP_PLANNER_ONLY=1 .../envs/acc/bin/python -m pytest -q tests/test_contactsdf_standalone.py --tb=short`：**13 passed**。
- `.../envs/acc/bin/python output/diagnostics/contactsdf_audit_20261022/profile_detection.py`：固定状态分段计时与几何诊断。
- PDF 多遍编译完成，最终日志无未解析引用/引文；提取 PDF 文本核对了第 13 页 TABLE IV。
- 详细证据：`output/diagnostics/contactsdf_audit_20261022/verification_manifest.json`、`profile_detection.json`、`table_changes.json`、`detector.diff`、`compile.log`。审计范围只包括本次 ContactSDF rotation 与频率问题，未复核其它方法或整个论文。

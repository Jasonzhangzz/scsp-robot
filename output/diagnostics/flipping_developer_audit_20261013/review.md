# flipping 与远端 developer 的核对

结论：当前快照中未发现新增 rotation/flipping 分支引入的 flipping 行为回归。翻转难以完成的闭环原因仍未定位；本次检查不能证明成功率与远端相同。

远端已通过 `git fetch origin developer` 更新，基准为 `5aa024dbb018be9af677617f7001c36560fcaf87`。参考工作树为 `/tmp/ik2-flipping-audit-20261013/scsp-robot`。

## 逐项结果

| 检查项 | 结果与证据 |
|---|---|
| 命令分派 | 主入口 main 的 AST 相同；viewer、headless、异步/同步路径没有改变。 |
| 默认任务与参数 | 当前 `--flipping` 与远端默认 `ground-rotation` 解析出的所有参数相同。 |
| 初始/目标姿态 | 初始位姿、目标位置/四元数、standing_q、object_up_axis 全部逐元素一致。 |
| 初始指尖与随机数 | 指尖位置一致；参数构造后的 NumPy RNG 完整状态一致。新增旋转分支不会消耗 flipping 的额外随机数。 |
| 网格与几何 | 实际凸分解和原始网格配置均核对，碰撞顶点、法向、惯量与支持高度一致；mug/duck 使用同一份有哈希校验的凸分解。 |
| 物理参数 | dt、substeps、PhysX 求解器/迭代数、接触参数、质量、摩擦、惯量一致；模拟器类与接触缓存实现 AST 相同。 |
| MPC 状态/动作约束 | 完成同一物理与 Cartesian 适配后的所有参数一致，包含 Q、刚度、动作界和状态界。 |
| MPC 预测与代价 | 7 个物体分别评价预测、阶段代价、终端代价，每个函数 20 个样本，最大绝对差均为 0。未构造 native solver。 |
| CSO soft-SCM | 默认速度响应及其对自由速度的 Jacobian，在 50 个采样输入中最大绝对差均为 0。新增 return_impulses 默认 False。 |
| 接触选择与 CPO | test_0902 的唯一差异是 rs_decision_hook；普通运行为空。移除这个未启用块后的函数 AST 与远端完全相同。接近轨迹、执行坐标转换、接触测量文件相同。 |
| 双臂相关 MPC 修改 | 当前单臂 MPC 的 transports_contacts、residual_fn、terminal_residual_fn、custom_solver_profile 条件均为 False，因此这些新增分支不生效。Lambda 双臂模块不在这条单臂路径的调用链中。 |
| A1/A2/A3 forwarding | 当前 `--flipping` 与远端 `--task flipping` 产生相同的子进程参数。消融 MPC 和 contact_policy_ablation 文件与远端相同。 |
| 成功与阶段切换 | ObjectCostStages、位置/四元数误差函数相同；run_mpc_planner 移除未启用的 hook 赋值后 AST 相同。 |

## 数值覆盖

420 组参数对照，差异数 0。覆盖：

- 7 个物体，种子 0–4。
- 主入口 full、消融入口 full、消融入口 A3。
- 默认配置、固定倾斜轴、自定义 35–75 度倾角、原始网格。
- 每组比较 argparse Namespace、参数构造后的全部属性、RNG 状态，以及物理/Cartesian/rollout 适配后的全部属性。

## 当前与远端共有的完成条件

- 默认初始倾角为 90 度，启用随机倾斜轴。
- 默认使用异步规划；planning 回调与实时物理推进之间存在延迟。
- 进入 pose 阶段前，需要倾角 ≤ 10 度、线速度 ≤ 0.02 m/s、角速度 ≤ 0.25 rad/s，持续 0.3 s。
- 最终完整任务成功还要求位置误差 ≤ 0.02 m、四元数误差 `1 - (q·q_target)^2` ≤ 0.015，且处于 pose 阶段。
- `table-clearance` 为 0.012 m；这是远端已有值。旧的 FLIP_DIAGNOSIS.md 曾记录桌面参考裁剪对翻转的影响，但其历史配置和本次远端基准不同，不能据此直接认定当前失败原因。
- A3 取消 lift/place via 势场属于远端已有消融定义。

## 范围与限制

本次完成源码、参数、随机数、预测与代价的等价检查；没有运行新的成对闭环成功率实验，未测量异步调度差异。未修改生产控制参数或回退用户其他工作区修改。

详细数据：`audit.json`、`additional_checks.json`、`audit.log`。本次源码差异保存在 `source_diff.patch`。

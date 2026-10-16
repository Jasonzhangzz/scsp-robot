# 用 ADMM 交替求解 FA 的物体接触点与 hand 动作

新增的物体侧 ComFree 力/轨迹评分变体与可视化入口见 [FA_FORCE_CONTACT_ADMM.md](FA_FORCE_CONTACT_ADMM.md)。
本页描述保留纯 FA 全局目标的旧几何分裂对照。

当前实现为 `fa_contact_admm.py`。此前的“原 FA Adam + 附加接触一致性代价”实现已经停用，
源码与旧实验仅保存在 `outputs/fa_admm_20261016/source_superseded/`；其结果不能作为本实验结论。

## 目标函数

保留 FA 各项的公式、权重、单位和时间语义。唯一的目标参数变化是：
标记查询点项原先固定的物体侧模板点，现在成为优化变量 `c`。
原始 MANO 点只提供初值，不加入持续拉回 MANO 的额外代价。

```text
minimize F_FA(u,c) + I_surface(z)
subject to c = z
```

- `u`：整条预测窗口中的位置伺服控制节点。
- `c`：FA 查询点代价使用的物体局部接触目标，可在 ADMM 内层暂时离开表面。
- `z`：`c` 的表面可行副本，位于真实物体三角网格表面。
- `y`：`c=z` 的缩放对偶量。
- 手侧 body-local 标记点沿用原始 MANO→Wuji 对应，不再额外优化手侧局部点。

原 FA 的物体/手部跟踪、实际碰撞间隙闭合、穿透、自碰撞、控制平滑和状态平滑均保持原式。
实际碰撞点和接触力仍由 ComFree 计算；`c,z` 不直接注入接触力或替换实际碰撞输出。

`VariableContactObjective` 直接调用原 FA 的参考、物理闭合、穿透与自碰撞函数。
查询点项沿用同一公式与 sigmoid 权重的停止梯度行为，仅将固定目标数组换成各候选自己的 `c`。
固定为原模板时，数值与梯度检查必须匹配原 FA。

## 交替更新

每一轮采用以下多块、非精确 ADMM 更新：

1. 固定 hand 预测轨迹，求解
   `c <- argmin_c F_FA(u,c) + rho/2 * sum(mask*weight*||1000*(c-z+y)||²)`。
2. 固定 `c`，用独立的批量 L-BFGS 求解 hand 动作子问题 `min_u F_FA(u,c)`。
3. `z <- Project_surface(c+y)`。
4. `y <- y+c-z`。

物体点子问题是分离的：FA 中只有标记查询点项依赖 `c`，其余项对该子问题为常数。
它可沿“当前手侧查询点到 `z-y`”这条线化为一维径向问题。
实现使用 129 点全局扫描和 24 次局部黄金分割细化，直接比较原始 sigmoid 距离代价，
不把它替换为最近点吸附，也不加入力平衡 QP。

物体点步和 hand 步共同近似求解 ADMM 的 `(c,u)` 块。由于动力学、碰撞与表面非凸，
这里不假设有限次更新保证全局收敛。每轮的 `||c-z||` 单独记录。

执行前令 `c=z`，再做一次 hand 子问题优化，保证执行动作对应真实表面上的目标点。
最后这一可行化不等于 ADMM 已收敛，所以保存可行化前的 `c_before_surface_polish` 和残差。
最终选择多起点结果时只比较纯 `F_FA(u,z)`；rho 项、对偶项都不进入最终 FA 评分。

默认 4 轮 ADMM 和一次最终 hand 优化，把原配置的 40 次 hand 更新分成 5 组，每组 8 次。
L-BFGS 带独立的候选历史和 Armijo 搜索；实际前向/梯度评价次数另行记录，不能视作与 40 次 Adam 等计算量。
`rho=1` 只用于 `c=z` 的算法增广项，不是新增 FA 代价权重。
L-BFGS 使用数值坐标预条件：平移 0.001 m、腕部转角 0.01 rad、手指 0.05 rad 为单位尺度。
这是可逆的变量变换，不是代价权重、关节限速或控制步长上限。

## 原版 FA 求解循环已被替换

`FAContactADMMSolver.solve` 自己执行上述更新；hand 子问题不使用 Adam。
`EvaluationBackend` 只复用 FA 的初始化、ComFree 前向、原生伴随与结果打包。
其原版 `solve` 和 `_iteration` 被显式禁用。
验证脚本还把原版 `WindowOptimizer.solve` 与 `_adam_update` 替换为报错函数，确保不能静默回退。

## 输入与对照

`prepare_fa_admm.py` 保留原版 DexForge CA 手部参考，使用
`original_MANO_semantic_contacts.npz` 中完整 19 槽的原始接触掩码与锚点。
三个对照共用这一输入，全部关闭初始姿态偏移优化和 Lambda：

1. 固定 MANO 模板，原 FA Adam。
2. 固定 MANO 模板，独立 L-BFGS：分离求解器变化的影响。
3. MANO 初始化，ADMM 接触点 + L-BFGS hand 动作。

本轮三组均使用 8 个相同随机种子的起点，在 MuJoCo 与 ComFree 中分别闭环执行 5.4 s；
CA、模型、原始模板、hand 更新预算保持一致。代码默认仍读取 FA 配置的起点数，可用 `--multistarts` 明确指定。

此前采用 CA 后处理接触锚点的原版 DexForge 结果，只能作为另一个输入条件的参考，不能代替本轮固定模板基线。

## 验证

`validate_fa_contact_admm.py` 检查：固定同一接触点时原 FA 的值与 hand 梯度一致；
接触点梯度和一维近端子问题与独立计算一致；原版 FA/Adam 被禁用后仍可求解；
最终分数严格等于纯 FA 代价。生产 retarget 入口不受此实验改变。

```bash
python -m examples.retarget.validate_fa_contact_admm \
  --experiment outputs/fa_admm_solver_20261016 \
  --forge thirdparty/retarget_assets/DexForge

python -m examples.retarget.matrix_fa_worker \
  --experiment outputs/fa_admm_solver_20261016 \
  --forge thirdparty/retarget_assets/DexForge \
  --executor comfree --variant admm --case-name admm \
  --rho 1 --admm-outer 4 --multistarts 8 --duration 5.4
```

参考：[GraspADMM 原始说明](https://github.com/JYChen18/Dexonomy/blob/main/getting_started/3_ADMM.md)。
本实现沿用表面副本与对偶分裂思路，使用 FA 动态目标，不包含其静态抓取力平衡 QP。

## 本地执行（2026-10-16）

实现与输入均保存在 `/home/zz/scsp-robot`，不依赖远端临时目录。
本地使用 `pi0`、MuJoCo 3.11.0、Warp 1.17.0 和 RTX 5060 Ti。
远端根文件系统只有约 415 MiB 可用；远端中断结果只作归档，不能混入本地对照。

已完成的本地 ComFree 闭环测试均为 5.4 s、180 个窗口、8 个起点：

| 方法 | 全程平均位置误差 | 全程平均旋转误差 |
|---|---:|---:|
| 固定原始 MANO 模板，FA Adam | 7.875 mm | 3.229° |
| MANO 初始化，ADMM + L-BFGS | 2.622 mm | 0.554° |

以上使用同一完整 19 槽原始 MANO 模板，不能直接与此前采用 CA 后处理模板的 DexForge 数值比较。
固定模板 L-BFGS 与本地 MuJoCo 各组仍需完整跑完，再判断接触点自由度本身的收益。
阶段结果见 `outputs/fa_admm_solver_20261016/partial_report.md`；六组完成后生成 `report.md`。

本地复现 ADMM：在项目根目录、激活 `pi0` 后执行（实验目录需已由 `prepare_fa_admm.py` 准备）：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export COMFREE_WARP_ROOT="$PWD/thirdparty/retarget_assets/DexForge/forceaware/third_party/comfree_warp"
export HAND_ASSETS_ROOT="$PWD/thirdparty/retarget_assets/DexForge/assets"
python -m examples.retarget.matrix_fa_worker \
  --experiment outputs/fa_admm_solver_20261016 \
  --forge thirdparty/retarget_assets/DexForge \
  --executor comfree --variant admm --case-name admm_reproduce \
  --rho 1 --admm-outer 4 --multistarts 8 --duration 5.4
```

将 `--executor` 改为 `mujoco` 可在 MuJoCo 中执行相同的 ComFree 预测控制器。
使用新的 `--case-name` 保留已完成结果，避免将不同运行的逐窗口日志追加在一起。

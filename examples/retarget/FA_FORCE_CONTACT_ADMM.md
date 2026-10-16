# FA 手动作 + ComFree 接触点/力搜索

入口为 `run_fa_contact_mpc.py`，实现为 `fa_contact_admm.py` 和
`comfree_contact_search.py`。这是新增的 `admm-force` 变体，旧的几何 ADMM
及固定模板对照仍可独立运行。

入口现在默认使用 `--contact-solver batch-lm`：接触点/力子问题采用解析
ComFree Jacobian 的批量 LM，hand 仍是原 L-BFGS。每轮接触搜索实测约 43 ms，
完整 MPC 仍需 1.78–2.63 s。计时、验证与复现见
[FA_CONTACT_MPC_TIMING.md](FA_CONTACT_MPC_TIMING.md)。

## 代价与交替求解

手动作子问题完整保留 FA 代价公式、权重、ComFree 实际碰撞动力学和原生伴随。
新增的物体侧代价是：

```text
G(z) = min_lambda sum_k pose_cost(T_ComFree[k], T_demo[k])
                         + force_weight * sum_k,i ||f[k,i]||²

minimize F_FA(u,c) + object_weight * G(z)
subject to c=z, z on object surface
```

`c` 是 FA 查询点项使用的物体侧目标。ADMM 依次执行原 FA 查询点的 c 近端步、
完整 FA 的 hand L-BFGS 步、物体侧的采样/力优化 z 步、对偶更新。
z 步按 `object_weight*G(z) + rho/2*sum(mask*weight*||1000*(c-z+y)||²)` 选择候选。
执行前 c=z 并再优化 hand。多起点最终比较 `F_FA+object_weight*G`，不包含 rho 或对偶项。
**手动作 FA cost 没有改；整个算法的目标已经增加 G，不能再称为只改变原 FA 的求解器。**

物体轨迹使用与 FA 相同的物体位置、转角、参考和终端权重：
`1000²*reference_pose_scale*(obj_pos²*||p-pd||²+obj_rot²*angle(q,qd)²)`。
当前在 5 个 60 ms 节点评分，每节点权重为 6，末节点再加 terminal；
每个节点之间仍做 6 次真实 10 ms ComFree 积分。它是物体子问题的节点积分近似，
不修改手动作 FA 每个时间步的原始评分。

物体子模型从同一 SDF 场景生成，保留物体质量、惯量、armature、重力、地面和碰撞材质。
给定候选点的理想手部接触力通过其作用点产生力矩，地面反力由原生硬接触 ComFree 求解。
力采用四边摩擦棱锥的非负系数：`f=sum_r lambda_r*(n_in+mu*t_r)`，
`lambda>=0, sum_r lambda<=force_cap`，其中上限为每个槽的法向力上限（N）。

默认力解沿用 `planning/lambda_batched_lm.py` 的独立阻尼、信赖域、投影及
真实非线性代价接受规则。ComFree 原生伴随批量求各时间步的局部状态/合力
Jacobian，沿时间组合，再乘接触点到合力/力矩的解析映射；求解循环没有差分。
旧 `--contact-solver shooting` 保留中心差分、投影二次子问题及线搜索，供对照。
零力、上一窗口力、重力支撑和轨迹逆动力学初值都由同一个真实轨迹 cost 评价。
初值拟合不参与最终评分，也不作为可行性筛选。有限次采样和迭代是局部近似解，
不保证全局最优。

**规划力仅用于物体侧选点。实际执行器只接收 hand servo controls，绝不向物体施加这些理想力。**
实际接触点和力继续由手与物体的碰撞形成。

## 候选点与跨窗口状态

每次 z 步保留：上一轮接触点、原始 MANO 模板、向 c+y 的局部步、向当前实际碰撞点的局部步，
以及切平面上随机的小范围样本；全部投影到物体三角网格。
默认半径 5 mm。原 MANO 模板始终是候选，不因力闭合或力不足被删除，也不增加持续 MANO 拉回代价。
当前点附近可逐窗口继续移动，半径不是相对最初 MANO 点的永久边界。

当前实际碰撞几何：MuJoCo 执行时使用测得 q/v 下的原生接触；ComFree 执行时使用
预测第一步在测得 q/v 下的 ComFree 碰撞几何。它为局部样本提供初值，
不把理想接触点强行写进实际 hand 碰撞行。

每个窗口执行 30 ms 后，选中候选的 c/z/对偶量向前平移 3 个密集时间步；
力解按 60 ms 节点插值平移。相同 hand body-local 槽且掩码连续时保留上次解；
新激活的槽从 MANO 初始化。hand 控制仍沿用原 FA warm start。
未接入旧 LambdaOptimizer，也没有力闭合/接触力可行性淘汰门槛。

## 本地启动

在项目根目录执行。默认复用本地已准备的 Wuji/SugarBox 输入；可用 `--experiment` 指定其他已准备输入。

```bash
cd /home/zz/scsp-robot
conda run --no-capture-output -n pi0 python -m examples.retarget.run_fa_contact_mpc \
  --executor mujoco --viewer
```

默认 2 个起点、20 次 hand 更新、2 轮 ADMM、每轮 7 组点、2 次力优化、每槽法向力上限 40 N。
输出使用带时间的独立目录，不覆盖已有对照。启动时打印输出路径。
界面在独立进程读取每窗口原子快照，GLFW 与 CUDA 求解器不共享进程。
这是仿真等待求解的闭环可视化，未达到 30 ms 墙钟实时控制。

较快的检查配置（与功能测试预算相同）：

```bash
conda run --no-capture-output -n pi0 python -m examples.retarget.run_fa_contact_mpc \
  --executor mujoco --viewer --multistarts 1 --iterations 10 --admm-outer 1 \
  --contact-samples 5 --contact-force-iterations 1
```

蓝点：原 MANO 模板；绿点：优化目标点；灰点：邻域样本；红点：观测碰撞点；
紫色箭头：理想规划力，8 mm/N；半透明橙色手：MANO 示范动作；半透明青色物体：示范物体位姿。
MANO 默认开启，透明度 0.3；按仿真时间插值，保持段停留在示范最后一帧。
显示使用独立的非碰撞模型，不改变执行器状态、接触或控制。
鼠标可旋转、缩放视角；空格暂停，X 切换透明手/物体以查看遮挡的接触点。
实时模式在下一个窗口边界暂停。紫色箭头从外侧指向受力点。

回放已保存结果（不再求解）：

```bash
conda run --no-capture-output -n pi0 python -m examples.retarget.view_fa_admm \
  --case outputs/fa_admm_solver_20261016/mujoco/force_contact_mujoco_verified
```

尚未跑完时也可读取该目录已保存的 `partial.npz`；回放的是启动时读取的快照。
支持 `--speed`、`--no-loop`、`--video output.mp4` 和 `--check-image frame.png`。
可用 `--demo-alpha 0.5` 调整示范透明度，`--no-demo-overlay` 隐藏 MANO。
示范网格从实验目录的 `demonstration_visual.npz` 加载，准备输入时一并复制。
已打开的旧版窗口需重新启动可视化才能加载新显示代码。

## 验证

`validate_force_contact_admm.py` 在 0.9 s 的实际活跃接触状态检查：
FA 的值/hand 梯度与原版一致；禁用原版 Adam 后独立求解；物体和动力学参数一致；
力满足摩擦锥与上限；非线性 ComFree 评分不劣于初值；改变期望轨迹会改变力；
候选点在表面；跨窗口接触点与对偶量精确平移；不同批量的 CUDA graph 缓冲区可重复交错使用。
验证结果为 `outputs/fa_admm_solver_20261016/force_contact_validation.json`。

本地已完成的快速闭环配置为 1 起点、10 次 hand 更新、1 轮 ADMM、5 组样本、1 次力优化：

| 执行器 | 时长 | 全程平均位置/旋转误差 | 末帧位置/旋转误差 |
|---|---:|---:|---:|
| MuJoCo | 5.4 s | 6.795 mm / 2.477° | 14.549 mm / 4.828° |
| ComFree | 1.2 s（短程） | 1.280 mm / 0.722° | 2.910 mm / 0.390° |

MuJoCo 保持段的物体最低点始终高于地面 166.5 mm，没有掉落，但保持误差仍有漂移。
这不是与旧 8 起点/40 次更新结果的同预算性能对照；默认启动配置的预算更高。
汇总和源码校验值见 `outputs/fa_admm_solver_20261016/force_contact_verification.json`。

结果文件新增 `search_*` 数组，保存每组样本评分、选中候选、理想力、预测物体轨迹，
以及 `observed_points/body/mask`。目标点与实际碰撞点的距离、预测物体轨迹误差、
实际执行的物体轨迹误差是三个不同指标，需分别分析。

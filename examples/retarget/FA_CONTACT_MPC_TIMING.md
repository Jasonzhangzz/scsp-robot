# ComFree contact batch LM 与单次 MPC wall time

后续直接 acados OCP 试验见 [FA_DIRECT_MPC.md](FA_DIRECT_MPC.md)。该路径使用
真实 native ComFree 外部 dynamics 和解析导数，并实现低频、软偏移约束的接触
模板更新；属于独立实验，下面接触 LM 的同预算数据不包含该路径。

`run_fa_contact_mpc.py` 默认使用 `--contact-solver batch-lm`。
本次只替换接触点/力子问题；hand 的完整 FA 代价、原生 ComFree 预测、
L-BFGS 及其更新预算不变。旧接触求解器可用 `--contact-solver shooting` 显式选择。

## 同预算实测

RTX 5060 Ti，pi0，Warp 1.17.0，MuJoCo 3.11.0。
取已有 MuJoCo rollout 的 0.9、1.5、3.0 s 状态；每种方法在每个状态先预热一次，
再固定同一 q/v、warm hand controls、随机种子和 regularizer 状态重复三次。
重复之间清空 contact/force memory，测量具有活跃接触的同一输入。
两种方法顺序运行，最终配对测试时没有其他 CUDA 计算任务。

预算保持 2 hand 起点、20 次 hand 更新、2 轮 ADMM、每轮 7 组接触候选、
2 次力更新；即每轮批量优化 14 组接触集合。每个候选均预测 5 个 60 ms
节点，每节点仍有 6 个 10 ms 原生 ComFree 积分步。没有缩短预测窗口。

| 冻结状态 | 旧力求解 / 轮 | 解析 batch LM / 轮 | 新接触搜索总计 / 轮 | 旧完整 MPC | 新完整 MPC |
|---|---:|---:|---:|---:|---:|
| 0.9 s | 119.0 ms | 39.4 ms | 43.1 ms | 2779.3 ms | 2625.9 ms |
| 1.5 s | 118.4 ms | 38.6 ms | 43.1 ms | 1926.5 ms | 1781.4 ms |
| 3.0 s | 118.0 ms | 38.4 ms | 42.9 ms | 2141.7 ms | 2007.4 ms |

完整 MPC 是直接包住 `solver.solve()` 并在前后 CUDA 同步的 `perf_counter`
wall time，包含 prepare、两轮接触搜索、hand 求解、结果打包、contact/force
memory 更新与所有 CPU/GPU 传输。模型加载、JIT、首次 graph 构建在预热中；
首次调用耗时也保存在 JSON 的 `cold=true` 行。
完整入口还会记录日志与可视化快照，另计在 worker 返回的 `Plan.seconds` 内。

**接触子问题已经进入几十 ms，完整 MPC 尚未达到。** 每轮力求解约快 3 倍，
完整 MPC 仅快约 5.5–7.5%。保留的 hand 轨迹评价仍占 1.67–2.51 s：
在上述状态分别调用完整手—物体 ComFree 轨迹 99、45、57 次。
要让整次 MPC 达到几十 ms，还需要另行改变 hand 求解路径。

初次排查时另有一个 8 起点 L-BFGS 任务占用 GPU，测到约 5.6 s。
该任务随后结束；这个数字不能用来与最终 2.63 s 计算加速比。
早期 PCG / SciPy / OSQP 探索日志也不属于最终配对测试。

## 解析 Jacobian 与实现

参考 `planning/lambda_batched_lm.py` 的每候选独立阻尼、信赖域、投影与实际/预测
下降量接受规则，实现为 `comfree_contact_lm.py`，保留原摩擦棱锥与每槽 40 N
法向力上限。原 MANO、incumbent、consensus、观测接触和局部采样的规则不变。
原零力、warm、重力支撑、跟踪初值仍以真实非线性 ComFree 代价比较。

每个 ComFree 时间步的 13 维状态（qpos 7 + qvel 6）到下一状态的 Jacobian，
以及对六维物体局部 wrench 的 Jacobian，由 **ComFree 原生解析伴随** 提供。
将所有候选、时间步和 13 个输出基向量放入同一 batch，随后沿真实轨迹组合。
再乘闭式 quaternion geodesic residual 导数、接触点/法向到 wrench 的线性映射，
构造 LM 的 Gauss–Newton 系统。**LM 求解循环没有差分 Jacobian，也没有差分回退。**

局部 force→world force 的姿态依赖也求导。LM 的私有几何导数 tape 开启完整
hard-GS normal/frame 导数；不修改 FA hand 使用的导数 tape 或前向碰撞程序。
硬接触/最近特征选择本身只有分段导数，分支切换处没有光滑 Jacobian；
提议仍必须通过真实非线性 ComFree cost 才能被接受。

物理与 LM 大矩阵为 FP32，敏感的小角度 quaternion residual 使用 FP64。
支撑初值原来的两组 128 步投影循环融合为 Warp kernel，保持原拟合公式；
测试中与原 FP64 初值的最大系数差为 1.77e-5 N。
接触状态传输从逐时间步 `.numpy()` 改为打包传输，Z/Y 更新留在 GPU。
Warp 与 Torch 的线性代数使用同一 CUDA stream，缓存 graph 时保留所有分配所有者。

每候选的完整物体轨迹评价从 134 次降至 8 次；解析局部伴随的计算时间也包含在
上表中，不能将这句话解读为导数没有计算成本。

## 验证与局限

`validate_comfree_contact_lm.py` 验证 closed-form pose residual / Jacobian、
原生解析轨迹导数、GPU 初始化、重复 graph、摩擦锥、力上限、零接触、
真实目标响应及最终评分。独立数值方向探测仅用于离线验证，不参与求解。
自由飞行的方向导数误差为 0.11–0.52%；接触活跃单步为 1.13%，均使用 FP32
原生 forward。30 步落地轨迹的数值探测跨越 15–44 个硬接触分支，因此不将其
有限步差商称为该轨迹的光滑 Jacobian 验证。

`validate_force_contact_admm.py --contact-solver batch-lm` 集成检查通过：
FA 值相对差 1.14e-7、hand 梯度相对差 2.41e-18、最终组合评分相对差 1.83e-8；
320 个 warm contact 槽精确恢复；候选点位于表面、摩擦约束成立、不同 batch 的
CUDA graph 交错运行正常。实际执行仍只接收 hand controls。

三个冻结状态中，新物体侧 cost 分别为 189.88、0.433、0.378，旧解为
320.45、2.839、3.336；这些是有限预算局部解，不保证所有状态都优于旧方法。
这次没有进行完整新闭环轨迹的成功率/跟踪精度对照。

## 复现与结果

```bash
# 默认入口：解析 ComFree batch LM，hand L-BFGS 保持原预算
conda run --no-capture-output -n pi0 python -m examples.retarget.run_fa_contact_mpc

# 同状态 wall time；改为 shooting 可复测旧接触求解器
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n pi0 python -m examples.retarget.benchmark_fa_contact_mpc \
  --contact-solver batch-lm --dense-steps 90 150 300 --repeats 3 \
  --output outputs/fa_contact_timing/reproduce_batch_lm.json

conda run --no-capture-output -n pi0 python -m examples.retarget.validate_comfree_contact_lm
conda run --no-capture-output -n pi0 python -m examples.retarget.validate_force_contact_admm \
  --contact-solver batch-lm --output outputs/fa_contact_timing/integration_repeat
```

环境变量 `COMFREE_WARP_ROOT` / `HAND_ASSETS_ROOT` 沿用原 FA 配置；默认入口和
benchmark 会自动设置。独立集成验证可按 `FA_CONTACT_ADMM.md` 导出这些变量。

结果目录：`outputs/fa_contact_timing/`。最终比较为 `final_shooting.json` 和
`final_batch_lm.json`；导数/约束验证为 `validation.json`，集成验证为
`integration/force_contact_validation.json`。每窗口诊断新增 `wall_seconds`、
`prepare_seconds`、`hand_seconds`、各 ADMM phase 的 `force_seconds` 与
`contact_search_seconds`，以及 `force_jacobian=analytic_native_ComFree_adjoint`。

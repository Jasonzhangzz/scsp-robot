# 保留 Delassus 对角的单遍接触模型

2026-10-16。结论：可以跳出 softSCM 的响应形式，同时保留质量先验和现有 LM 接口。
已增加可选模型 `diagonal-contact`。它是标准对角预条件顺序接触更新的一次有限展开，不是新的物理定律，也不是训练后的参数模型。
默认仍保留旧模型，没有自动替换原有控制实验。

## 1. 先澄清 ComFree 的“滞后”

本仓库固定版本的 ComFree 在每一步重新计算 `v_free` 和反力，并把各接触贡献累加成广义力。
`atomic_add` 不是累积历史预测误差；该仿真路径也不在线学习 `k,d`。
它的局限是各行只看到同一个 `v_free`，没有利用本步其他接触已经产生的速度修正。
这属于本步接触解耦近似，不能等同于在线参数辨识滞后。

`models/scm_online.py` 才是另一条利用历史执行数据辨识参数的路径。新模型不走这条路径，也拒绝 `set_scm_parameters`。

## 2. 最小动力学改动

保留完整的广义逆质量矩阵 `M^-1`，只对接触空间使用对角有效质量：

```text
v = v_free
D_i = j_i M^-1 j_i.T + r
p_n = max(-(j_n v + max(phi,0)/h)/D_n, 0)
v += M^-1 j_n.T p_n
p_t1 = clip(-j_t1 v/D_t1, -mu*p_n, mu*p_n)
v += M^-1 j_t1.T p_t1
p_t2 = clip(-j_t2 v/D_t2, -mu*p_n, mu*p_n)
v += M^-1 j_t2.T p_t2
```

对检测到的接触点按固定顺序各做一遍。后续行使用当前预测中已修正的 `v`。
输入仍是已有的四条循环排列的接触棱 `n+mu*t1,n+mu*t2,n-mu*t1,n-mu*t2`，内部恢复法向和两个切向。

没有 `tau,zeta,k,d` 或从历史误差更新的系数。`r=1e-8` 是数值正则，`mu` 是场景的摩擦参数。
正间隙只有在当前步足以到达时才能传力；负间隙没有额外穿透恢复项。
这不是完整 LCP 求解，也没有收敛判据或自适应内循环。

“保留对角”在这里指保留对角有效质量/预条件器，**不是彻底删除一切接触间耦合信息**。
通过当前 `v` 传播反力，隐式利用了 `J M^-1 J.T` 的耦合，但不形成或反演完整 Delassus 矩阵。
如果严格要求只允许使用 `diag(W)`，连这样的反馈也不允许，则一般不能确定不同接触之间的正确反力分配：不同的 `W` 可以有相同对角。

### 质量先验的具体作用

对于一个无间隙的法向行，`p=-u/(D+r)` 恰是按有效质量消除闭合速度的单行响应，
不是凭一个全局系数决定反力大小。每条反力施加后再更新速度，可以避免重复接触各自消除同一闭合运动。

在冻结几何、物理质量正定、默认单遍且没有负间隙恢复时，相对于自由预测有

```text
T(v_after) <= T(v_free)
```

理由是每个标量修正都满足 `delta_T=p*u+0.5*D_physical*p²<=0`：
法向修正不超过停止当前闭合运动所需的冲量；切向截断区间包含零，且截断的是该标量动能的下降解。
这个结论不代表所有接触在单遍之后都满足非穿透，也不代表真实摩擦力分配已经精确。

## 3. 为什么兼容当前 LM

当前 `planning/scm_batched_lm.py` 的旧 softSCM 路径已经在每个候选力评估内展开 8 轮接触更新，
LM 通过有限差分计算残差 Jacobian。有限展开本身不是它的禁区。

新模型更简单：只有一遍，且在一次 LM 优化期间冻结 `M,J,phi,mu,D`。
当 `v_free=A*f+c` 时，映射仅由固定线性变换和有限次 `max/min` 组成，因此速度预测关于候选力是分段仿射的。
不会引入此前 `alpha(f)` 商式校正的额外输入相关分母，也没有需要对求解终止条件求导的内层优化器。
位姿积分和原有残差仍使用原实现。

具体核对：

- NumPy、CasADi、Torch 前向结果一致；解析/自动微分 Jacobian 与有限差分一致。
- 冻结几何后，各激活分支内的速度 Hessian 为零。
- 实际 GPU LM 对七个物体、两个 seed 共 14 个状态进行了验证，其中一组带非零初速度。
- 共 718 个候选点，20 次 LM 更新后每个候选的目标值均下降。
- GPU 与 CasADi 位姿/代价输出的误差保存在 `lm_validation.json`，没有把“有限返回值”当作优化成功的唯一标准。

相关模型与 LM 回归测试 39 项通过，预测比较目录测试 78 项通过（分开进程使用各自依赖环境）。

这些是当前三维候选力 LM 的兼容性证据，不是全局收敛或任意闭环工况必然成功的保证。
接触切换处仍有分段映射的非光滑性，单遍结果也依赖接触顺序。

### LM 内的物理质量与单位

原 LM 使用准动态接口 `v_free=Q^-1*f/h`，旧 `Q` 是阻抗度量。
仅仅对这个旧 `Q` 取 Delassus 对角，不能声称获得了真实惯性先验。

因此在 MuJoCo 理想力执行入口启用新选项时，设置

```text
Q_object = M_measured/h²
Q_object^-1*f/h = h*M_measured^-1*f
```

同时保留机械臂 Cartesian spring 块和既有 LM 残差接口，并带入当前测得的物体速度。
`M_measured` 来自当前配置的 MuJoCo 质量/惯性，包含一般质量耦合，不通过累计误差拟合。
这个单位转换只在新选项的直接力执行路径启用；其他接口若自行提供旧准动态 `Q`，其对角仍只是阻抗先验。

## 4. 实测误差与参考模型边界

### DART 方程参考

新 seed=73157，七个网格，h=2 ms，mu=.5；每物体每模式 32 初始样本，参考路径 rollout 每样本 10 窗口。
共 2912 共享状态。参考为 DART boxed contact 方程的两个 1 ms 子步。
每格为平均线速度误差 m/s / 平均角速度误差 rad/s。

| 工况 | ComFree | 单遍 diagonal-contact |
|---|---:|---:|
| 冲击，n=224 | .5863 / 3.1993 | **.1173 / 2.2314** |
| 滑动，n=224 | .0142 / .5030 | **.0059 / .2914** |
| 静止，n=224 | .0141 / .0537 | **.0033 / .0497** |
| 参考路径 rollout，n=2240 | .0268 / .3030 | **.0128 / .1890** |

不改任何增益，在另一个 seed=14593、h=4 ms、mu=.8 的 448 共享状态上，四种工况的汇总线速度和角速度误差也均低于 ComFree。
这不是每个物体、每个单独样本都更好的声明。

### MuJoCo Warp 参考

另用原 `compare_prediction.py` 在八形状上验证，共 384 共享状态，seed=94317。
ComFree 显式使用与上表相同的 `(k,d)=(.2,.001)`，重放相同输入。

| 工况 | ComFree | 单遍 diagonal-contact |
|---|---:|---:|
| 滑动，n=128 | **.0531 / .7018** | .0598 / .7315 |
| 静止，n=128 | .0156 / **.0245** | **.0034** / .0707 |

这里没有全面胜出。新模型采用 DART 式切向 box 和接近刚性的速度响应，ComFree/MuJoCo Warp 的棱锥、柔顺性和几何管线与之不同。
**DART 参考上的改进不能外推成对 MuJoCo Warp 或真实世界的普遍精度优势**，更不能把全部收益只归因于 Delassus 对角。
这次也没有验证自主长时 rollout 的轨迹误差或完整 MPC 闭环成功率。

## 5. 入口、代码和数据

- 核心 NumPy/CasADi/Torch 接触律：`models/diagonal_contact.py`。
- 原预测脚本：`--engines diagonal_contact`；此处是 CPU float64 准确性 backend，不是 GPU 吞吐基准。
- 现有 GPU LM：`--contact-model diagonal-contact --solver batched-lm`，预测核同样使用单遍更新。
- DART holdout：`output/diagnostics/diagonal_prior_contact/holdout/velocity_error_20261016_215714/`。
- 跨参数 holdout：`output/diagnostics/diagonal_prior_contact/parameter_holdout/`。
- MuJoCo Warp matched comparison：`output/diagnostics/diagonal_prior_contact/mjwarp_matched/`。
- LM 验证：`output/diagnostics/diagonal_prior_contact/lm_validation.json`、`validate_lm.py`。

```bash
conda run -n pi0 python examples/mpc/fingertips/test/benchmark_velocity_error.py \
  --samples 32 --rollout-steps 10 --seed 73157 \
  --output-dir output/diagnostics/diagonal_prior_contact/repeat

conda run -n pi0 python examples/mpc/fingertips/test/test_scm_mujoco.py \
  --obj foam_brick --rollout --contact-model diagonal-contact --solver batched-lm
```

该选项可供继续评估；当前证据支持“保留质量先验、兼容 LM、在 DART 基准上得到更小误差”，不支持“已经全面超越 ComFree”。

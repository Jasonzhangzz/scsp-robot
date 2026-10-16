# MANO → Wuji retarget

仅保留三条求解路径。根目录只放入口、输入准备、查看器和本说明；其余实现按用途分目录。

| 参数 | 保留版本 | 实现 |
| --- | --- | --- |
| `--solver fa`（默认） | 已验证成功的固定 MANO 模板 FA/L-BFGS | `fa/` |
| `--solver compiled` | 用生成方程与解析梯度替换 ComFree rollout | `compiled/`，复用 `fa/` 的优化循环 |
| `--solver direct-mpc` | ComFree 动力学直接进入 acados OCP | `direct/` |

## 运行

在项目根目录、包含所需依赖的 Python 环境中运行：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python -m examples.retarget.run_fa_contact_mpc --solver fa
python -m examples.retarget.run_fa_contact_mpc --solver compiled \
  --compiled-cache outputs/fa_compiled/cache
python -m examples.retarget.run_fa_contact_mpc --solver direct-mpc \
  --executor comfree --output outputs/fa_direct_new
```

FA 默认保持成功实验 `outputs/fa_admm_solver_20261016/comfree/lbfgs_fixed` 的配置：
8 个起点、40 次 L-BFGS 更新、5 个插值控制节点、固定原始 MANO 接触模板，
10 ms 预测步长、300 ms 预测窗口、30 ms 执行周期，初始速度为零，
关闭初始姿态偏移优化，ComFree 闭环执行。约 3 s/窗口是既有实验记录，未在当前机器复测。
compiled 沿用同一优化预算、warm start 和线搜索。

FA/compiled 可用 `--executor mujoco` 切换执行器、`--viewer` 开启可视化，
用 `--output-base DIR --case-name NAME` 指定新结果目录。
`--iterations`、`--multistarts` 可覆盖预算。每次运行使用新 case。
`--help` 显示所选版本的参数。

Direct 使用 30 个 10 ms 阶段、117 维状态和 26 维控制，
通过外部函数调用真实 ComFree，并使用原生解析 Jacobian。
其 Gauss–Newton 提案经过可行非线性 rollout 和回溯检查；完整模拟器没有编译进 acados C。
统一入口保持原有低频模板 ADMM。显式修正开关仍可使用：

```bash
python -m examples.retarget.run_fa_contact_mpc --solver direct-mpc \
  --executor comfree --post-step-tracking --relative-warm \
  --consistent-template-commit --output outputs/fa_direct_corrected_new
```

Direct 的控制参数化和预算与成功 FA 基线不同。
compiled 使用 FP64，native 使用 FP32，网格特征边界处的碰撞选择及导数可能不同。
这两版均保留源码，不将基线成功记录当作它们的闭环验证。

## 必需输入

默认 `--forge thirdparty/retarget_assets/DexForge`，
默认 `--experiment outputs/fa_admm_solver_20261016`，均可覆盖。
统一入口设置 `COMFREE_WARP_ROOT` 和 `HAND_ASSETS_ROOT` 的默认路径。

需要 NumPy、SciPy、MuJoCo、Warp、CasADi、PyYAML、trimesh、rtree、Torch；
Direct 另需 acados，compiled 另需 C/C++ 编译器。
实验目录必须包含准备好的输入及场景引用的模型文件：

```text
forceaware.yaml
sequence/
contactaware/
  config.json
  wuji_right_qpos.npy
  object_pose_7.npy
  contact_guidance.npz
demonstration_visual.npz           # MANO 可视化需要
```

从已有 CA 实验和原始 MANO 接触文件准备共用输入：

```bash
python -m examples.retarget.prepare_fa_admm \
  --experiment outputs/fa_admm_solver_20261016 \
  --baseline PATH_TO_PREPARED_CA_EXPERIMENT \
  --mano PATH_TO_ORIGINAL_MANO_CONTACTS.npz --mano-scene PATH_TO_SOURCE_SCENE.xml
```

旧 retarget/offline Lambda 流程、旧数据导入、诊断入口、SLSQP、condensed acados、
额外值缓存与断点续跑已删除。Direct 自身的 acados OCP 保留。

## 目录

```text
run_fa_contact_mpc.py              # 三种方法的统一入口
prepare_fa_admm.py                 # 共用输入准备
view_fa_admm.py                    # 结果查看器
fa/                               # 固定模板 FA/L-BFGS、原生伴随
compiled/                         # 方程、碰撞搜索、代价与解析伴随
direct/                          # Direct OCP、桥接、接触模板及预测记录
common/                           # 两种梯度方法共用的动力学、接触搜索与可视化
validation/                       # 保留实现的物理/梯度验证
```

通用接触几何在 `planning/wuji_contact_geometry.py`；规划代码不再依赖此示例目录。
新日志、轨迹、视频和编译产物放到 `outputs/` 或外部缓存目录，不放入本目录。

## 验证

```bash
python -m pytest -q tests/test_fa_compiled_spd.py \
  tests/test_wuji_retarget.py tests/test_wuji_online_execution.py

python -m examples.retarget.validation.fa --experiment EXPERIMENT --forge DEXFORGE
python -m examples.retarget.validation.compiled --windows 0 80 120 --solve
python -m examples.retarget.validation.direct --rollout-dir outputs/fa_direct_new
```

独立运行物理验证前需设置 `COMFREE_WARP_ROOT` 和 `HAND_ASSETS_ROOT`，并具有原实验输入。
compiled 验证还读取原成功基线 `comfree/lbfgs_fixed/rollout.npz`，
Direct 验证读取原 `mujoco/force_contact_mujoco_verified/rollout.npz` 的活跃接触状态。
`validation/contact_lm.py` 和 `validation/contact_admm.py` 验证 Direct 所需的接触子问题。
缺少资产时的单元测试不能替代完整闭环验证。

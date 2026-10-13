# DyWA 运行代码与接触失败审计

检查日期：2026-10-13。未修改算法源文件，未安装依赖或下载模型权重。

## 结论与证据边界

本仓库有 DyWA 学生策略运行入口 `examples/mpc/franka/ik2/test_dywa_isaac.py`，核心适配层为 `planning/dywa_interface.py`。当前实现启动 DyWA 自己的模拟器、感知和控制器；不是把学生策略直接放进 SCSP 的 OSC 模拟器。

当前机器无法执行完整策略：DyWA 第三方源码和权重不在预期位置，并且运行环境也缺少依赖。没有得到当前版本的接触轨迹，因此不能认定已复现用户记忆中的“不接触”。但静态检查和独立数值检查发现可复现的集成错误。

上游核对版本：`jiangranlv/DyWA` 的 `52eeee070c873f1c65ed62989ba4169551ab3106`。该版本不能代替原机器上可能存在的本地修改版本。

## 1. 当前运行阻塞：已复现

运行命令：

```bash
/home/zz/miniconda3/envs/acc/bin/python examples/mpc/franka/ik2/test_dywa_isaac.py --headless --trial_num 1 --max_rollout_length 1
```

首先在 Isaac Gym 导入阶段报 `ImportError: libpython3.8.so.1.0`，见 `startup.log`。添加 `/home/zz/miniconda3/envs/acc/lib` 到 `LD_LIBRARY_PATH` 后，正式入口能加载 gymtorch，但随后报 `ModuleNotFoundError: tensorboard`，见 `startup_with_library_path.log`。

另外已核实：

- `planning/dywa_interface.py:23` 固定查找 `thirdparty/DyWA`，该目录不存在。
- 默认权重为该目录下 `ckpts/last.ckpt`、`teacher-last.ckpt`、`env-last.ckpt` 和 ICP 编码器权重，当前均不存在。
- acc 环境没有 `pytorch3d`。`_install_pytorch3d_ops()` 本身先 `import pytorch3d`，所以替代函数不能解决完全未安装该包的情况。
- `ACC_PYTHON` 硬编码为旧机器的 `/home/lab423/miniconda3/envs/acc/bin/python`，当前不存在。入口因此不会自动切换到本机的 `/home/zz/miniconda3/envs/acc/bin/python`；以上实验显式指定了正确解释器。

正式入口会把子进程的 Python bin 加入 PATH。直接运行内部 `planning/dywa_interface.py` 时曾出现 Ninja 找不到的错误，但 acc/bin/ninja 实际存在，故没有将其列为正式入口的依赖缺失。

## 2. 初始状态和目标状态的桌面高度错配：高优先级、已数值验证

`examples/mpc/franka/ik2/params.py` 使用桌面高度 `0.35 m`。`planning/dywa_interface.py:140–142` 仅重置 XY，保留初始和目标 Z。Hydra 覆盖项没有设置 DyWA 桌面高度。

上游 [arm_div_base.yaml](https://github.com/jiangranlv/DyWA/blob/52eeee070c873f1c65ed62989ba4169551ab3106/dywa/src/data/cfg/env/arm_div_base.yaml) 设置 `table_dims=[0.4,0.5,0.4]`；继承的 [TableTopScene](https://github.com/jiangranlv/DyWA/blob/52eeee070c873f1c65ed62989ba4169551ab3106/dywa/src/env/scene/tabletop_scene.py) 默认 `table_pos=[0,0,0.2]`，桌面顶部为 `0.4 m`。

使用正式入口生成的 foam_brick、trial=0 参数，按其 WXYZ 四元数变换真实碰撞网格顶点后得到：

| 状态 | 物体几何底面 Z | 相对 DyWA 桌面的间隙 |
| --- | ---: | ---: |
| 初始 | 0.352 m | -0.048 m |
| 目标 | 0.350 m | -0.050 m |

如果原机器的自定义初始化/目标钩子直接应用这些参数，物体初始化时会明显穿入桌面；目标则要求物体底面位于桌面内部。目标方向下正常支撑的物体中心至少需要抬高约 5 cm，超过外部成功位置阈值 2 cm。因此这会破坏初始接触状态，并使正常桌面上的目标姿态不能同时满足位置判据；策略输入的绝对目标与目标点云也会对应错误高度。

建议保持预训练环境桌面配置，将初始 Z 和目标 Z 都加上 `DyWA_tabletop_z - source_tabletop_z`。同时在 reset 后读回实际物体姿态和目标，检查几何底面间隙与四元数转换，防止配置被忽略。

## 3. 依赖未保留的自定义 DyWA 修改：已核对上游接口

本地适配层要求这些配置字段：

```text
env.single_object_scene.mesh.mass
env.single_object_scene.init_obj_qpos_
env.task.target_p_
env.task.target_q_
env.franka.use_custom_end_effector
```

核对上游 `MeshObjectSet.Config`、`TableTopWithObjectScene.Config`、`PushTask.Config` 和 `Franka.Config`，上述自定义字段不存在。上游 mesh 质量按 `density * volume` 得到，机器人默认加载原 Franka 末端；本仓库也没有包含这些上游类的补丁。因此仅克隆官方源码并放入权重不能完整恢复当前适配层，还需找回或实现这些修改。结构化配置合并可能拒绝这些字段；即使某个加载路径容许额外字段，没有消费代码也不能改变初始化、目标或实际末端。

球形末端是否准确对齐训练时的 `tool` 坐标系、是否使用正确 hand body/Jacobian、碰撞是否开启，均需要恢复该自定义源码后才能核实。现阶段不能认定球形末端已经正确接入。

## 4. ICP 替代函数的旋转方向错误：已独立复现，默认接触失败因果关系未证实

`planning/dywa_interface.py:439–446` 计算 `H=X_centered.T @ Y_centered`，对其 SVD 后使用 `V @ U.T`，但后续按行向量应用 `X @ R`。这里需要 `U @ V.T`（加对应的反射修正）。当前实现返回逆旋转。

对 100 个标准差 3 cm 的合成点，仅旋转 5 度，当前函数单次更新得到约 4.23 mm 的对应点 RMSE；正确公式误差约为浮点精度，且实际函数返回矩阵等于真实旋转的逆。见 `numerical_checks.json`。

不过上游该几何 ICP 调用出现在 `AddTrackingReward` 中，其开关 `add_tracking_reward` 默认关闭。本地 `icp_ckpt` 指向的 ICP 网络编码器也不等同于此几何配准函数。因此该 bug 已确认，但不能据此认定默认学生策略“不接触”就是它造成的。

## 5. 动作时序与训练分布：需要区分已修复和待验证

上游训练配置使用 `dt=0.0125`、`action_period=8`，每次动作保持 `0.1 s`。当前 `build_experiment()` 已设为 `dt=0.002`、`action_period=50`，同样为 `0.1 s`。接口注释明确记录旧的 `20 ms` 保持时间会使机械臂跟不上每次目标，但当前版本已补偿此项，不能再把它当作当前仍存在的确定原因。入口 docstring 和模块级常量依然写着 `20 ms` / `10`，属于过期描述，不是实际生成参数。

当前共享参数把物体质量设为 `0.01 kg`，上游训练随机质量范围为 `0.1–0.5 kg`；另有球形工具替换和物理步长变化。这些属于训练分布变化，可能影响接触后的执行，不足以单独解释接触前的失败。

## 验证与后续诊断顺序

现有 `tests/test_dywa_interface.py` 的 4 个测试通过。其覆盖 XY 放置和相机矩阵，未覆盖桌面 Z、模型加载、末端接触和动作执行。

独立数值检查可重跑：

```bash
/home/zz/miniconda3/envs/acc/bin/python output/diagnostics/dywa_audit_20261013/check_geometry_and_icp.py
```

该脚本需要 spec.json 引用的 `/tmp/scsp_robot_canonical_meshes/...stl`，可先用上面的正式入口命令重新生成。它不会加载模拟器或模型权重。

优先恢复包含自定义末端、初始化和目标钩子的 DyWA 源码及权重，解决启动环境，再纠正桌面 Z。完成这些后，记录真实动作、末端控制点和球心、目标/实际跟踪误差、末端到物体的最小几何间隙、物体接触力及 done 原因。现有 `run_spec()` 只记录物体姿态误差，没有这些接触前诊断量，无法从它的输出区分“没有看到物体”“动作方向错误”“跟踪不足”和“球体碰撞未正确接入”。

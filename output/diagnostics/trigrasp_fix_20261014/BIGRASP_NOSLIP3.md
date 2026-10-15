BigRasp 新在线接触规则下的参数对照（2026-10-14）

执行模型默认已改为 impratio=10 / NoSlip=3。新增 `--contact-impratio` 和 `--contact-noslip-iterations`，可直接切换 3 / 10 / 0；NoSlip 和 impratio 是整个 MuJoCo 模型的求解设置，影响所有接触，不是单个物体材质的摩擦系数。默认指腹/物体和台面/物体滑动摩擦系数保持 5。

为分离启动静置的影响，配对实验均用 NoSlip=3 完成初始化，然后在开始控制前切换指定的执行 NoSlip。五组初始 qpos 完全一致。MPC 运行 1000 周期；FA/MPPI 使用 horizon=8、samples=64、iterations=2、CA iters=10，预算 160 周期。均使用 elephant、seed=0，单次结果不是成功率评估。

| MPC 配置 | 双边受力周期（Fn > 0.034335 N） | 最高抬升 | 最终抬升 | 稳定达到 6 cm |
| --- | ---: | ---: | ---: | --- |
| NoSlip=10，指腹 μ=5 | 837/1000 | 5.70 mm | 0.05 mm | 否 |
| NoSlip=3，指腹 μ=5 | 937/1000 | 6.28 mm | 2.24 mm | 否 |
| NoSlip=3，指腹 μ=8 | 887/1000 | 17.64 mm | -0.34 mm | 否 |

提高 μ 的对照只改变指腹材质，台面/物体配对 μ 始终为 5，Lambda、MPC 和 FA planner 使用的指腹配对系数同步更新。本次增加 μ 对短时抬升有帮助，但未形成稳定保持，且双边受力周期少于 μ=5 的 NoSlip=3 配置。NoSlip=3 比 10 有更多双边受力周期，但最终横向漂移反而更大；不能把它宣称为所有稳定性指标都改善。

FA/MPPI 的 NoSlip=10 完成 160 周期但没有稳定抬升；NoSlip=3 对照出现 native multistart 全部非有限值，重复实验也遇到数值失败，尚不能认定这个设置对 FA 更稳。有限预算及不同控制后端的结果不能混作统一成功率。错误记录在 mppi_ns3_mu5.json，日志在 /tmp/bigrasp_mppi_ns3_mu5_retry.log。

本轮另外修复了两个实现问题：

- 共享 MPC 的力约束写死 `u[:3]`，导致 BigRasp 的 6×6 刚度矩阵与 3 维向量相乘，启动直接报错。现在按刚度矩阵列数处理受力命令：双臂使用 6 维，单球工具的平移使用 3 维，旋转命令保留。
- 在线全局重抓的 Lambda 搜索没有可行候选时曾直接抛异常。现在保留当前 live patches、记录拒绝原因并遵守重试冷却，不重置真实状态，也不宣称假想接触已能承重。该行为有回归测试。

相关回归 66 passed。扩大到 IK2/shared-cartesian 的测试受当前环境缺少 open3d 限制，未完成，未修改或安装该依赖。

结论：按请求保留新的默认 impratio=10 / NoSlip=3；摩擦默认仍为 5。可以用 `--arm-friction 8 --object-friction 5` 复现实验性增强指腹摩擦，用 `--contact-noslip-iterations 10` 切回原执行设置。整体稳定抬升仍未验证通过，不能把剩余失败仅归因于摩擦不足。

完整机器可读对照见 bigrasp_contact_settings_comparison.json；复现实验使用 run_bigrasp_contact_settings.py，示例：

```bash
python3 output/diagnostics/trigrasp_fix_20261014/run_bigrasp_contact_settings.py \
  --experiment-noslip 3 --mpc --no-visualize --control-steps 1000 --hold-steps 0 \
  --mppi-device cpu --arm-friction 5 --object-friction 5 --contact-impratio 10 \
  --scene-output /tmp/bigrasp_contact_settings_20261014/mpc_ns3_mu5/scene.xml \
  --diagnostics-output output/diagnostics/trigrasp_fix_20261014/mpc_ns3_mu5.json
```

MuJoCo 文档依据：[soft friction 和 NoSlip](https://mujoco.readthedocs.io/en/stable/modeling.html)。提高 impratio/使用 NoSlip 可以抑制正则化摩擦漂移，但不是改变 Coulomb 系数，也不保证任意控制器达到零滑移。

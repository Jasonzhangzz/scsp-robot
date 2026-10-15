BigRasp 在线接触与力分配（2026-10-14）

Trigrasp 的 NoSlip=10 试验已回退到已验证的 3 次，impratio 保持 10。两组相同初始 IK / 物体目标的重复试验中，3 次两次均完成 20 cm 抬升及 20 s 保持；10 次两次均在第 284 步附近 lost_grasp。摘要为 noslip_comparison.json。这个单变量对照证明本控制器在 10 次配置下退化，但并不证明控制算法/执行模型无误。BigRasp 的执行设置保持原来的 impratio=10 / NoSlip=10。

已实现的在线路径：

- `--mpc`、`--mppi` 用 Lambda 提供初始接触猜测；随后每周期使用实际接触点、法向、力臂重新求解力，不再以冻结的离线接触点代替真实接触。
- 保留已加载手臂的全部真实接触，未接触手臂在附近 2 cm 搜索候选；有界 FK 与碰撞检查后，候选和现有真实接触共同做圆锥摩擦/载荷验证。候选只是未来目标，不计入当前承重能力。
- 真实接触已形成时，按当前质量、速度、物体目标和外部 wrench 重新分配带法向预载的接触力。承重 equality 优先于小的力正则项；未来预载不被当前较弱的法向力锁死，当前可用容量另用测得法向力验证。
- 在线 MPC 保留物体位置/姿态成本，跟踪合力与力矩，避免逐点切向力惩罚。有效分配建立后释放已接触指腹的法向位置误差，允许压力调节。RTI 控制执行前通过非线性 rollout 复核，接触分支跨越造成的坏步可回退到 warm control 或缩短步长。
- 确认失联/物体移动后可以更新全局接近猜测，同时保留仍加载的手臂，不重置仿真状态。日志为 `[local-contact]`、`[regrasp]`。
- 修复 native FA 加载 elliptic 执行 XML 时的报错：只在其独立的 planner XML 使用所支持的四射线内接近似，并把 GS 指腹/物体材料改为实际接触摩擦系数；执行模型的 elliptic / impratio / NoSlip 不变。

两点接触的限制：两个纯点力接触的 wrench 秩最多为 5，不能提供任意六维 wrench。诊断中的 `full_wrench_force_closure`、`force_closure_in_contact_span`、`load_feasible`、`task_load_feasible` 分别报告完整力闭合、可用子空间正跨度、完整静态承重、可控任务方向承重；不可控力矩保留为 `uncontrolled_task_wrench`。任务投影只移除不可控力矩，保持请求的合力不变，不把它冒充完整力闭合。额外真实的非共线 manifold 接触可能提供秩 6。

验证范围与尚未解决的问题：

单元/物理回归包括双点秩限制、真实多点满秩、就近替换保留已加载点、不使用远端虚拟点承重、未来预载与实际容量分离、真实 MuJoCo 接触替代错误的离线锚点、非线性坏步拒绝、执行 XML 与 FA 适配。MPC 原生 acados 路径和 FA CUDA 路径都已执行，明细在 bigrasp_live_summary.json。

完整 elephant 试验确实触发了就近接触替换和实时力分配，但尚未完成稳定的 6 cm 抬升，不能把这次接口与算法改动宣称为已解决整体抓取稳定性。MPC 运行中还观察到非指腹手部对物体的明显额外接触；例如第 650 步向下约 1.155 N，物体重力只有 0.0981 N。这些反力已计入当前力分配，但两点 Cartesian 预测并不能预测所有腕部/前臂碰撞的后续变化；当前日志也存在明显一步预测误差。该现象是本场景的具体执行/预测限制，不是关于所有 softSCM 模型误差大小的结论。

复现 MPC：

```bash
python3 examples/mpc/franka/bigrasp/bigrasp.py --mpc --no-visualize \
  --control-steps 1000 --hold-steps 0 --mppi-device cpu \
  --diagnostics-output output/diagnostics/trigrasp_fix_20261014/bigrasp_live_elephant_verified.json
```

FA 在本环境无 mink 时显式使用 `--ik-backend mujoco`。FA 160 周期验证采用 horizon=8、samples=64、iterations=2、CA iters=10；这是有限预算的机制验证，不是默认配置的成功率评估。

最终回归：126 passed、8 subtests passed。执行侧配置保持 Trigrasp impratio=10 / NoSlip=3，BigRasp impratio=10 / NoSlip=10。

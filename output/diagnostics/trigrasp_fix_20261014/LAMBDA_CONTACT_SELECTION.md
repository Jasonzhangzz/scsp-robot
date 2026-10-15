# BigRasp 接触预测与稳定抬升修复

## 当前结果与生效改动

恢复原执行接触仿真：pyramidal、condim=6、impratio=10、NoSlip=10。A/B 已证明旧算法仅删除扭转/滚动摩擦就从成功变成失败；但只恢复物理配置仍不能修复当前算法。

Lambda 不再将 K_osc × squeeze_depth = 1.8 N 当作真实压力目标，改为由外载荷和摩擦确定的压力下限；只惩罚压力不足。力正则作用于实际预测的接触力，而非 OSC 指令力。默认候选的压力 cost 占比从约 97% 降至约 0.19%。选点仍由耦合物体运动预测评分，不使用耳朵/躯体等部位规则。

MPC 与 Lambda 在已测得真实接触时使用压力观测修正法向 RHS，保留真实 gap；分离接触没有虚构预压力。扭转/滚动力矩与法向、切向力在同一迭代中求解，在线力闭合证书与力分配也纳入真实 condim=6 的接触力矩及实测承载上限。共享 SCM 默认 NumPy/Torch/CasADi 后端的兼容性通过 12 个组合检查；本次结果不是“所有 softSCM 都预测差”的证据。

执行端修复两个遗漏：确认双端实际承载后保持完整腕部世界姿态（包括工具轴自转）；home posture 项投影到完整六维末端任务的零空间，不再与腕部姿态控制竞争。抓取丢失后清除并重新建立姿态参考。保持原 orientation stiffness=15、物体 pose cost、目标与成功阈值。

`mpc_full_pose_nullspace.json`：原 elephant、--mpc、目标抬升 60 mm，step 1586 达到连续 1000 周期（10 s）成功判定，最终抬升 59.85 mm、水平漂移 1.19 mm、姿态误差约 3.29°。这包含完整机器人、真实 MuJoCo 接触及实际 OSC 执行。

控制与预测对齐后的中间版本曾通过 3.5 s，却未通过 10 s；单独增大姿态刚度到 60 也未改善。另一次将 MPC 的实时法向力参考跟踪改为单边压力不足 cost 出现压力瞬时不足和连续保持失败，已撤回；Lambda 的载荷压力下限修复保留。所有中间失败记录仍保存在 PRESSURE_WRENCH_FIX.json。

回归检查：164 项接触/选点/重规划/MPC 测试通过；最后的完整腕部控制与零空间专项 18 项通过。验证范围为默认 elephant 案例，不能据此保证所有物体和任意目标姿态均稳定。最终重复验证 mpc_full_pose_nullspace_repeat.json 同样在 step 1586 成功，最终物体位置与前次一致（1e-10 m 容差）。

## 以下为修复过程的历史记录

下文的失败结论、NoSlip=3 等配置描述属于对应中间版本，以以上当前设置和最终回放为准。

## 选点由预测 cost 决定

旧 Warp Lambda 用可瞬时施加的 Fn/Ft 来预测物体一小步运动。两侧力可以立即抵消力矩；normal 初值并不是必须建立的预压力。它没有模拟真实指尖的有限质量、OSC、压力建立和接触变化，且旧 orientation 权重很小。耳朵和躯体是否导致旋转取决于实际力与力矩，不能仅按部位判定。

删除本轮曾加入的重力力矩比例、截面配对、被动稳定高度和法线夹角质量硬筛选；auto/all 也不自动限制为 body-sides。凸包 COM 截面仅增加采样种子，不决定接受或拒绝。保留实际球形指尖支撑、显式区域限制、关节可达性和整臂避碰等执行约束。

新的 `BimanualLambdaPrediction` 使用在线 MPC 同一个 `BimanualRigidContactModel`：25 维状态、6 维 Cartesian OSC 指令、真实惯量/外力、有限质量指尖、圆形 Coulomb 摩擦、单侧压力上限、物体运动后运输的接触 patch、有限桌面支撑。全局候选以假设指尖即将接触的状态预测夹持压力建立过程；这不是机器人当前位置的下一帧。就近候选则从当前真实指尖位置、速度和间隙开始预测。两者都显式输出 prediction_initial_state、prediction_dt 和 rollout。

候选评分包括物体位置、旋转、速度、压力建立和指令正则；最佳若干候选继续优化 Cartesian 指令。旋转误差采用与 MPC 相同的旋转矩阵 Frobenius 范数（四元数向量误差的 8 倍）。优化器使用缓存 CasADi rollout 和精确 Jacobian，不在每个迭代中重复 Python 逐帧计算与有限差分。选中候选的 reported pose/forces/cost 来自耦合预测；旧 applied-force warm start 的 cost 明确单独标注。

## 控制循环不等待全局 Lambda

旧 `mpc_predictive_selection.json` 的最慢周期为 4.278 s，live contact 更新占 4.243 s，acados 仅 17.2 ms。同步全局评分和多轮 FK 在控制线程内执行，失败冷却又按仿真时间计算，造成反复卡顿。

全局重抓现为单个后台任务，使用隔离的 MjData、Warp/LM 缓冲与 predictor 状态；控制线程只提交和读取已完成任务。失败重试按墙钟时间从 0.25 s 退避到最多 4 s，同一时间不排队多个搜索。后台最多检查 128 个 cost 排序候选，启动仍最多 32 个；避免每次只重试相同的前 32 个失败点。返回结果需检查物体平移/旋转和目标是否已过时。acados 每个周期继续运行。

入口在 numerical libraries 导入前配置 CPU 线程，默认 1（尊重已有环境配置），可用 --compute-threads 覆盖。日志分别显示 plan/live/loop 耗时。BigRasp 编译后自动清理中间文件；本次另清除 325.4 MiB 已完成缓存的中间文件，保留 .so、JSON 和日志。

## 不可行接触的恢复

数学证书区分真实可用压力与接触几何在允许加压后的可行性。短时压力不足不会立即退开；连续不可行或确认滑脱后进入：解除法向预压力并退出 8 mm → 抬离物体 → 等待可执行的新 Lambda 点 → 横移到其上方 → 下降接近 → squeeze。单根仍可用的实际接触优先保留；若固定它连续两次找不到可执行新组合，则释放两根重新搜索。新点只有真实接触后才贡献可用压力。

退出/抬离时保持腕部当前方向，避免在物体旁边直接转腕。恢复 waypoints 固定在世界坐标中，不跟随预测中的物体位移/旋转；真实 phi/J 仍使用测得的接触。新 squeeze 阶段有 0.3 s 建立接触窗口，防止刚开始闭合就再次触发重抓。物体最终 target pose cost 始终保留。

## 验证与限制

相关自动测试 146 项通过，包含后台延迟/退避、Warp 并发缓冲隔离、不可行双接触释放、恢复动作顺序、固定世界 waypoint、真实接触不被虚拟 waypoint 替换、预测/Jacobian、Trigrasp 回归与力分配。

`mpc_async_release_recovery.json`（600 周期）中 acados 中位 15.6 ms、P99 31.5 ms；周期中位 45.8 ms、最大 117.5 ms。出现 no accepted new guess 的周期总耗时约 32.7–43.1 ms。恢复实际完成 release → retreat → waiting，物体仍在桌面附近，但完整抓取 success=False。此前世界坐标 waypoint 修复后的 `mpc_async_world_recovery.json` 一次抬升至 74.2 mm，但未满足稳定成功条件，也不能宣称已稳定抓起。

两个摩擦点通常只能形成五维 wrench 子空间，不能宣称完整六维力闭合。隔离球形伺服实验也发现耳朵—躯体点在力正确实现时可以稳定抬升，因此不以物体部位硬拒绝候选。完整机器人 MPC 仍有接触保持/杆件碰撞问题；这些结果不证明完整稳定抓取已经解决。impratio=10、noslip=3、默认摩擦系数 5 保留。

补充最终回放：`mpc_async_recovery_separation.json`（700 周期）实际在 step 299/300/387/394 依次进入 release/retreat/waiting/approach，step 394 接受后台生成的接触点。acados 中位 15.3 ms、周期最大 127.5 ms；出现失败的周期 46.8 ms。`mpc_async_recovery_verified.json`（900 周期，128 个机器人候选检查）还在 step 586 接受新点，后台搜索耗时约 2.4 s 而控制持续运行；失败周期 32.9–47.7 ms。两次完整抓取仍 success=False，后续接触保持仍需解决。汇总见 `REGRASP_TIMING.json`。

## 接触仿真 A/B 补充与恢复

旧版 HEAD 1de9a8d 的 BigRasp 和在线 MPC 在同一 elephant、--mpc、800 周期下重新执行成功：抬升 64.3 mm、漂移 2.59 mm。旧仿真原本就是 impratio=10、pyramidal、condim=6、NoSlip=10。旧算法只改物体 condim=3 就失败，最终约 -1.1 mm；因此移除扭转/滚动摩擦确实是回归因素。旧算法换当前新仿真（elliptic/condim3/NoSlip3）也不稳定。

按用户要求恢复默认执行仿真为 pyramidal、condim=6、impratio=10、NoSlip=10，并允许显式 --contact-friction-cone/--object-contact-dim/--contact-noslip-iterations 覆盖。point-force 规划近似不再用于删除真实物体接触的扭转与滚动阻力。

当前算法配恢复后的旧仿真：双端接触 712/800 周期、最终抬升约 5.8 mm、漂移 6.59 mm，仍 success=False。物理回归和算法回归同时存在，不能只恢复物理就宣称抓取修复。详见 CONTACT_PHYSICS_AB.json。

压力专项审计：重力由完整自由关节 qfrc_bias 传入，10 g 物体 -0.0981 N，未漏掉。当前全局 Lambda 将 OSC K=600 N/m 与 squeeze_depth=3 mm 的乘积 1.8 N 直接当作两端压力目标；这不是标定得到的真实接触刚度。在最新候选中，压力 cost 594.09/总 cost 611.51，约 97.15%。它采用 hypothetical touch、含桌面支撑的 80 ms 预测，也不能视为已验证离桌后的两端承载。

pressure_audit.py 使用相同 10 g 物体、2.5 kg 有效指尖质量、1.8 N 双端 OSC 法向输入和真实 solref/solimp。初始穿透 0.3 mm：MuJoCo Fn=0.1272 N/端，当前参数下 SCM Fn=0.002687 N/端；3 mm：0.7917 N 对 0.01343 N。这说明当前双端压力的参数标定/模型与真实仿真不一致，不是证明所有 soft-SCM 场景预测都很差。数据 pressure_audit.json。尚未据此盲目调整共享 SCM 或 IK2 参数。

# batch_replay 启动及性能前后对照

机器：8.147.67.185:1016。当前用户批处理 PID 1724714，未中断。
对照基线：仓库 `.cache/glove_fix_backup_20261013/` 中修改前源码。
输入：10 月 8 日 17:19:04 的同一条轨迹，807 个原生 EMF 帧、399 个相机帧。

## 代码差异及隔离测量

逐项对照了 emf_offline、sdk_retarget、official_retarget、wuji_replay 和 calibrated/batch_replay。
之前的人工 FK 初始化已撤回。新增算法工作为 RX 朝向转换、标定 world 骨架重建/传递及 source 发布替换；没有改动视频 ramp、视频临时文件改名、清理、最终目录提交或权限修改的代码。

同机同环境，OMP/OpenBLAS/MKL 均 8 线程，使用同一输入、同一 cqq_v2 模型和本地字节码缓存；通过 cProfile 测量：

| 范围 | 修改前 | 当前 |
|---|---:|---:|
| 原生左右手 EMF → SDK hand IK → RetargetSession | 2.299 s | 2.044 s |
| 相机帧匹配 | 0.908 s | 0.901 s |

进一步把转换后的同一 EMF 提供给修改前的 solve_native：旧代码 2.101 s。其骨架和 q 与当前代码逐点相同，隔离了不同手势解算分支的影响。

新增部分在当前 native profile 中：RX 转换两侧合计约 0.009 s，glove FK 约 0.008 s，recorded_world_from_wrist 约 0.015 s。不能解释 154 秒的生产阶段墙钟时间。

另在独立进程、相同 GPU/8 线程配置下，完整 calibrated.wuji_replay 单次隔离测量为旧版 5.982 s、当前 8.557 s；CPU 时间为 6.639 s、7.113 s。此测量包含解析、对齐、导入和 I/O，且原版与当前的首次模块加载顺序不同，不能把墙钟差值全部归为算法增加。手臂 IK 分别 0.144/0.278 s。没有完整 RGB 视频生成或 CPFS 标准输出目录提交。

## 生产任务实际延迟

首批三条轨迹端到端均约 604 秒，13:56:43 已完成。第一条日志：

- trajectory worker 291.336 s，内部 total trajectory solve 225.919 s；其差约 65 秒为入口调用、首次依赖导入等开销。
- 标为 raw EMF SDK finger retarget 的阶段 154.339 s，包含首次模块加载、读取、处理及后续目标构造，不只是 SDK compute/step 的数值计算。
- arm IK trajectory 0.352 s，399/399 成功。
- 视频 ramp 116.502 s；清理源暂存文件 54.691 s；数值写出 2.361 s。
- 最终 CPFS 目录 rename、stat、iterdir/chmod 又产生长时间等待。

从 /proc 的 D 状态线程及正在执行的系统调用，直接捕获到 CPFS 输出路径的 rename 阻塞；之后还捕获到多个视频线程在同一 CPFS staging 内重命名 remux/ramp MP4。py-spy 也捕获 _chmod_tree_world_writable→iterdir 等待。线程栈及真实文件路径表明这些阶段正在等待文件系统，而不是 GPU IK。

第二批轨迹仍约 397 秒；其 EMF 阶段降至 8–10 秒，视频 ramp 约 128 秒、清理约 61 秒。说明冷启动开销消退后，视频和 CPFS 元数据操作仍是主要问题。

终端进度在整条轨迹完成后更新，详细日志写入每条 episode 的 batch_replay.log，因此初期容易看起来没有启动。

## 本次源码更新导致的冷启动遗漏及修正

更新使 CPFS 上的 .pyc 失效：official_retarget、wuji_replay、sdk_retarget、emf_offline、glove_ik、emf_orientation 的新字节码依次于 13:46–13:52 生成。结合首次导入等待，缓存重建是这次更新需要处理的冷启动问题。仅凭文件时间无法精确分摊全部 154 秒；一个后续微型 CPFS 缓存写入探针仅 0.004 秒，也说明分布式文件系统延迟有波动，不能笼统断言每次写入都很慢。

此前正确性验证显式设置了 /tmp 缓存，遗漏了默认命令的冷启动性能。已在 batch_replay.py 的重依赖导入前自动设置本地 Python 字节码缓存，并传给子进程。当前默认目录为 `/tmp/ego_fast_retarget_pycache_2207`；显式 PYTHONPYCACHEPREFIX 和 Python -B 仍被尊重。手部求解、模型和输出格式保持本次已验证的修复行为。

验证：默认解释器正常启动（未使用 -S 或手动设置 PYTHONPYCACHEPREFIX），`batch_replay.py --help` 1.675 秒。入口模块加载 0.419 秒，新增三个 helper 首次加载 4.413 秒；确认其 .pyc 位于 /tmp，子进程环境继承该设置。这里只验证入口及新增 helper 导入，不把 --help 的时间当成整套 batch 耗时。

备份：`.cache/glove_fix_rx_revision_20261013/batch_before_startup_fix.py`。

下一次运行原命令自动生效。运行中的批处理已加载旧入口，不会通过修改源码自动重新启动。

仍未迁移视频临时目录：目前视频 remux、ramp、临时文件及 staging 都直接放在 CPFS。要消除这部分持续等待，应在本地盘处理临时视频和 staging，仅将最终成品提交至 CPFS。这是独立于手部算法的后续优化；本次没有调整其发布路径或视频生成语义。

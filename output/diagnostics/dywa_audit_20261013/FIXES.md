# 本仓库代码修复验证

日期：2026-10-13。原 DyWA 自定义源码及权重位于另一台机器，本次只修改本仓库适配代码。

已修复：

- 物体初始与目标 Z 同时按桌面高度差平移；Hydra 配置同步指定桌面位置、尺寸。
- 物体半径计入 XY 可用区域；原任务位移放不下时明确报错，避免缩小任务。
- ICP 采用行向量约定 `X @ R + T`，纠正逆旋转，返回每个 batch 的 RMSE 和实际收敛状态。
- 自动寻找本机 acc Python，支持 `DYWA_PYTHON`、`DYWA_ROOT`，启动前设置库搜索路径。
- 在导入 GPU 扩展前报告全部缺失源码和权重；检查基础依赖及自定义配置接口。
- 学生权重严格加载；reset 后核对实际物体位姿和目标，捕捉高度、四元数顺序和参数未应用问题。
- 用评估 reset 替代训练包装器的随机 `mixed_reset`，避免计分前执行 timeout 长度的随机动作。
- 新增 `--dywa-dry-run` 和 `examples/mpc/franka/ik2/README_dywa.md`。

验证结果：

1. acc 环境运行 `tests/test_dywa_interface.py`：13 passed。覆盖无随机动作的 reset、高度映射、位移保留、ICP 数值、位姿核对、路径及缺失文件处理。
2. 从普通 `python` 运行正式入口的 dry-run 成功，自动进入本机 acc 环境，输出映射后 spec。
3. 按实际网格顶点验证 foam_brick 和 elephant 默认配置：初始底面均为 0.402 m，目标底面为 0.400 m，动作保持为 0.1 s。
4. rubber_duck 和 mug 默认碰撞分解因本机 acc 缺少 CoACD 无法生成。改用明确指定的 `--collision-geometry hull` 后，七种物体 foam_brick、elephant、rubber_duck、stanford_bunny2、mug、teapot、piggy_bank 的初始底面均为 0.402 m、目标底面均为 0.400 m。此项只验证凸包配置的几何映射，不代表已验证默认凹形碰撞执行。
5. 正式入口现在在启动模拟器前明确列出缺失的 DyWA 源码和四份权重，没有进入策略运行。
6. Python 编译检查通过；按照原文件 CRLF 换行约定执行 diff 空白检查通过。

没有完整接触或任务成功率结果。拿回自定义 DyWA 源码和权重后，仍需在其实际控制器和球形末端上进行完整 rollout 验证。

原 `REPORT.md`、`spec.json` 和 `numerical_checks.json` 保留修复前诊断证据。

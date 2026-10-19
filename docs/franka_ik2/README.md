# Franka IK2 MPC

`examples/mpc/franka/ik2/` 只保留一个公共入口：`test_mpc_isaac.py`。
入口负责选择物理后端和运行模式，且在导入时不初始化 Gym、Torch 或规划器，
供 multiprocessing 的 spawn 子进程安全加载。

从仓库根目录运行：

```bash
# Isaac，固定 20 ms 同步控制
python examples/mpc/franka/ik2/test_mpc_isaac.py --headless --sync-planner --obj foam_brick --trial_num 1

# Isaac，异步规划；--viewer 可换成 --headless
python examples/mpc/franka/ik2/test_mpc_isaac.py --viewer --async-planner --obj foam_brick --trial_num 1

# MJWarp，固定 20 ms 同步控制
python examples/mpc/franka/ik2/test_mpc_isaac.py --physics-backend mjwarp --headless --sync-planner --obj foam_brick --trial_num 1
```

原有参数和默认异步调度不变。Isaac 需在安装了 Isaac Gym 的环境运行；MJWarp
需安装 MuJoCo、Warp 和 mujoco_warp。详细运行说明见 [HEADLESS_ROLLOUT.md](HEADLESS_ROLLOUT.md)。

## 文件整理

原目录同时承担了公共入口、实现模块、独立实验、文档和旧路径兼容五种职责，
还保留了 12 个软链接。同步和异步并不需要各自一个启动目录，使用
`--sync-planner` / `--async-planner` 即可。

| 原文件 | 当前位置 / 处理方式 |
| --- | --- |
| `test_mpc_isaac.py` 的 MPC、OSC 实现 | `examples/mpc/franka/runtime/mpc.py` |
| `params.py` | `examples/mpc/franka/runtime/params.py` |
| `contact_frames.py`、`physx_contact.py`、`scene_friction.py` | 合并为 `examples/mpc/franka/runtime/contact.py`，不导入物理引擎 |
| `test_mppi_isaac.py` 的公共仿真器和观测载荷 | `examples/mpc/franka/runtime/simulator.py` |
| `test_mppi_isaac.py` 的 MPPI 规划和入口 | `examples/mpc/franka/experiments/ik2/test_mppi_isaac.py` |
| `isaac_bus.py`、`sync/isaac.py` | `examples/mpc/franka/runtime/bus.py`、`isaac.py` |
| `headless_rollout.py`、`sync/mjwarp.py`、`async/mjwarp.py` | `examples/mpc/franka/runtime/rollout.py`、`mjwarp.py`、`async_mjwarp.py` |
| `summarize_rollouts.py` | `examples/mpc/franka/runtime/rollout_statistics.py` |
| `test_rs.py`、`export_*.py` | `examples/mpc/franka/experiments/ik2/` |
| `*_ablation.py` 兼容软链接 | 删除；直接使用 `examples/mpc/franka/experiments/ik2/` 的实际文件 |
| `assets/isaac_franka_dynamics.json` | `examples/mpc/franka/runtime/assets/` |
| `*.md` | `docs/franka_ik2/` |
| `sync/run.py`、`async/run.py` | 删除；统一入口加对应的调度参数 |

公共实现仍然有必要：进程总线需要避免导入 Gym，规划器子进程需要避免初始化
物理引擎，而接触几何也供其他 Franka 示例使用。因此保留按职责划分的模块，
不把数千行实现堆入入口。仓库内调用方和测试已改用新模块路径；旧的 Python
导入路径不再提供兼容软链接。外部脚本也需要按上表更新。

## 整理验证（2026-10-19）

- 受影响的 33 个测试模块：570 项通过、2 项跳过；另有 5 项既有失败单独核对。
- 这 5 项用整理前的 IK2 代码备份均可复现：杯子碰撞几何 1 项、direct-contact
  旧基线 2 项、缺失 EvoTorch/Numba 等依赖 2 项；未修改其物理逻辑或安装依赖。
- 仓库 smoke 检查通过；语法和本次变更的空白检查通过。
- Isaac 与 MJWarp 各完成同步、异步两种实际 headless 短回合：foam_brick、
  1 trial、2 个控制步、acados；四组均正常汇总，求解器回退为 0。
  这验证启动、进程通信和资源路径，不代表完整任务成功率。
- MPC/OSC 的 39 个函数与类定义经 AST 比较，仅模块导入路径改变。
- 原有 params.py 和碰撞导出脚本的未提交修改均已保留。

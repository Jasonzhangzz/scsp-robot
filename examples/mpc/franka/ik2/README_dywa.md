# DyWA Isaac 运行入口

入口为 `examples/mpc/franka/experiments/ik2/test_dywa_isaac.py`，适配层为
`planning/dywa_interface.py`。
学生策略在 DyWA 自己的模拟器和控制器中执行，动作保持时间为预训练使用的 100 ms。
评估 reset 直接初始化环境和学生历史，跳过训练包装器的 `mixed_reset` 随机动作预热，
避免在开始计分前扰动物体和机械臂。

## 不加载权重的检查

从仓库根目录运行：

```bash
python examples/mpc/franka/experiments/ik2/test_dywa_isaac.py \
  --headless --trial_num 1 --max_rollout_length 20 --dywa-dry-run
```

这会生成 `outputs/dywa_isaac/trial_000/spec.json`，包含源位姿、映射后的位姿和桌面参数。
初始和目标 Z 都加上两张桌面的高度差，保留初始离桌间隙和目标支撑高度。
配置中的 DyWA 桌面中心为 `[0,0,0.2]`、尺寸为 `[0.4,0.5,0.4]`，桌面顶部为 0.40 m。
无法容纳原任务位移时会报错，避免悄悄缩小任务。默认凹形物体碰撞分解仍需要 CoACD；
`--collision-geometry hull` 可用于独立验证凸包场景参数，但会改变碰撞几何。

## 完整执行

需要恢复原机器上包含自定义球形末端、物体初始化、目标位姿和 mesh 质量设置的 DyWA 源码。
未修改的官方 DyWA 没有这些配置接口；当前入口会检查配置能力，并在 reset 后读回实际物体和目标位姿。
学生模型权重使用严格加载，避免架构不一致时部分加载后继续执行。

源码默认放在 `thirdparty/DyWA`。权重默认放在其 `ckpts` 目录：

- `last.ckpt`：学生策略。
- `teacher-last.ckpt`：教师策略。
- `env-last.ckpt`：环境归一化统计。
- `512-32-balanced-SAM-wd-5e-05-920.ckpt`：ICP 网络编码器。

可通过 `DYWA_ROOT` 指定源码根目录，通过 `--dywa-ckpt-dir` 指定权重目录。
解释器优先使用 `DYWA_PYTHON`，否则使用当前 acc 环境或用户主目录下的 acc 环境。
入口和子进程会配置 Python 的 lib 目录与 PATH，无需手工填写旧机器路径。

```bash
DYWA_ROOT=/path/to/custom/DyWA \
DYWA_PYTHON=/path/to/envs/acc/bin/python \
python examples/mpc/franka/experiments/ik2/test_dywa_isaac.py \
  --headless --trial_num 1 --max_rollout_length 300 \
  --dywa-ckpt-dir /path/to/ckpts
```

Isaac Gym 必须先于 PyTorch 导入。需要原 DyWA 环境，包括 tensorboard、PyTorch3D 的 Python 包及 transforms；
本仓库的替代实现处理不兼容的 FlashAttention/PyTorch3D 扩展，不会替代整个 DyWA 环境。
缺失源码、权重和依赖会明确报错。

## 验证

```bash
python -m pytest tests/test_dywa_interface.py -q
```

测试覆盖桌面高度与相对位移、ICP 旋转和平移、reset 位姿核对、禁止随机预热、运行路径和缺失文件处理。
这些测试不加载策略权重，不能代表完整策略已经建立接触或完成任务。

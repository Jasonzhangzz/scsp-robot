# 距离夹紧修复（2026-10-16）

按本次要求保留 MANO 接触区域，通过伺服刚度、闭合距离和力限增加夹持力。
没有加入按承重能力改换接触区域的规划。已移除承重通过才允许运输、按力证书
优先挑选候选、在线静力分配和接触力误差压力反馈。

最新入口默认屏蔽 Lambda：`lambda_mode: "off"`。所有 MANO 示范接触都保留为
吸引目标，几何不可达不再删点。MPC 自己评估附近投影控制；原始点始终保留为
软先验，脱离后仍回到示范点。旧 Lambda 模式仅保留给显式对照。

## 参数

| 参数 | 默认值 | 作用范围 |
| --- | ---: | --- |
| `finger_stiffness_scale` | 1 | 对齐 DexForge 原始指关节 kp |
| 阻尼倍率 | 1 | 对齐原始阻尼 |
| `finger_force_scale` | 1 | 对齐原始两层扭矩限幅 |
| `closure_depth` | 0.004 m | 原接触法向上的虚拟伺服压紧距离 |
| `force_limit` | 100 N | 同时传入 Lambda 与 MPC |
| `sim_dt` | 0.002 s | 同 FA 控制原生回放验证后的分步 |
| `command_dt` | 0.03 s | 保持原控制周期 |
| `physics_profile` | dexforge | 使用匹配 SDF 场景的物体与地面参数 |

原来 50 倍刚度、10 倍力限是历史调参实验。根据新的参数对齐要求，当前默认
恢复倍率 1；这些调节接口保留。最新实测见 [DEXFORGE_TRAJECTORY_TEST.md](DEXFORGE_TRAJECTORY_TEST.md)。

闭合距离作用在虚拟伺服目标上：实测手指可以保持在物体表面，伺服目标继续向内
形成弹簧压紧量。几何接触 IK 仍拟合 MANO 原表面，不要求实际手指穿透 4 mm，
也不会因此改变原接触区域。原有小邻域搜索仅处理几何 IK 不可达。

`scene.xml` 写入实际执行的伺服参数；MPC 从该编译模型读取刚度、阻尼和扭矩限幅。
`physics_parameters.json` 校验两者一致。acados 缓存键包含伺服和力限参数；闭合距离作为运行时目标传入，
无需为调整闭合量重新编译模型。物体质量、惯量和摩擦来自选定物理 profile，
与执行场景一致；保留原生碰撞几何，不注入力。

高刚度下伺服目标角仍可能因限幅而与实测角相差较大。因此目标点使用独立的
正向运动学和 Jacobian，原生候选评分逐阶段刷新真实目标点，避免把张开误判为闭合。
闭合距离是虚拟压紧量，与实际碰撞穿透容差分开设置。

接触后还提供按原生接触间隙计算的连续闭合候选偏置：只调整手指，单周期法向
增量最多 0.6 mm，并按关节刚度加权分配。它不读取接触力误差，不保证候选被
执行；MPC 仍按原生 rollout 代价和碰撞检查选择，包括保持位置的候选。

默认 `preload=0`、`load_balance=0`，自动重抓默认关闭。闭合/承重日志只是诊断，
不门控对象目标，也不参与 MPC 候选排序。显式启用 `--regrasp` 时只依据接触间隙
和 MANO 区域误差恢复，不按接触力强弱重开手指。

## 运行

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/zz/miniconda3/envs/pi0/bin/python examples/retarget/retarget.py \
  --sequence thirdparty/retarget_assets/dexycb_prepared/20200709_142802 \
  --output outputs/wuji_distance_grip --headless --no-demo-overlay \
  --max-steps 600
```

参数也可通过 `examples/retarget/default.yaml` 或同名 CLI 参数调整。

## 验证

测试覆盖实际伺服和两层限幅同步修改、MANO 几何不变、虚拟目标产生法向压紧量、
接触力或承重证书变化不改变候选排序或运输目标，以及真实状态不被影子预览修改。
三个 Wuji 测试文件共 72 项通过；这不等于已验证稳定抓取。

已保存的 `force100` 对照使用 50 倍刚度、10 倍扭矩限幅、100 N 力上限和 4 mm
虚拟闭合距离，包含独立伺服 FK，但早于新增的几何间隙偏置。240 个周期的原生
回放与记录位置一致，物体最低点最大离桌约 0.040 mm，未稳定抬升。
其第 120 周期的计划点和实际点均满足几何力闭合；实际手接触合力向上约
2.72 N，物重约 5.04 N。该结果不能再简单归因于 MANO 接触区域或力闭合缺失。

从这一状态开始的独立几何闭合试验，配合预设腕部抬升，曾短暂离桌约 10 mm，
随后丢失抓持。它不是完整 MPC 成功，也没有验证稳定保持。当前稳定夹持修复
仍未完成；DexForge 的具体差异见 [DEXFORGE_COMPARISON.md](DEXFORGE_COMPARISON.md)。

旧版力门控实验及失败记录保存在 `outputs/wuji_grip_fix_20261015/`，
不是当前默认控制逻辑。新版本试跑记录保存在 `outputs/wuji_distance_grip_20261015/`。

# 力闭合与仅腕部搬运的实测复核（2026-10-15）

本文件记录修复前的历史控制器。最新接触点驱动、无外部 wrench 目标的实现与测试
见 [CONTACT_POSE_MPC_VALIDATION.md](CONTACT_POSE_MPC_VALIDATION.md)。

如果真实抓持能够保持接触、预载及承载裕度，沿轨迹运动腕部就可以带动物体。
当前失败不能归因于“腕部运动本身不能搬运”：反事实试验中腕部确实跟上了命令，
但抓持没有保持。此前把瞬时几何力闭合解释为已抓稳，结论过强。

## 完整 Sugar Box 序列

运行目录：[wuji_soft_trigrasp_complete_142802](../../outputs/wuji_soft_trigrasp_complete_142802)。
完整执行 81 个控制周期、覆盖 69/69 个有效 MANO 手部帧；每帧几何 IK 和软跟踪。
物体只受真实 MuJoCo 接触驱动，没有注入 Lambda 理想力或示教物体位姿。
保持 DexForge 的 timestep=0.01、iterations=10、ls_iterations=50、金字塔摩擦锥、
物体质量 0.514 kg、重力 5.04234 N、手物摩擦系数 0.5；结构安装件碰撞排除
沿用 README 中已明确记录的配置。

| 实测量 | 142802 完整运行 |
| --- | ---: |
| 物理步数 | 243 |
| 实际原生摩擦锥的几何力闭合 | 2/243 步 |
| 最长连续几何力闭合 | 0.02 s |
| 保守证书：μ=0.4、最小法向力 0.05 N | 0/243 步 |
| 当前实测法向力预算内平衡重力和力矩 | 0/243 步 |
| 物体质心最大抬升 | 5.04 mm |
| 启用 MPC grasp response | 0/81 周期 |

审核使用全部实际手物接触，不含桌面、不含未建立的参考槽位。每个物理步重新计算，
常规接触筛选为法向力 ≥0.01 N。由保存的手部控制复现真实仿真，物体位置与原始运行
最大差异为 2.41e-14 m。[逐物理步证据](../../outputs/wuji_soft_trigrasp_complete_142802/full/force_closure_audit/certificates.json)。

MuJoCo 的 condim=3 原生金字塔使用实际接触切向轴上的四条射线；同一 μ 的圆锥
近似具有不同可行集合。该运行用圆锥近似判为闭合的物理步有 32 个，原生模型仅 2 个。
新增证书支持传入实际切向基，避免任意旋转一个四边形代替执行模型。
物理实现参考 [MuJoCo friction cones](https://mujoco.readthedocs.io/en/stable/computation/#friction-cones)。
当前 Lambda、SCM 和在线分配器仍使用其既有圆锥近似；**本次审核没有修改控制器摩擦模型**。
Lambda 的 `box` 是外接方形，不能直接当作 MuJoCo 原生金字塔替代。

## 固定手指目标、仅移动腕部

[test_wrist_transport.py](test_wrist_transport.py) 从上述运行真实出现原生几何闭合的
cycle=41/substep=1 状态开始，保存并恢复完整积分状态。每个分支从同一状态开始，
固定 20 个手指的原有位置伺服目标，保留当时伺服压入量，只改变六自由度腕部目标。
不运行 MPC、不使用搬运门控、不施加理想接触力、不改变增益、摩擦或质量。
保持分支末尾 0.4 s，检查最终状态；“立即”分支不等待预稳定。

| 分支 | 腕部实际上升 | 物体短暂最大上升 | 结果 |
| --- | ---: | ---: | --- |
| 固定全部伺服目标 | −1.48 mm | 0.19 mm | 起始闭合在下一个控制采样前丢失；最终桌面承担约 5.04 N |
| 立即平滑垂直抬升 40 mm | 38.51 mm | 1.58 mm | 手物接触最终全部丢失 |
| 立即沿当前参考帧 36 至峰值帧 52 的剩余腕轨迹 | 36.95 mm | 1.18 mm | 最终手部向上力为 0，物体侧倒到桌面 |
| 从该抓持状态重放完整抬升段的相对腕轨迹 | 115.41 mm | 1.93 mm | 接触丢失、物体侧倒 |
| 完整相对腕轨迹放慢三倍，先固定 0.3 s | 115.41 mm | 1.74 mm | 仍然失败 |

剩余轨迹试验中手指目标始终相同，但实测手指相对初始姿态的 RMS 偏移最大
0.136 rad。位置伺服保持目标并不等于负载下手形刚性固定。
初始证书中的全部接触法向力约 7.68 N；闭合几何包含弱支撑方向，不能仅凭
接触数或法向力总和推断六维承载能力。在线实测力并没有维持平衡物体重力所需的预载。

[全部分支数值](../../outputs/wuji_soft_trigrasp_complete_142802/full/wrist_only_test/summary.json)、
[腕部与物体位移图](../../outputs/wuji_wrist_transport_review/wrist_only_comparison.png)、
[相对完整腕轨迹的视频](../../outputs/wuji_soft_trigrasp_complete_142802/full/wrist_only_test/demo_immediate/rollout.mp4)。
图中的 `demo_immediate` 是完整抬升段的相对腕运动，不能混称为当前帧之后的剩余轨迹。

作为交叉检查，142517 的 85 周期记录前缀中也找到了原生闭合状态：固定目标可保持
闭合，腕部垂直升 40.32 mm 时物体仅升 16.12 mm；完整相对示教腕轨迹最终丢失接触。
该起始状态存在约 10 mm 最大穿透，不能作为有效稳定抓持或成功抬升结果。
它也不是完整闭环验收，不能与上面的完整 142802 混计。

## 对失败原因的限定

1. 几何力闭合表示允许的接触力方向能覆盖六维 wrench；不保证当前已经施加了
   所需力，也不保证有限驱动力、关节柔顺性和滑移下接触能保持。
2. 直接绕过 MPC 后仍然抓不住，排除了“只因 MPC 不让腕部抬升”这一单一解释。
   不能据此断言 MPC 没问题：当前以实测法向力下的承载证书开启 grasp response
   和运输代价，可能形成建立承载前不愿搬运的保守循环。
3. MPC/Lambda 的摩擦模型与原生执行模型不一致，且实际预载反馈没有形成稳定承载。
   这些需要修正；本次仅完成诊断和腕部对照，没有宣称稳定抬升已解决。
4. 审核中的静态电机力矩 LP 忽略关节限位、自接触和接触重排。它的不可行结果
   只针对该近似，不能作为真实抓持绝对无法抬升的证明。

因此合理的执行方式仍是：腕部跟踪示教，手指维持并调整抓持预载；手形与语义点
继续作为软代价。验收应看持续承载、物体随腕运动和保持结果，而不是瞬时闭合布尔值。

## 复现与检查

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/zz/miniconda3/envs/pi0/bin/python \
  examples/retarget/verify_force_closure.py outputs/wuji_soft_trigrasp_complete_142802
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/zz/miniconda3/envs/pi0/bin/python \
  examples/retarget/test_wrist_transport.py outputs/wuji_soft_trigrasp_complete_142802
```

相关测试 **102 passed**：Wuji retarget、SCM、Lambda、grasp response、wrench 分配和
闭合验证。新增检查覆盖原生切向基、摩擦方向、控制重放、有限法向力承载和静态
电机界限；单元测试通过不代表闭环抬升通过。

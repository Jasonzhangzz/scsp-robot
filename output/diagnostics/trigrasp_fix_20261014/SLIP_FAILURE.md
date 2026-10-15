当前修复已通过默认 rubber_duck 场景的 20 cm 抬升及连续 20 s 保持。逐周期日志和实际接触证书在 `default_hold20_t8.json`，摘要在 `slip_validation.json`；本结论限定于该默认场景。

控制侧把所需预载截低。原压力参考统一截到 0.6 N，而四指平衡分配需要拇指约 1.7–1.8 N、其余约 0.6 N；原分配器还把测量法向力同时当作未来法向力上下界，载荷变弱时不能请求补压。现在分开求解目标预载和当前可用力，目标遵守物理力预算并满足所需 wrench 等式，当前接触动力学及承重证书继续使用实际测量上限。真实抓取整体运动不再被当作相对静止物体的张开而补偿；参考 bigrasp 的保持门槛使用滞回，预载目标未达到也不代表已经脱离真实接触。

仿真侧有软摩擦稳态漂移。独立四点夹持实验中，两组都具有几何力闭合和实际载荷上限下的承重解，每点约 3 N，切向力仅占摩擦锥容量约 3.5%，没有外加力。NoSlip=0 仍在 1 s 内下滑约 0.122 mm；仅改为 3 次 NoSlip 后漂移约 0.005 mm，摩擦系数和法向力相同。测试为 `tests/test_mujoco_grasp_slip.py`。

这符合 [MuJoCo 官方说明](https://mujoco.readthedocs.io/en/stable/modeling.html)：正则化软摩擦即使有充足 Coulomb 余量，也可能存在非零稳态滑移；NoSlip 重新求解摩擦方向以削弱漂移，不增加摩擦系数。当前在线默认 elliptic / impratio=10 / Newton tolerance=1e-8 / NoSlip=3，法向预载下限 0.6 N；`--contact-noslip-iterations 0` 可用于软摩擦对照。效果取决于收敛，不代表所有情形都达到严格零滑移。

单项回退压力截断时，最高抬升只有约 5.4 mm。足够且平衡的预载可抬升到 20 cm，但关闭 NoSlip 的长时间试验仍会漂移、最终滑脱。指腹法向成本和额外法向姿态约束试验没有解决问题，已撤回；不能据此认定球形指腹滚动是主要原因。调整力成本权重也未验证为有效修复。

最终直接执行 trigrasp.py，未使用 monkeypatch、物体外加力或状态锁定：

```bash
python3 examples/mpc/franka/trigrasp/trigrasp.py --headless --screenshot-interval 0 \
  --scene-output /tmp/trigrasp_default_hold20_t8.xml --hold-steps 800 --max-steps 1600 \
  --diagnostics-output output/diagnostics/trigrasp_fix_20261014/default_hold20_t8.json
```

最终高度增量 0.199985 m；连续 800 个保持周期（20 s）的真实接触力闭合和承重证书全部通过，最少三指有效接触，无全局重抓。最后 5 s 的位置误差中位数 0.491 mm、最大 2.426 mm，最终三维位置误差 1.555 mm。单步位置预测误差中位数 0.041 mm、95 分位 0.261 mm；此结果不支持 softSCM 整体预测偏差很大的说法。

71 项相关测试通过。库缓存复用启动已验证，编译中间文件已再次清理，共享库与模型 JSON 保留。

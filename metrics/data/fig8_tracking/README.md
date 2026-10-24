# Fig. 8(b) 曲线重建

`tro_revision/experiment.tex` 中 `fig:simulation_data` 使用的是
`tro_revision/figures/franka_exp_result.pdf`。其中 (b) 的原图位于
`/home/zz/figs/论文所有图.pptx` 的以下归档成员：

| 本地原图 | 内容 | 用户指定的时间换算 |
| --- | --- | --- |
| image177.png | Case 1 位置误差，原横轴最后标注 1200 | timestep × 0.016 |
| image178.png | Case 1 四元数误差，原横轴最后标注 1200 | timestep × 0.016 |
| image179.png | Case 2 位置误差，原横轴最后标注 1500 | timestep × 0.013 |
| image180.png | Case 2 四元数误差，原横轴最后标注 1500 | timestep × 0.013 |

这四张原图是 PNG，没有找到对应绘图脚本或逐试验数值数据。CSV 是根据颜色分割、
原图刻度标定得到的近似读图结果，不能当作原始实验测量数据。时间因子来自本次用户
指示，不是从日志恢复的实测耗时。原曲线末端略超出最后标注的刻度，因此换算后的
末端约为 19.66 s 和 24.67 s。

每份 CSV 保存 timestep、换算时间、提取的 mean、显示的 mean、可见阴影下界和上界。
显示曲线使用 20% 原曲线加 80% Gaussian 平滑曲线，Gaussian sigma 为 6 个 timestep，
保留首尾值，不添加随机噪声。阴影边界独立提取，不随 mean 平滑；原图末段没有阴影、
被图例遮挡或无法可靠分割的部分不补造标准差。因此阴影是原图可见区域的近似重建，
不代表重新计算了 10 次试验的统计量。四元数误差沿用原图无量纲数值，不转换为角度。

输出位于 `metrics/figures/fig8_tracking/`，包括四张单图 SVG、合并 SVG 和 PNG 预览。
所有文字统一使用 Times New Roman，生成时检查该字体可用，不回退到其他字体。
SVG 的文字字号为轴标签 21 pt、刻度 18 pt、图例 14 pt；文字和曲线均为矢量元素，
没有嵌入位图。来源说明也写入各 SVG 的元数据。

在仓库根目录重新生成：

```bash
/usr/bin/python3 metrics/rebuild_fig8_tracking.py
```

初次从 PPTX 提取源 PNG（现已保存在此目录，无须重复提取）：

```bash
/usr/bin/python3 metrics/rebuild_fig8_tracking.py \
  --source-pptx /home/zz/figs/论文所有图.pptx
```

`reconstruction.json` 记录 PNG 哈希、标定参数、平滑参数及平滑前后波动量。
本次仅输出独立矢量图，没有替换论文现有组合 PDF 或修改 `experiment.tex`。

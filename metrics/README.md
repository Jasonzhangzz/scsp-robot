# Metrics plotting bundle

这些文件来自
`/home/lab423/scsp/Franka-contact-face-detection-manipulation-main`，集中用于三类结果图。

## 文件对应关系

- `plot_resultt_panda.py`：Panda 实验成功率柱状图。
- `plot_result_fingertip.py`：Fingertip 实验成功率柱状图。
- `data/bar_chart_data.json`：两份柱状图脚本中的数值数据。
- `plot_comparision.py`：各方法综合 pose error 与真实 Execution Time (s) 曲线。
- `data/error_curves/*.csv`：误差与每条曲线对应的 `*_execution_time_s` 数据。
- `plot_pose_errors.py`：位置误差和姿态误差分别随真实 Execution Time (s) 的曲线；位置误差为欧氏距离，姿态误差为四元数最短夹角（度）。
- `trajectory_io.py`：读取 `output/trials/trajectories` 的真实缓存，并按 A5=DyWA、A4=Sampling、A2=ContactSDF、A7=STOCS、A6=CF-MPC 修正历史命名。
- `plot_ablation.py`：按同一修正映射绘制 A1--A7 的真实缓存曲线；缺失的 A5/A6/A7 缓存保持为空。
- `data/error_curves/execution_time/source_selection.json`：记录 piggy_bank/foam_brick 的成功 trial 选择和缓存缺失情况。
- `data/pose_errors/*.csv`：对应六个对象/试验的 pos/ori 误差数据。
- `data/trajectories/`、`data/trials/`：生成上述曲线所需的精简轨迹和目标位姿数据。
- `all_metrics.py`、`all_results.json`：原仓库的平均步数/时间统计工具及其数据。
- `figures/`：原仓库已有的 SVG 图，作为结果参考。

## 运行示例

在本目录执行：

```bash
python plot_resultt_panda.py
python plot_result_fingertip.py

python plot_comparision.py \
  --object-name elephant --trial-id 0 \
  --output-root data/trajectories \
  --trails-path data/trajectories/videos_panda/elephant/trails.json \
  --execution-time-root ../output/trials/trajectories \
  --save-dir data/error_curves/generated --no-show

python plot_pose_errors.py \
  --object-name elephant --trial-id 0 \
  --execution-time-root ../output/trials/trajectories \
  --output-dir data/pose_errors/generated
```

缓存 CSV 的 `sim_time_s` 会被归一化为从 0 开始的秒数；没有真实缓存时，脚本才回退到精简 JSON 轨迹中的时间字段。使用 `plot_comparision.py --auto-success` 可自动选择所有指定真实缓存都成功的 trial；若 A5/A4/基线没有共同成功轨迹，脚本会明确报告缺失，不会用合成成功曲线替代它。

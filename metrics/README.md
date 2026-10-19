# Metrics plotting bundle

这些文件来自
`/home/lab423/scsp/Franka-contact-face-detection-manipulation-main`，集中用于三类结果图。

## 文件对应关系

- `plot_resultt_panda.py`：Panda 实验成功率柱状图。
- `plot_result_fingertip.py`：Fingertip 实验成功率柱状图。
- `data/bar_chart_data.json`：两份柱状图脚本中的数值数据。
- `plot_comparision.py`：各方法综合 pose error 与 timestep 曲线。
- `data/error_curves/*.csv`：原仓库已经生成的 error–timestep 数据。
- `plot_pose_errors.py`：位置误差和姿态误差分别随 timestep 的曲线；位置误差为欧氏距离，姿态误差为四元数最短夹角（度）。
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
  --save-dir data/error_curves/generated --no-show

python plot_pose_errors.py \
  --object-name elephant --trial-id 0 \
  --output-dir data/pose_errors/generated
```

`plot_pose_errors.py` 会把位置误差和姿态误差写入 CSV，并保存上下排列的 SVG 图。源数据中没有 `w/o RS` 或某些对象/试验的轨迹时，相应方法会跳过；`I-MPC` 使用初始位姿误差形成固定参考曲线。

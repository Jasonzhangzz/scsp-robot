#!/usr/bin/env python3
"""Pool mean±std and paired amplitude-correction results from recorded probes.

No simulation or window is opened. MJWarp is the numerical reference. Paired
bootstrap intervals resample inputs within each shape, keeping this particular
shape/parameter mixture fixed; they are not a model-equivalence test.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

if __package__:
    from .runtime_env import activate_dependencies
else:
    from runtime_env import activate_dependencies
activate_dependencies()

import numpy as np
if __package__:
    from .compare_prediction import MODES, finite_stats, plot_results
else:
    from compare_prediction import MODES, finite_stats, plot_results

ENGINES = ("comfree", "scm", "scm_corrected")
METRICS = ("linear_velocity_error_m_s", "angular_velocity_error_rad_s",
           "position_error_m", "orientation_error_rad", "velocity_weighted_error",
           "increment_angle_deg", "step_seconds", "scm_amplitude_scale")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    rows, checks = [], []
    for folder in sorted(root.glob("*/")):
        if not (folder / "predictions.csv").exists():
            continue
        rows.extend(csv.DictReader((folder / "predictions.csv").open()))
        metadata = json.loads((folder / "metrics.json").read_text())["metadata"]
        source = metadata.get("replay_source")
        if source:
            with np.load(Path(source) / "states.npz", allow_pickle=False) as old, \
                    np.load(folder / "states.npz", allow_pickle=False) as new:
                mask = old["engine"] == "mjwarp"
                old_inputs = {int(old["sample_id"][i]): i for i in np.flatnonzero(mask)}
                input_checks = 0
                for i in range(len(new["sample_id"])):
                    original = old_inputs[int(new["sample_id"][i])]
                    for field in ("qpos_initial", "qvel_initial", "wrench_world_N_Nm"):
                        if not np.array_equal(new[field][i], old[field][original]):
                            raise AssertionError(f"{folder}: replayed {field} differs")
                    input_checks += 1
                deltas = {}
                for engine in ("mjwarp", "comfree", "scm"):
                    old_indices = np.flatnonzero(old["engine"] == engine)
                    new_indices = np.flatnonzero(new["engine"] == engine)
                    np.testing.assert_array_equal(old["sample_id"][old_indices], new["sample_id"][new_indices])
                    deltas[engine] = float(np.max(np.abs(old["qvel_predicted"][old_indices] - new["qvel_predicted"][new_indices])))
                checks.append({"configuration": folder.name, "verified_prediction_inputs": input_checks,
                               "baseline_max_qvel_change": deltas})
    if not rows or any(r["status"] != "ok" for r in rows):
        raise ValueError("Expected a completed four-model prediction run; inspect failed/missing probes")
    stats, comparisons = [], []
    rng = np.random.default_rng(20261009)
    for mode in MODES:
        for engine in ENGINES:
            selected = [r for r in rows if r["reference_mode"] == mode and r["engine"] == engine]
            for metric in METRICS:
                values = [float(r[metric]) for r in selected if r.get(metric)]
                stats.append({"reference_mode": mode, "engine": engine, "metric": metric,
                              **finite_stats(values)})
        for base in ("scm", "comfree"):
            for metric in METRICS[:-1]:
                selected = [r for r in rows if r["reference_mode"] == mode and r["engine"] == base and r.get(metric)]
                corrected = {(r["shape"], r["sample_id"]): r for r in rows
                             if r["reference_mode"] == mode and r["engine"] == "scm_corrected" and r.get(metric)}
                pairs = [(r["shape"], float(r[metric]), float(corrected[(r["shape"], r["sample_id"])][metric]))
                         for r in selected if (r["shape"], r["sample_id"]) in corrected]
                if not pairs:
                    continue
                a, b = np.array([p[1:] for p in pairs]).T
                difference = a - b
                bootstrap = np.zeros(2000)
                for shape in sorted({p[0] for p in pairs}):
                    values = difference[[p[0] == shape for p in pairs]]
                    bootstrap += values[rng.integers(0, len(values), (2000, len(values)))].sum(axis=1)
                bootstrap /= len(pairs)
                lo, hi = np.quantile(bootstrap, [.025, .975])
                comparisons.append({"reference_mode": mode, "baseline": base, "metric": metric,
                                    "paired_count": len(pairs), "baseline_mean": float(a.mean()),
                                    "corrected_mean": float(b.mean()),
                                    "reduction_fraction": float(1 - b.mean() / a.mean()) if a.mean() > 1e-12 else None,
                                    "corrected_to_baseline_mean_ratio": float(b.mean() / a.mean()) if a.mean() > 1e-12 else None,
                                    "paired_difference_mean": float(difference.mean()),
                                    "paired_difference_std": float(difference.std()),
                                    "paired_difference_mean_ci95_low": float(lo),
                                    "paired_difference_mean_ci95_high": float(hi),
                                    "lower_value_rate": float(np.mean(b < a)), "equal_value_rate": float(np.mean(b == a))})
    write_csv(root / "pooled_mean_std.csv", stats)
    write_csv(root / "paired_comparisons.csv", comparisons)
    report = ["# SCM 解析幅值修正结果", "",
              "数值参考：固定版本 MJWarp。每种模式汇总五种形状各 100 个输入；std 为总体标准差。",
              "输入由上一次实验的原始位姿、速度和外力直接回放；每个模型每次查询均重置。", "",
              "| 模式 | 模型 | n | 线速度误差 (m/s) | 角速度误差 (rad/s) | 单步耗时 (ms) |",
              "|---|---|---:|---:|---:|---:|"]
    for mode in MODES:
        for engine in ENGINES:
            group = {r["metric"]: r for r in stats if r["reference_mode"] == mode and r["engine"] == engine}
            cells = []
            for metric, scale in ((METRICS[0], 1), (METRICS[1], 1), ("step_seconds", 1000)):
                s = group[metric]
                cells.append(f"{s['mean'] * scale:.6g} ± {s['std'] * scale:.6g}")
            report.append(f"| {mode} | {engine} | {group[METRICS[0]]['count']} | " + " | ".join(cells) + " |")
    report += ["", "逐形状结果见配置目录；逐样本配对改善比例和置信区间见 paired_comparisons.csv。",
               "补充置信区间按形状分层重采样 2000 次，仅描述这些测试输入；未进行模型等价或非劣效检验。",
               "计时为单世界同步完整 step，不含首次编译、输入上传、读回、CPU 诊断和渲染；不是并行吞吐量。",
               "结论限于当前参数、点接触以及 full-dynamic SCM 扩展，不直接代表原始准动态 CSO。", ""]
    report += ["## 当前参数下的判断", ""]
    for mode in ("sliding", "sticking"):
        linear = next(r for r in comparisons if r["reference_mode"] == mode and r["baseline"] == "scm" and r["metric"] == METRICS[0])
        cf_linear = next(r for r in comparisons if r["reference_mode"] == mode and r["baseline"] == "comfree" and r["metric"] == METRICS[0])
        report.append(f"{mode}：平均线速度误差下降 {100 * linear['reduction_fraction']:.1f}%，"
                      f"{100 * linear['lower_value_rate']:.1f}% 输入的线速度误差改善；"
                      f"修正后均值仍为 ComFree 的 {cf_linear['corrected_to_baseline_mean_ratio']:.2f} 倍。")
    report += ["", "修正有效，但这批数据尚不支持整体媲美 ComFree；sticking 的平均角速度误差稍低，sliding 两类速度误差仍更高。", ""]
    (root / "amplitude_comparison.md").write_text("\n".join(report))
    (root / "replay_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    numeric_rows = [{**r, **{metric: float(r[metric]) if r.get(metric) else None for metric in METRICS}} for r in rows]
    plot_results(numeric_rows, root)
    print(f"Saved pooled/paired reports and verified replay inputs in {root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    analyze(parser.parse_args().input_dir)

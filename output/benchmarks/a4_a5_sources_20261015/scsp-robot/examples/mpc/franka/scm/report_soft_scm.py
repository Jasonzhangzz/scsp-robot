#!/usr/bin/env python3
"""Offline mean±std, paired errors and exact-input checks for soft SCM."""

import argparse
import csv
import json
from pathlib import Path

from runtime_env import activate_dependencies
activate_dependencies()
import numpy as np
from compare_prediction import MODES, finite_stats, plot_results

METRICS = ("linear_velocity_error_m_s", "angular_velocity_error_rad_s", "velocity_weighted_error",
           "position_error_m", "orientation_error_rad", "increment_angle_deg")


def report(root):
    rows, checks = [], []
    for folder in sorted(root.glob("*/")):
        if not (folder / "predictions.csv").exists():
            continue
        with (folder / "predictions.csv").open() as stream:
            rows.extend(dict(r, configuration=folder.name) for r in csv.DictReader(stream))
        with np.load(folder / "states.npz", allow_pickle=False) as states:
            ref = states["engine"] == "mjwarp"
            for engine in np.unique(states["engine"]):
                selected = states["engine"] == engine
                for key in ("sample_id", "qpos_initial", "qvel_initial", "wrench_world_N_Nm"):
                    np.testing.assert_array_equal(states[key][ref], states[key][selected])
            checks.append(dict(configuration=folder.name, shared_states=int(ref.sum()),
                               engines=list(np.unique(states["engine"])), all_inputs_exact=True))
    if not rows or any(r["status"] != "ok" for r in rows):
        raise ValueError("Report requires completed finite predictions; inspect raw failures")
    engines = sorted({r["engine"] for r in rows} - {"mjwarp"})
    groups = [("pooled", rows)]
    settings = sorted({(r["friction"], r["timestep_s"]) for r in rows})
    groups += [(f"mu={mu},h={h}", [r for r in rows if r["friction"] == mu and r["timestep_s"] == h]) for mu, h in settings]
    stats, pairs = [], []
    lines = ["# 软化 SCM 独立验证", "", "参考为固定版本 MJWarp；mean±std 为逐样本汇总，总体标准差。",
             "所有模型接收相同初始位姿、速度和外力，逐项相等已检查。此报告不包含训练样本。",
             "", "| 分组 | 模式 | 模型 | n | 线速度误差 (m/s) | 角速度误差 (rad/s) |",
             "|---|---|---|---:|---:|---:|"]
    for label, group in groups:
        for mode in MODES:
            for engine in engines:
                selected = [r for r in group if r["reference_mode"] == mode and r["engine"] == engine]
                if not selected:
                    continue
                metrics = {key: finite_stats([float(r[key]) for r in selected if r.get(key)]) for key in METRICS}
                for key, value in metrics.items():
                    stats.append(dict(group=label, mode=mode, engine=engine, metric=key, **value))
                a, b = metrics[METRICS[0]], metrics[METRICS[1]]
                lines.append(f"| {label} | {mode} | {engine} | {len(selected)} | {a['mean']:.6g} ± {a['std']:.6g} | {b['mean']:.6g} ± {b['std']:.6g} |")
            base = {(r["configuration"], r["sample_id"]): r for r in group if r["reference_mode"] == mode and r["engine"] == "comfree"}
            soft = [r for r in group if r["reference_mode"] == mode and r["engine"] == "scm_soft"]
            for key in METRICS:
                matched = [(float(base[(r["configuration"], r["sample_id"])][key]), float(r[key])) for r in soft
                           if r.get(key) and base[(r["configuration"], r["sample_id"])].get(key)]
                if not matched:
                    continue
                a, b = np.array(matched).T
                pairs.append(dict(group=label, mode=mode, metric=key, count=len(a),
                                  comfree_mean=float(a.mean()), scm_soft_mean=float(b.mean()),
                                  mean_ratio=float(b.mean() / a.mean()) if a.mean() > 1e-12 else None,
                                  soft_lower_error_rate=float(np.mean(b < a)),
                                  paired_difference_mean=float(np.mean(a - b)),
                                  paired_difference_std=float(np.std(a - b))))
    for name, data in (("soft_mean_std.csv", stats), ("soft_vs_comfree_paired.csv", pairs)):
        with (root / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    lines += ["", "输入/参数分布、训练记录和验证命令见 manifest.json 以及 soft_scm_calibration/selected_parameters.json。",
              "本实验为全动态四面接触乘子扩展，不能直接替代原论文准动态 CSO 的端到端验证。",
              "未作等价或非劣效检验；更低的数值参考误差不等于真实物理精度证明。", ""]
    (root / "soft_comparison.md").write_text("\n".join(lines))
    (root / "same_input_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    plot_results([{**r, **{key: float(r[key]) if r.get(key) else None for key in METRICS}} for r in rows], root)
    print(f"Saved {root.resolve()}/soft_comparison.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    report(parser.parse_args().input_dir)

"""Paired old/new SCM statistics, with identical input states and LCP reference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CONDITIONS = ("transition", "sliding", "sticking", "rollout")
LABELS = {"transition": "接触转换", "sliding": "滑动", "sticking": "粘着",
          "rollout": "GT 路径上的单步预测"}


def stats(values):
    values = np.asarray(values, dtype=float)
    assert values.size and np.isfinite(values).all()
    return dict(n=len(values), mean=float(values.mean()), std=float(values.std()),
                p95=float(np.percentile(values, 95, method="linear")), max=float(values.max()))


def key(row):
    return tuple(row[k] for k in ("object", "condition", "sample", "step"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(source, baseline):
    new, old = (json.loads(path.read_text()) for path in (source, baseline))
    assert new['soft_scm_corrected']['law_version'] == 'softplus_nonnegative_gap_v2'
    assert old['soft_scm_corrected']['law_version'] == 'softplus_impulse_v1'
    assert new['soft_scm_corrected']['temperature_N_s'] == old['soft_scm_corrected']['temperature_N_s']
    assert new['reference'] == old['reference'] == 'mjwarp_coupled_pyramid_lcp_2x0.001'
    for field in ('objects', 'samples', 'rollout_steps', 'seed', 'h_model', 'h_gt',
                  'friction', 'mass', 'angular_frame', 'lcp'):
        assert new[field] == old[field], field
    previous = {key(r): r for r in old['rows']}
    assert len(previous) == len(new['rows']) == len(old['rows'])
    assert len({key(r) for r in new['rows']}) == len(new['rows'])
    rows = []
    max_reference_difference = 0.
    for row in new['rows']:
        before = previous[key(row)]
        for field in ('qpos_initial', 'qvel_initial', 'wrench_world_N_Nm'):
            np.testing.assert_allclose(row[field], before[field], atol=1e-12, rtol=0)
        ref = np.asarray(row['reference_velocity_world'])
        max_reference_difference = max(max_reference_difference,
                                      float(np.max(np.abs(ref-before['reference_velocity_world']))))
        np.testing.assert_allclose(ref, before['reference_velocity_world'], atol=1e-10, rtol=0)
        np.testing.assert_allclose(row['prediction_velocity_world']['SCM'],
                                   before['prediction_velocity_world']['SCM'], atol=5e-5, rtol=1e-4)
        entry = {k: row[k] for k in ('object', 'condition', 'sample', 'step')}
        entry['penetrating'] = row['scm_diagnostics']['softSCM_corrected']['penetrating_rows'] > 0
        if not entry['penetrating']:
            np.testing.assert_array_equal(row['prediction_velocity_world']['softSCM_corrected'],
                                          before['prediction_velocity_world']['softSCM_corrected'])
        for name, raw_row, model in [('signed_hard', row, 'SCM'),
                                      ('signed_softplus', before, 'softSCM_corrected'),
                                      ('zero_gap_softplus', row, 'softSCM_corrected'),
                                      ('ComFree', row, 'comfree'), ('MuJoCo_QP', row, 'M1'),
                                      ('ContactSDF', row, 'ContactSDF')]:
            delta = np.asarray(raw_row['prediction_velocity_world'][model])-ref
            for metric, components in [('linear', slice(0, 3)), ('angular', slice(3, 6))]:
                value = float(np.linalg.norm(delta[components]))
                np.testing.assert_allclose(value, raw_row[model+'_'+metric], atol=1e-10, rtol=1e-10)
                entry[name+'_'+metric] = value
        rows.append(entry)
    certificates = [c for r in new['rows'] for c in r['reference_certificates']]
    assert len(certificates) == 2*len(rows)
    assert all(c['residual'] <= new['lcp']['residual_tolerance'] for c in certificates)
    models = ('signed_hard', 'signed_softplus', 'zero_gap_softplus', 'ComFree', 'MuJoCo_QP', 'ContactSDF')
    records, pairs, lookup = [], [], {}
    # Report conditions, objects, and the penetration subset separately.
    groups = [('condition', c, [r for r in rows if r['condition'] == c]) for c in CONDITIONS]
    groups += [('object', obj, [r for r in rows if r['object'] == obj]) for obj in new['objects']]
    groups += [('penetration', str(p), [r for r in rows if r['penetrating'] == p]) for p in (True, False)]
    groups += [('all', 'all', rows)]
    for category, group, selected in groups:
        if not selected:
            continue
        for model in models:
            for metric in ('linear', 'angular'):
                block = stats([r[model+'_'+metric] for r in selected])
                records.append(dict(category=category, group=group, model=model, metric=metric, **block))
                lookup[category, group, model, metric] = block
        for base in ('signed_hard', 'signed_softplus'):
            for metric in ('linear', 'angular'):
                a = np.array([r['zero_gap_softplus_'+metric] for r in selected])
                b = np.array([r[base+'_'+metric] for r in selected])
                pairs.append(dict(category=category, group=group, baseline=base, metric=metric,
                                  mean_reduction_percent=float(100*(1-a.mean()/b.mean())),
                                  improved=int(np.count_nonzero(a < b)), equal=int(np.count_nonzero(a == b)),
                                  worsened=int(np.count_nonzero(a > b)),
                                  **{'delta_'+k: v for k, v in stats(a-b).items()}))
    timing = {model: stats([r['prediction_step_ms'][model] for r in new['rows']])
              for model in ('SCM', 'softSCM_corrected')}
    metadata = dict(source=str(source.resolve()), source_sha256=sha(source),
                    baseline=str(baseline.resolve()), baseline_sha256=sha(baseline),
                    law=new['soft_scm_corrected'], n_states=len(rows),
                    reference_substeps=len(certificates),
                    max_reference_velocity_difference=max_reference_difference,
                    max_lcp_residual=max(c['residual'] for c in certificates),
                    penetrating_states=sum(r['penetrating'] for r in rows),
                    prediction_failures=0, step_timing_ms=timing,
                    timing_scope='synchronized GPU step incl. geometry, dynamics and integration; excludes reset, mass update and diagnostic downloads',
                    statistics=records, paired_differences=pairs)
    for filename, entries in [('nonnegative_gap_statistics.csv', records), ('nonnegative_gap_paired.csv', pairs)]:
        with (source.parent/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=entries[0].keys())
            writer.writeheader()
            writer.writerows(entries)
    (source.parent/'nonnegative_gap_comparison.json').write_text(json.dumps(metadata, indent=2)+'\n')
    lines = ['# 非负接触距离 softplus SCM：配对评估', '',
             f"{len(new['objects'])} 个物体；每种条件每物体 {new['samples']} 个初态；共 {len(rows)} 次共享状态单步预测，{len(certificates)} 个 GT 子步。",
             f"穿透状态 {metadata['penetrating_states']} 个；预测失败 0；最大 LCP 残差 {metadata['max_lcp_residual']:.6g}。",
             f"与旧数据逐条核对输入初态，GT 世界速度最大分量差 {max_reference_difference:.6g}。", '',
             '模型步长 0.002 s；GT 为带符号距离的耦合摩擦锥 LCP，两个 0.001 s 子步。',
             f"T={new['soft_scm_corrected']['temperature_N_s']:.8g} N·s，r=1e-8，沿 softplus 初始冲量做一次全局幅值修正。",
             '初态保留 MuJoCo 软接触静置的残余穿透。GT 路径上的预测每步重置到 GT 状态，不是候选模型独立累计的轨迹误差。', '',
             '## 相对原硬 SCM', '',
             '| 条件 | n | 原线速度 mean±std | 新线速度 mean±std | 原/新 P95 | 原角速度 mean±std | 新角速度 mean±std | 原/新 P95 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        blocks = [lookup['condition', condition, model, metric]
                  for metric in ('linear', 'angular') for model in ('signed_hard', 'zero_gap_softplus')]
        a, b, c, d = blocks
        lines.append(f"| {LABELS[condition]} | {a['n']} | {a['mean']:.6f}±{a['std']:.6f} | {b['mean']:.6f}±{b['std']:.6f} | {a['p95']:.6f}/{b['p95']:.6f} | {c['mean']:.6f}±{c['std']:.6f} | {d['mean']:.6f}±{d['std']:.6f} | {c['p95']:.6f}/{d['p95']:.6f} |")
    lines += ['', '单位：线速度 m/s，角速度 rad/s；std 为总体标准差。', '',
              '## 相对已有 softplus（带符号距离）', '',
              '| 条件 | 旧线速度 mean | 新线速度 mean | 线误差降低 % | 旧角速度 mean | 新角速度 mean | 角误差降低 % |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        values = []
        for metric in ('linear', 'angular'):
            a = lookup['condition', condition, 'signed_softplus', metric]['mean']
            b = lookup['condition', condition, 'zero_gap_softplus', metric]['mean']
            values.extend([f'{a:.6f}', f'{b:.6f}', f'{100*(1-b/a):.2f}'])
        lines.append('| '+LABELS[condition]+' | '+' | '.join(values)+' |')
    lines += ['', '负百分比表示误差增加。所有条件、物体及穿透子集的 n/mean/std/P95/max 和逐样本改善计数见 CSV/JSON。', '',
              '## 同步 GPU 单步耗时', '', '| 模型 | mean±std (ms) | P95 (ms) |', '|---|---:|---:|']
    for model, block in timing.items():
        lines.append(f"| {model} | {block['mean']:.6f}±{block['std']:.6f} | {block['p95']:.6f} |")
    lines += ['', '耗时包括碰撞、动力学、接触求解和积分以及设备同步，不含重置、质量更新和诊断数据下载；为单世界诊断运行，不代表批量吞吐。', '',
              'GT 的负距离项会主动产生穿透恢复速度；新模型按要求把该距离设为零。因此对这个 GT 的误差变化同时包含接触语义的变化，不能只归因于 softplus。', '',
              f'原始新数据：`{source.resolve()}`，SHA256 `{sha(source)}`。',
              f'原始旧数据：`{baseline.resolve()}`，SHA256 `{sha(baseline)}`。']
    output = source.parent/'nonnegative_gap_comparison.md'
    output.write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    args = parser.parse_args()
    report(args.source, args.baseline)

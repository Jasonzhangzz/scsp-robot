"""Recompute the manuscript velocity table from a single softplus-SCM run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MODELS = (("MuJoCo QP", "M1"), ("ComFree", "comfree"),
          ("ContactSDF", "ContactSDF"), (r"\textbf{SCM}", "softSCM_corrected"))
CONDITIONS = (("transition", "Contact transition"), ("sliding", "Sliding"),
              ("sticking", "Sticking"), ("rollout", "Reference-path rollout"))


def report(source, tex=None):
    raw = json.loads(source.read_text())
    soft = raw["soft_scm_corrected"]
    if soft["law_version"] not in ("softplus_impulse_v1", "softplus_nonnegative_gap_v2") or soft["response_time"] != 0:
        raise ValueError("Expected the new softplus SCM, not the historical spring-damper model")
    if soft["law_version"] == "softplus_nonnegative_gap_v2":
        assert soft["gap_policy"] == "max(signed_gap,0) in both initial impulse and amplitude correction"
    if raw["reference"] != "mjwarp_coupled_pyramid_lcp_2x0.001":
        raise ValueError("Expected the coupled MJWarp LCP reference")
    if raw.get("angular_frame") != "world; reference uses final body rotation after both substeps":
        raise ValueError("Expected corrected world angular velocities")
    rows = raw["rows"]
    assert len(rows) == len(raw["objects"])*raw["samples"]*(3+raw["rollout_steps"])
    assert len({(r['object'], r['condition'], r['sample'], r['step']) for r in rows}) == len(rows)
    certificates = [c for r in rows for c in r["reference_certificates"]]
    assert len(certificates) == 2*len(rows)
    assert all(c["residual"] <= raw["lcp"]["residual_tolerance"] for c in certificates)
    records, lookup = [], {}
    for condition, _ in CONDITIONS:
        group = [r for r in rows if r["condition"] == condition]
        expected = len(raw["objects"])*raw["samples"]*(raw["rollout_steps"] if condition == "rollout" else 1)
        assert len(group) == expected
        for _, name in (*MODELS, ("hard SCM", "SCM")):
            for metric in ("linear", "angular"):
                values = [r[name+"_"+metric] for r in group]
                assert all(v is not None and np.isfinite(v) for v in values), (condition, name, metric)
                values = np.asarray(values)
                stats = dict(n=len(values), mean=float(values.mean()), std=float(values.std()),
                             p95=float(np.percentile(values, 95, method="linear")), max=float(values.max()))
                saved = raw["summary"][condition][name]["linear_m_s" if metric == "linear" else "angular_rad_s"]
                assert stats == saved, (condition, name, metric)
                lookup[condition, name, metric] = stats
                records.append(dict(condition=condition, model=name, metric=metric, **stats))
    table = []
    for condition, label in CONDITIONS:
        cells = []
        for _, name in MODELS:
            for metric in ("linear", "angular"):
                block = lookup[condition, name, metric]
                best = min(lookup[condition, m, metric]['mean'] for _, m in MODELS)
                macro = r"\minerr" if block['mean'] == best else r"\err"
                cells.append(macro+f"{{{block['mean']:.2f}}}{{{block['std']:.2f}}}")
            for metric in ("linear", "angular"):
                value = lookup[condition, name, metric]['p95']
                best = min(lookup[condition, m, metric]['p95'] for _, m in MODELS)
                cells.append(r"$\bm{"+f"{value:.2f}"+"}$" if value == best else f"${value:.2f}$")
        table.append(label+" & "+" & ".join(cells)+r" \\")
    (source.parent/"paper_table_rows.tex").write_text("\n".join(table)+"\n")
    with (source.parent/"velocity_error_statistics.csv").open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    manifest = dict(source=str(source.resolve()), source_json_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_sha256=raw['source_sha256'], soft_scm_corrected=soft, rows=len(rows),
                    reference_substeps=len(certificates), max_lcp_residual=max(c['residual'] for c in certificates),
                    failed_paper_predictions=0)
    (source.parent/"report_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if tex:
        text = tex.read_text()
        start = text.index("Contact transition &")
        end = text.index(r"\bottomrule", start)
        text = text[:start]+"\n".join(table)+"\n"+text[end:]
        text = text.replace(r"\multicolumn{4}{c}{\textbf{softSCM\_corrected}}", r"\multicolumn{4}{c}{\textbf{SCM}}", 1)
        text = text.replace("MuJoCo's native quadratic-programming (QP) model, ComFree, softSCM\\_corrected, and ContactSDF",
                            "MuJoCo's native quadratic-programming (QP) model, ComFree, SCM, and ContactSDF", 1)
        marker = "% All table statistics from "
        start = text.index(marker)
        end = text.index('\n', start)
        text = text[:start]+marker+str(source.resolve().relative_to(ROOT))+"."+text[end:]
        text = text.replace("% SCM uses backend scm_corrected (response_time=0, r=1e-8, one global amplitude correction).",
                            "% softSCM_corrected: "+soft['law_version']+"; T="+str(soft['temperature_N_s'])+" N s; response_time=0; hard-clipped global alpha.")
        tex.write_text(text)
    print(json.dumps(manifest, indent=2))
    print("\n".join(table))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--tex", type=Path, help="Update this manuscript from the same raw run")
    args = parser.parse_args()
    report(args.source, args.tex)


if __name__ == "__main__":
    main()

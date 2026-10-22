"""Audit fixed rotation records against a fresh official checkout; optionally fill Table IV."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT/'output/benchmarks/contactsdf_official_rotation_15_20261020'
OUT = ROOT/'output/diagnostics/contactsdf_audit_20261022'
VENDOR = Path('examples/mpc/franka/scm/vendor/contactsdf')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(values):
    return dict(n=len(values), mean=statistics.mean(values), sample_std=statistics.stdev(values))


def audit():
    upstream = OUT/'upstream'
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    provenance = json.loads((ROOT/VENDOR/'PROVENANCE.json').read_text())
    assert commit == provenance['commit']
    checks = {}
    for name, expected in provenance['original_sha256'].items():
        checks[name] = dict(remote=digest(upstream/name), preserved=digest(ROOT/VENDOR/'upstream'/name), expected=expected)
        assert len(set(checks[name].values())) == 1, name
    source = (upstream/'planning/ball_mpc_lse.py').read_text()
    assert (ROOT/VENDOR/'ball_mpc_lse.py').read_text() == source.replace(
        'from dynamics.lse_dynamics import LseDynamic', 'from .lse_dynamics import LseDynamic')
    assert digest(upstream/'dynamics/lse_dynamics.py') == digest(ROOT/VENDOR/'lse_dynamics.py')
    original = (upstream/'contact/ball_lse_contact_detection.py').read_text()
    live = (ROOT/VENDOR/'ball_lse_contact_detection.py').read_text()
    # The import/type/reshape changes are inspected separately in detector.diff.
    import difflib
    (OUT/'detector.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), live.splitlines(True))))
    snapshot = BATCH/'source_snapshot/scsp-robot'
    frozen = {}
    for relative in [Path('planning/contactsdf_franka.py'), Path('examples/mpc/franka/contactsdf_run.py'),
                     *[VENDOR/name for name in ['ball_mpc_lse.py','ball_lse_contact_detection.py','lse_dynamics.py','single_ball_cost.py']]]:
        frozen[str(relative)] = dict(live=digest(ROOT/relative), frozen=digest(snapshot/relative))
        frozen[str(relative)]['matches_current'] = frozen[str(relative)]['live'] == frozen[str(relative)]['frozen']
    rows, solves, totals, squares, failures = [], [], 0., 0., 0
    cases = list(csv.DictReader((BATCH/'results.csv').open()))
    for case in cases:
        folder = BATCH/case['case_directory']
        trial = json.loads((folder/'trials.jsonl').read_text().strip())
        assert trial['target_type'] == 'rotation' and not trial['aborted']
        assert case['status'] == 'complete' and int(case['returncode']) == 0
        assert trial['planner_solver_backend'] == 'ipopt' and trial['policy_interval_s'] == .1
        samples = [float(x) for x in re.findall(r'mpc solving time:\s*([0-9.eE+-]+)', (folder/'run.log').read_text())]
        assert len(samples) == trial['planning_wall_count'] == trial['steps'] == 1000
        solves.extend(samples)
        totals += trial['planning_wall_total_s']
        squares += trial['planning_wall_sum_squares_s2']
        failures += trial['mpc_failures']
        geometry = next((folder/'contactsdf_geometry').glob('*_subsampled.txt'))
        planes = sum(1 for _ in geometry.open())
        rows.append(dict(case_id=case['case_id'], object=case['object'], planes=planes,
            full_cycle_mean_ms=trial['planning_wall_mean_ms'], solver_ms=moments([1000*x for x in samples]),
            outside_solver_mean_ms=trial['planning_wall_mean_ms']-1000*statistics.mean(samples),
            pos_err=trial['final_pos_err'], quat_err=trial['final_quat_err'], success=trial['success'],
            source=str(folder.relative_to(ROOT)), log_sha256=digest(folder/'run.log')))
    n = len(solves)
    full = dict(n=n, mean=totals/n, sample_std=math.sqrt((squares-totals*totals/n)/(n-1)))
    result = dict(upstream_commit=commit, original_hash_checks=checks, frozen_checks=frozen,
        cases=rows, success_count=sum(x['success'] for x in rows), trial_count=len(rows), mpc_failures=failures,
        planning_cycle_s=full, logged_solver_s=moments(solves),
        outside_logged_solver_mean_s=full['mean']-statistics.mean(solves),
        reciprocal_mean_cycle_hz=1/full['mean'],
        reciprocal_mean_solver_hz=1/statistics.mean(solves),
        terminal_position_m=moments([x['pos_err'] for x in rows]),
        terminal_quaternion_metric=moments([x['quat_err'] for x in rows]),
        terminal_angle_rad=moments([2*math.asin(math.sqrt(max(0,min(1,x['quat_err'])))) for x in rows]),
        note='Cycle includes detection and planning. Log timer includes IPOPT plus solution extraction, not pure IPOPT. Reciprocal means are throughput estimates, not mean instantaneous frequencies. Five concurrent processes in original batches.')
    summary = json.loads((BATCH/'summary.json').read_text())
    assert math.isclose(full['mean'],summary['planning_cycle_time_s']['mean'],rel_tol=1e-12)
    assert math.isclose(full['sample_std'],summary['planning_cycle_time_s']['std'],rel_tol=1e-12)
    (OUT/'verification_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def fill(result):
    path = ROOT/'tro_revision/experiment.tex'
    text = path.read_text()
    (OUT/'experiment.before.tex').write_text(text)
    def cell(m, scale=1):
        return '$%.2f\\pm%.2f$' % (m['mean']*scale,m['sample_std']*scale)
    values = [f"{result['success_count']/result['trial_count']:.2f}", '--',
        cell(result['planning_cycle_s'],1000), cell(result['terminal_position_m']),
        cell(result['terminal_quaternion_metric'])]
    begin = text.index(r'\multirow{5}{*}{Rotation}')
    end = text.index(r'\midrule',begin)
    lines = text[begin:end].splitlines(True)
    assert len(lines) == 5
    changes = []
    for index, (line, value) in enumerate(zip(lines,values)):
        cells = line.split(' & ')
        changes.append(dict(metric=cells[1], before=cells[4], after=value))
        cells[4] = value
        lines[index] = ' & '.join(cells)
    text = text[:begin]+''.join(lines)+text[end:]
    note = ('% ContactSDF Rotation: 15 fixed tasks, 15000 detection+MPC cycles; sample SD.\n'
            '% Source: output/benchmarks/contactsdf_official_rotation_15_20261020/summary.json\n')
    if note not in text:
        text = text.replace(r'\multirow{5}{*}{Rotation}',note+r'\multirow{5}{*}{Rotation}',1)
    path.write_text(text)
    (OUT/'table_changes.json').write_text(json.dumps(changes,indent=2)+'\n')
    # The current table uses the user's requested solver-log timing for both tasks.
    import sys
    subprocess.run([sys.executable,str(ROOT/'tro_revision/update_contactsdf_solver_timing.py')],check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    result = audit()
    if args.apply:
        fill(result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','original_hash_checks','frozen_checks']},indent=2))

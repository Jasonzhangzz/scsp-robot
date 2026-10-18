"""Run the 7-object x 5-seed x 3-model cohort and fill its paper table.

Designed for detached execution: worker logs, raw trials, aggregate statistics
and the final table update are written without interactive result monitoring.
One object runs sequentially on each distinct physical CPU core. No model is
retuned, and failed trials remain in the success denominator.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
OBJECTS = ('foam_brick','mug','rubber_duck','elephant','piggy_bank','stanford_bunny2','teapot')
MODELS = ('scm','qp','lcp')
SOURCE_FILES = (
    'planning/point_contact_baselines.py','planning/direct_contact_force.py',
    'planning/mlqp_point.py','planning/object_pose_cost.py','planning/rollout_timing.py',
    'models/soft_scm_contact.py','examples/mpc/franka/ik2/params.py',
    'examples/mpc/fingertips/test/params.py',
    'examples/mpc/fingertips/test/test_scm_mujoco.py',
    'examples/mpc/fingertips/test/benchmark_contact_models.py',
)


def validate_cohort(report, trials=5):
    expected={(m,o,s) for m in MODELS for o in OBJECTS for s in range(trials)}
    actual=[(t['model'],t['object'],t['seed']) for t in report['trials']]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Incomplete or duplicate cohort: expected %d unique trials, got %d'%
                         (len(expected),len(actual)))
    for model in MODELS:
        if report['models'][model]['total_trials']!=len(OBJECTS)*trials:
            raise ValueError('Incorrect success denominator for '+model)


def timing_cell(stats):
    if stats['n']==0:
        return '--'
    mean,std=stats['mean'],stats['std']
    if not all(math.isfinite(v) and v>=0 for v in (mean,std)):
        raise ValueError('Invalid timing statistic')
    return '$%.2f \\pm %.2f$'%(mean,std)


def fill_table(text,report,trials=5,data_dir=''):
    """Replace only this table's three model cells, plus its local protocol."""
    validate_cohort(report,trials)
    match=re.search(r'\\begin\{table\}[^\n]*\n(?:(?!\\end\{table\}).)*'
                    r'\\label\{tab:cso_contact_guidance\}'
                    r'(?:(?!\\end\{table\}).)*\\end\{table\}',text,re.S)
    if match is None:
        raise ValueError('Unique target table not found')
    table=match.group(0)
    values={
        'Success Rate':['%.1f\\%%'%(100*report['models'][m]['success_rate']) for m in MODELS],
        'Execution Time (s)':[timing_cell(report['models'][m]['execution_time_s_success_only']) for m in MODELS],
        'Planning Time (ms)':[timing_cell(report['models'][m]['planning_time_ms_all_rounds']) for m in MODELS],
    }
    for metric,cells in values.items():
        pattern=r'^'+re.escape(metric)+r'\s*&.*?\\\\\s*$'
        row=re.search(pattern,table,re.M)
        if row is None:
            raise ValueError('Missing metric row: '+metric)
        original=row.group(0).split('\\\\',1)[0].split('&')
        if len(original)!=8:
            raise ValueError('Expected three model columns and four solver columns')
        # Preserve the independent solver comparison cells, including any
        # measurements subsequently entered there by the user.
        replacement=' & '.join([metric]+cells+[c.strip() for c in original[4:]])+r' \\'
        table=table[:row.start()]+replacement+table[row.end():]
    table=re.sub(r'\\caption\{[^\n]*\}',
                 lambda _:r'\caption{CSO contact guidance: seven objects, '+str(trials)+
                 r' paired trials per object and model. Times are mean $\pm$ std; '
                 r'execution time is success-only wall time, and planning time covers all rounds.}',
                 table,count=1)
    table=table.replace(r'& MuJoCo & LCP &',r'& MuJoCo QP & LCP &')
    table='% Data: '+data_dir+'; seeds 0--'+str(trials-1)+'; 5000-step limit.\n'+table
    result=text[:match.start()]+table+text[match.end():]
    paragraph=(
        "We evaluate CSO contact guidance by applying the optimized point force at the selected contact location in MuJoCo. "
        "Seven objects (foam brick, mug, rubber duck, elephant, piggy bank, Stanford bunny, and teapot) each receive "
        +str(trials)+" paired initial states with seeds 0--"+str(trials-1)+", giving "+str(7*trials)+" tasks per model. "
        "These are flip-to-pose tasks with randomized yaw and initial tilt sampled between $35^\\circ$ and $75^\\circ$, "
        "with a 5000-step budget. Planning is synchronous with a $20\\,\\mathrm{ms}$ policy interval; the full optimized "
        "force is applied for $0.5\\,\\mathrm{ms}$ per interval. Success requires the target-pose tolerances, at least ten "
        "consecutive unforced settling intervals, and low terminal velocities. SCM uses the reference Isaac quasi-dynamic "
        "model and ACADOS; the MuJoCo baseline uses its native pyramidal-contact QP, and the LCP baseline is a regularized "
        "monotone friction-ray LCP solved with Lemke's algorithm. QP and LCP force optimization use finite-difference "
        "local sensitivities and one local SLSQP update. Thus the comparison includes both model and solver implementation "
        "differences. Table~\\ref{tab:cso_contact_guidance} reports measured success rates and mean $\\pm$ population "
        "standard deviations. Execution time is wall-clock time from the first plan to successful settling, including "
        "planning and simulation but excluding initialization; failed trials are excluded from this time statistic. "
        "Planning time includes every attempted planning round, including failed tasks. The separate solver-comparison "
        "columns are retained and are not populated by this contact-model experiment."
    )
    result,count=re.subn(r"We further evaluate whether CSO's contact guidance[^\n]*",
                        lambda _:paragraph,result,count=1)
    # If this paragraph has already been updated, preserve it on a rerun.
    return result


def source_hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_FILES}


def main():
    raise SystemExit('This paper-table runner describes the archived uncorrected SCM cohort. '
                     'Use benchmark_contact_models.py for the current corrected-SCM protocol; '
                     'historical table-formatting helpers remain available for recorded results.')


if __name__ == '__main__':
    main()

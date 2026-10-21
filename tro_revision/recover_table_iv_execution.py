"""Recover measured execution SDs and explicitly labeled proxy dispersion estimates.

Retains every published mean. Never manufactures trial records or labels another
method's trajectory as a measured DyWA/Sampling/CF-MPC execution.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT/'tro_revision'
METHODS = ('SCSP', 'DyWA', 'ContactSDF', 'STOCS-3D', 'Sampling', 'CF-MPC',
           'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7')
HASHES = {}


def read(path):
    path = Path(path)
    content = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
    return content.decode()


def load(path):
    return json.loads(read(path))


def normalize(row, path, cohort, method, task):
    if row.get('aborted'):
        raise ValueError('Aborted trial: '+str(path))
    success = row.get('success', row.get('contactsdf_success'))
    if success not in (True, False, 0, 1):
        raise ValueError('Missing boolean outcome: '+str(path))
    value = next((row[k] for k in ('completion_time_s', 'execution_time_s')
                  if row.get(k) is not None), None)
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError('Invalid execution time: '+str(path))
    if success and value is None:
        raise ValueError('Missing successful execution time: '+str(path))
    obj = row['object']
    seed = row.get('seed', row.get('trial'))
    if seed is None:
        raise ValueError('Missing trial identity')
    return dict(trajectory_id='%s/%s/%s/%s/%s' % (task, cohort, method, obj, seed),
                task=task, cohort=cohort, source_method=method, object=obj,
                trial=seed, success=bool(success), execution_time_s=value,
                source=str(Path(path).relative_to(ROOT)))


def jsonl_group(pattern, cohort, method, task):
    result = []
    for path in sorted(ROOT.glob(pattern)):
        for line in read(path).splitlines():
            if line.strip(): result.append(normalize(json.loads(line), path, cohort, method, task))
    return checked(result)


def metrics_group(pattern, cohort, method, task):
    return checked([normalize(load(p), p, cohort, method, task) for p in sorted(ROOT.glob(pattern))])


def checked(rows):
    ids = [r['trajectory_id'] for r in rows]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate trajectory identities')
    return rows


def file_group(relative, cohort, method, task, predicate=lambda r: True):
    path = ROOT/relative
    return checked([normalize(r, path, cohort, method, task) for r in load(path) if predicate(r)])


def summary(rows):
    values = [r['execution_time_s'] for r in rows if r['success']]
    return dict(trials=len(rows), successes=len(values),
                success_rate=len(values)/len(rows) if rows else None,
                mean_s=statistics.mean(values) if values else None,
                sample_std_s=statistics.stdev(values) if len(values) > 1 else None,
                successful_trajectory_ids=[r['trajectory_id'] for r in rows if r['success']])


def parse_paper(text):
    section=text.split(r'\label{tab:all_data}',1)[1].split(r'\end{table*}',1)[0]
    cells={}
    for task in ('Rotation','Flipping'):
        part=section.split(r'\multirow{5}{*}{'+task+'}',1)[1]
        success_line=part.splitlines()[0]
        time_line=next(line for line in part.splitlines() if ' & Execution Time (s) & ' in line)
        success=success_line.rsplit(r' \\',1)[0].split(' & ')[2:]
        timing=time_line.rsplit(r' \\',1)[0].split(' & ')[2:]
        assert len(success)==len(timing)==len(METHODS)
        cells[task]={}
        for method,s,t in zip(METHODS,success,timing):
            values=re.findall(r'\d+\.\d+',t)
            cells[task][method]=dict(original_latex=t, retained_mean_s=float(values[0]) if values else None,
                target_success_rate=float(re.search(r'\d+\.\d+',s).group()))
    return cells


def cohorts():
    direct={task:{} for task in ('Rotation','Flipping')}
    direct['Rotation']['SCSP']=jsonl_group('output/table4_rotation_20261020/*/trials.jsonl',
        'table4_rotation_20261020','SCSP','Rotation')
    for key in ('isaac','a1','a2','a3','a4','a5'):
        method='SCSP' if key=='isaac' else key.upper()
        direct['Flipping'][method]=metrics_group('output/trials/%s/*/*/metrics.json'%key,
            'trials_flipping',method,'Flipping')
    recent={}
    for i in range(1,8):
        method='A%d'%i; rows=[]
        for p in sorted(ROOT.glob('output/benchmarks/rotation_a1_a7_20261021_001433/a%d/*/normalized_trials.json'%i)):
            rows.extend(normalize(r,p,'rotation_20261021',method,'Rotation') for r in load(p))
        recent[method]=checked(rows)
        if i<=3:direct['Rotation'][method]=recent[method]
    archive='output/benchmarks/a4_a5_async_7x3_20261015/trials.json'
    for i in (4,5):
        direct['Rotation']['A%d'%i]=file_group(archive,'async_20261015','A%d'%i,'Rotation',
            lambda r,i=i:r['task']=='rotation' and r['variant']=='a%d'%i)
    direct['Rotation']['A6']=file_group('output/benchmarks/a6_rotation_async_7x3_20261015/trials.json',
        'async_a6_20261015','A6','Rotation')
    for task in direct:
        direct[task]['A7']=file_group('output/benchmarks/a7_mppi_7x3_20261015/trials.json',
            'mppi_20261015','A7',task,lambda r,task=task:r['task']==task.lower())
    pools={task:[r for rows in methods.values() for r in rows] for task,methods in direct.items()}
    # Additional distinct, complete archived cohorts; source/protocol is recorded.
    for method,rows in recent.items():
        if method not in ('A1','A2','A3'):pools['Rotation'].extend(rows)
    for i in (4,5):
        pools['Flipping'].extend(file_group(archive,'async_20261015','A%d'%i,'Flipping',
            lambda r,i=i:r['task']=='flipping' and r['variant']=='a%d'%i))
    for base,cohort in [('current_mpc_async_7x3_20261016','async_scsp_20261016'),
                        ('current_mpc_viewer_serial_7x3_20261016','viewer_scsp_20261016')]:
        pools['Flipping'].extend(file_group('output/benchmarks/'+base+'/trials.json',cohort,
            'SCSP','Flipping',lambda r:r['task']=='flipping'))
    for task in pools:checked(pools[task])
    return direct,pools


def zero_success_groups():
    result={task:{} for task in ('Rotation','Flipping')}
    for task,relative in (
        ('Rotation','output/benchmarks/contactsdf_official_rotation_15_20261020/summary.json'),
        ('Flipping','output/benchmarks/contactsdf_official_single_tip_15_20261020/summary.json')):
        path=ROOT/relative
        rows=[normalize(r,path,'contactsdf_official_20261020','ContactSDF',task)
              for r in load(path)['cases']]
        result[task]['ContactSDF']=rows
    result['Rotation']['STOCS-3D']=jsonl_group(
        'output/benchmarks/stocs3d_rotation_7x5_20261021/cases/*/trials/stocs3d/*/trials.jsonl',
        'stocs_rotation_20261021','STOCS-3D','Rotation')
    path=ROOT/'output/benchmarks/stocs3d_table4_15_20261020/table4_metrics.json'
    result['Flipping']['STOCS-3D']=[normalize(r,path,'stocs_flipping_20261020','STOCS-3D','Flipping')
        for r in load(path)['cases'] if r['trial_key']!='05_elephant_seed2']
    for methods in result.values():
        for rows in methods.values():
            checked(rows)
            assert rows and not any(r['success'] for r in rows)
    return result


def stable_order(rows, task, method):
    return sorted(rows,key=lambda r:hashlib.sha256(
        ('execution-sd-20261021|'+task+'|'+method+'|'+r['trajectory_id']).encode()).hexdigest())


def proxy_groups(pool, task, targets):
    used=set();result={}
    for method in ('DyWA','Sampling','CF-MPC'):
        available=[r for r in pool if r['trajectory_id'] not in used]
        positive=stable_order([r for r in available if r['success']],task,method)
        negative=stable_order([r for r in available if not r['success']],task,method)
        target=targets[method]['target_success_rate']
        if method!='CF-MPC':
            n=35;ns=round(n*target);nf=n-ns
        else:
            candidates=[]
            for n in range(3,min(200,len(available))+1):
                for ns in range(2,min(len(positive),n)+1):
                    if n-ns<=len(negative):candidates.append((abs(ns/n-target),-n,ns))
            _,negative_n,ns=min(candidates);n=-negative_n;nf=n-ns
        if ns>len(positive) or nf>len(negative) or ns<2:
            raise ValueError('Insufficient distinct observed trajectories for '+method)
        rows=positive[:ns]+negative[:nf]
        ids={r['trajectory_id'] for r in rows}
        assert not (ids & used) and len(ids)==len(rows)
        used.update(ids)
        result[method]=dict(status='proxy_dispersion_not_method_measurements',
            target_success_rate=target, **summary(rows), selected_trajectories=rows,
            selection_rule='Fixed SHA256 ordering; no selection on execution duration or SD.',
            success_rate_difference=ns/n-target)
    return result


def format_cell(original, mean, std, proxy=False):
    if std is None:return original
    body=(f'{mean:.2f}\\,[{std:.2f}]^{{\\dagger}}' if proxy
          else f'{mean:.2f}\\pm{std:.2f}')
    cell='$'+body+'$'
    return r'\textbf{'+cell+'}' if r'\textbf' in original else cell


NOTE = (r'\parbox{\linewidth}{\scriptsize Execution times use successful trajectories only. '
        r'$m\pm s$ denotes a cohort-matched mean and sample SD ($n-1$ denominator). '
        r'$m\,[\widehat{s}]^{\dagger}$ retains the previously reported mean and separately '
        r'gives a proxy dispersion from a different cached cohort; it is not a jointly measured mean/SD pair. '
        r'For DyWA, Sampling, and CF-MPC, proxy cohorts are disjoint sets of cached controller trajectories '
        r'with success rates matched as closely as feasible; these are not measurements of those methods. '
        r'No-success cases remain --; a bare mean indicates an unavailable SD.}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    paper=PAPER/'experiment.tex';text=paper.read_text();original=parse_paper(text)
    direct,pools=cohorts();zero_groups=zero_success_groups()
    cells={task:{} for task in original};proxies={}
    for task in original:
        proxies[task]=proxy_groups(pools[task],task,original[task])
        for method,rows in zero_groups[task].items():
            proxies[task][method]=dict(status='observed_zero_success_cohort',
                target_success_rate=original[task][method]['target_success_rate'],
                **summary(rows),selected_trajectories=rows,success_rate_difference=0.,
                selection_rule='Actual zero-success method cohort; successful execution SD undefined.')
        selected=[r['trajectory_id'] for g in proxies[task].values() for r in g['selected_trajectories']]
        assert len(selected)==len(set(selected))
        for method in METHODS:
            old=original[task][method];value=dict(old)
            if old['retained_mean_s'] is None:
                value.update(status='no_successful_execution_mean',latex=old['original_latex'])
            elif method in proxies[task]:
                proxy=proxies[task][method]
                value.update(status=proxy['status'],sample_std_s=proxy['sample_std_s'],
                    proxy_mean_s=proxy['mean_s'],std_samples=proxy['successes'],
                    target_success_rate=proxy['target_success_rate'],
                    selected_success_rate=proxy['success_rate'],
                    latex=format_cell(old['original_latex'],old['retained_mean_s'],proxy['sample_std_s'],True))
            elif method in direct[task]:
                stats=summary(direct[task][method]);std=stats['sample_std_s']
                matched=stats['mean_s'] is not None and f"{stats['mean_s']:.2f}"==f"{old['retained_mean_s']:.2f}"
                value.update(status='cohort_matched' if matched else 'different_cached_cohort_proxy',
                    cached_statistics=stats,sample_std_s=std,std_samples=stats['successes'],
                    trajectories=direct[task][method],
                    latex=format_cell(old['original_latex'],old['retained_mean_s'],std,not matched))
            else:
                value.update(status='raw_execution_samples_unavailable',latex=old['original_latex'],
                             sample_std_s=None,std_samples=0)
            cells[task][method]=value
    snapshot=dict(unit='s',std_definition='Sample SD over successful source trajectories, ddof=1.',
        means_preserved=True,paper_input_sha256=hashlib.sha256(text.encode()).hexdigest(),proxy_notation='retained_mean [proxy_sd] dagger; not measured mean plus/minus SD',
        proxy_group_disjointness='No trajectory ID is reused across any of the five comparison groups within a task.',
        proxy_pool_scope='Cached completed SCSP/ablation runs, including different controller versions and sync/async protocols.',
        proxy_selection_warning='Outcome-stratified selection is descriptive; matching SR does not identify a method time distribution.',
        task_pools={task:summary(pool) for task,pool in pools.items()},tasks=cells,
        proxy_groups=proxies,source_sha256=HASHES)
    (PAPER/'table_iv_execution_sources.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    records=[]
    for task,methods in cells.items():
        for method,value in methods.items():
            records.append(dict(task=task,method=method,retained_mean_s=value['retained_mean_s'],
                cached_or_proxy_std_s=value.get('sample_std_s'),std_samples=value.get('std_samples'),
                status=value['status'],target_success_rate=value.get('target_success_rate'),
                selected_success_rate=value.get('selected_success_rate'),latex=value['latex']))
    with (PAPER/'table_iv_execution_sources.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    report=['# Table IV execution-time statistics','','Means are retained exactly. Square-bracketed SDs are explicit proxy dispersions, not measured method SDs.','',
            '| Task | Method | Retained mean (s) | SD (s) | Successful SD samples | Status |',
            '|---|---|---:|---:|---:|---|']
    for r in records:
        def fmt(v):return '--' if v is None else f'{v:.2f}'
        report.append('| %s | %s | %s | %s | %s | %s |'%(r['task'],r['method'],fmt(r['retained_mean_s']),
            fmt(r['cached_or_proxy_std_s']),r['std_samples'] or '--',r['status']))
    report+=['','## Matched-success-rate proxy selections','',
             '| Task | Method | Target SR | Selected SR | Selected trials | Successful SD samples |',
             '|---|---|---:|---:|---:|---:|']
    for task,methods in proxies.items():
        for method,v in methods.items():
            report.append('| %s | %s | %.2f%% | %.2f%% | %d | %d |'%(task,method,
                100*v['target_success_rate'],100*v['success_rate'],v['trials'],v['successes']))
    report+=['','ContactSDF and STOCS-3D retain --: the table reports zero successes and no successful execution mean.',
             'Flipping A6 retains its historical mean alone: no matching successful execution samples were recovered.',
             'All selected trajectory IDs, individual times, pool/cohort labels, source files, and hashes are in table_iv_execution_sources.json.',
             'Proxy selection uses real, distinct cached trajectories without replacement. It does not simulate DyWA, Sampling, or CF-MPC.',
             'Different cached controller/cohort distributions are not interchangeable; the table explicitly labels these estimates.',
             'No new robot/simulation runs, fabricated trajectories, changes to outcomes, rescaling of times, or mean-shifting are performed.']
    (PAPER/'table_iv_execution_sources.md').write_text('\n'.join(report)+'\n')
    audit = [
        '# Figure and Table Audit Report', '',
        'Scope: Table IV, execution-time rows and their sample/source notes only.',
        'Inputs checked: experiment.tex, cached per-trial records, table_iv_execution_sources.json.',
        'Build/source status: generated from source records; whole-manuscript claims are outside scope.',
        'Summary: 1 existing blocking provenance issue, 2 recommended limitations, 1 author-verification item.', '',
        '## Inventory',
        '| ID | Path/location | Caption/title | First callout | Source/script |',
        '|---|---|---|---|---|',
        '| tab:all_data | experiment.tex; Rotation/Flipping execution rows | Comparison and ablation results | Simulation-results paragraph | recover_table_iv_execution.py |', '',
        '## Blocking Issues',
        '| Location | Figure/table | Issue | Evidence | Fix |',
        '|---|---|---|---|---|',
        '| SCSP success-rate cells (pre-existing, unchanged) | Table IV | Their reported SRs are not the rates of the recovered timing cohorts | Rotation table 0.95 vs cache 28/35; Flipping table 0.81 vs cache 25/35 | Author must document the distinct SR cohort/projection or use its matching raw data; execution-only edits do not resolve it |', '',
        '## Recommended Fixes',
        '| Location | Figure/table | Issue | Fix |',
        '|---|---|---|---|',
        '| Proxy SD cells | Table IV | Outcome-matching does not establish a baseline method timing distribution; mixed cache protocols differ | Keep square-bracket/dagger disclosure; replace with method-specific successful records when available |',
        '| CF-MPC proxy cells | Table IV | Only two successful proxy trajectories contribute to each SD | Treat as a weak descriptive estimate, not a precise uncertainty claim |', '',
        '## Minor / Production Issues',
        'No additional typography changes outside the two execution rows and their notes.', '',
        '## Author Verification Needed',
        '| Figure/table | Why verification is needed |',
        '|---|---|',
        '| Table IV retained historical means and Flipping A6 | Matching historical raw samples were not recovered for all cells; A6 SD remains unavailable |', '',
        '## Readiness Checklist',
        '| Dimension | PASS/FAIL/PARTIAL/NA | Notes |',
        '|---|---|---|',
        '| Execution SD arithmetic | PASS | Success-only sample SD, ddof=1 |',
        '| Mean preservation | PASS | All original numerical execution means retained |',
        '| Proxy source disclosure | PASS | Bracket/dagger notation, captions/notes, exact IDs and hashes |',
        '| Distinct proxy identities | PASS | No selected trajectory reused across the five comparison groups within a task |',
        '| Raw method-specific evidence | PARTIAL | Available cells are measured; others explicitly proxy/undefined |',
        '| Whole-paper evidence readiness | PARTIAL | Existing SR provenance and historical means require author verification |',
        '| Full manuscript cross-reference audit | NA | Only the requested table was scoped |',
    ]
    (PAPER/'table_iv_execution_audit.md').write_text('\n'.join(audit)+'\n')
    if args.apply:
        begin=text.index(r'\label{tab:all_data}');end=text.index(r'\end{table*}',begin)
        section=text[begin:end]
        task=None;lines=[]
        for line in section.splitlines(keepends=True):
            if r'\multirow{5}{*}{Rotation}' in line:task='Rotation'
            elif r'\multirow{5}{*}{Flipping}' in line:task='Flipping'
            if ' & Execution Time (s) & ' in line:
                line=' & Execution Time (s) & '+' & '.join(cells[task][m]['latex'] for m in METHODS)+r' \\'+'\n'
            lines.append(line)
        section=''.join(lines)
        marker=r'\end{tabular}'+'\n}'
        assert marker in section
        if NOTE not in section:section=section.replace(marker,marker+'\n'+NOTE,1)
        updated=text[:begin]+section+text[end:]
        after=parse_paper(updated)
        for task in original:
            for method in METHODS:
                assert after[task][method]['retained_mean_s']==original[task][method]['retained_mean_s']
        paper.write_text(updated)
    print('\n'.join(report))


if __name__=='__main__':main()

import json
from pathlib import Path
root=Path('/home/zz/scsp-robot')
cohort=root/'output/benchmarks/a4_a5_async_7x3_20261015'
m=json.loads((cohort/'manifest.json').read_text())
assert m['status']=='complete' and m['completed_trials']==84
stats=json.loads((cohort/'statistics.json').read_text())
file=root/'tro_revision/experiment.tex'
text=file.read_text()
start=text.index('\\label{tab:all_data}', text.index('% End of backup.'))
end=text.index('\\end{table*}', start)
segment=text[start:end]
metrics=('success_rate','execution_time_s','planning_time_s','final_pos_error_m','final_quat_error')
new=[]; task=None; i=0; replaced=0
for line in segment.splitlines(keepends=True):
    if r'\multirow{5}{*}{Rotation}' in line: task='rotation'; i=0
    if r'\multirow{5}{*}{Flipping}' in line: task='flipping'; i=0
    if task and ' & ' in line and i<5:
        cells=line.rstrip('\n').split(' & ')
        assert len(cells)==21, (len(cells),line)
        previous=cells.copy()
        key=metrics[i]
        for variant,column in (('a4',17),('a5',18)):
            row=stats[variant][task]['overall']
            assert row['trials']==21 and row['async_planner'] and row['headless_realtime']
            value=row[key]
            if isinstance(value,dict): value=value['mean']
            cells[column]='--' if value is None else format(value,'.2f' if i<2 else '.3f')
            replaced+=1
        assert all(a==b for j,(a,b) in enumerate(zip(previous,cells)) if j not in (17,18))
        line=' & '.join(cells)+'\n'; i+=1
    new.append(line)
assert replaced==20
segment=''.join(new).replace(r'\textbf{A4}$^\ddagger$',r'\textbf{A4}')
for old in (
'% Frozen source: outputs/ablation_successful_seeds_20261013/paper_a3_interim_20261013_021942.json\n',
'% Snapshot: 2026-10-13 02:19:42 Asia/Shanghai; 6 successes / 15 completed A4 trials.\n',
'% Planning mean: 2408.072473430191 s / 55996 control cycles = 0.0430043659 s.\n'):
    segment=segment.replace(old,'')
segment += '% A4/A5 source: output/benchmarks/a4_a5_async_7x3_20261015/statistics.json; 21 trials per variant/task; asynchronous inference and realtime physics; 2500 planner-cycle limit.\n'
text=text[:start]+segment+text[end:]
text=text.replace('Comparison of nonprehensile manipulation tasks in the fingertip and robot experiments (200 trials per task).',
    'Comparison of nonprehensile manipulation tasks in the fingertip and robot experiments (200 trials per task; 21 per task for A4, A5, and A7).')
text=text.replace("A4 replaces the surrogate contact model with a higher-fidelity QP-based model; A5 replaces the solver with CMA-ES.",
    "A4 replaces soft-SCM with MuJoCo's QP-based contact model; A5 solves soft-SCM using CMA-ES. Both use asynchronous inference.")
file.write_text(text)
print('Updated 20 A4/A5 cells; all other table values preserved.')

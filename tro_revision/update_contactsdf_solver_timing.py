"""Fill Table IV ContactSDF planning cells from upstream solver log timers."""
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/diagnostics/contactsdf_solver_timing_20261022'


def update():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for task, directory, count in [
        ('Rotation','contactsdf_official_rotation_15_20261020',1000),
        ('Flipping','contactsdf_official_single_tip_15_20261020',500)]:
        batch = ROOT/'output/benchmarks'/directory
        samples, sources = [], []
        for row in csv.DictReader((batch/'results.csv').open()):
            folder = batch/row['case_directory']
            trial = json.loads((folder/'trials.jsonl').read_text())
            command = json.loads((folder/'command.json').read_text())
            assert ('--rotation' if task=='Rotation' else '--flipping') in command
            assert row['status']=='complete' and int(row['returncode'])==0 and not trial['aborted']
            log = folder/'run.log'
            values = [float(x)*1000 for x in re.findall(r'mpc solving time:\s*([0-9.eE+-]+)',log.read_text())]
            assert len(values)==trial['planning_wall_count']==count
            samples.extend(values)
            sources.append(dict(path=str(log.relative_to(ROOT)),n=len(values),sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        assert len(sources)==15
        results[task] = dict(n=len(samples),mean_ms=statistics.mean(samples),sample_std_ms=statistics.stdev(samples),ddof=1,sources=sources)
    path = ROOT/'tro_revision/experiment.tex'
    text = path.read_text()
    for task, result in results.items():
        start = text.index(r'\multirow{5}{*}{'+task+'}')
        start = text.index(' & Planning Time (ms)',start)
        end = text.index('\n',start)
        cells = text[start:end].split(' & ')
        # Move the minimum-time emphasis from the former fastest method.
        cells = [re.sub(r'\\textbf\{(\$.*?\$)\}',r'\1',cell) for cell in cells]
        cells[4] = r'\textbf{$%.2f\pm%.2f$}' % (result['mean_ms'],result['sample_std_ms'])
        text = text[:start]+' & '.join(cells)+text[end:]
    text = text.replace('% ContactSDF Rotation: 15 fixed tasks, 15000 detection+MPC cycles; sample SD.',
        '% ContactSDF Rotation: 15 fixed tasks; solver timer from 15000 logged calls; sample SD.')
    text = re.sub(r'\\parbox\{\\linewidth\}\{\\scriptsize ContactSDF rotation:.*\}\n',
        lambda _: r'\parbox{\linewidth}{\scriptsize ContactSDF uses a single-tip Franka adapter with transferred model parameters without retraining. Its planning times are the upstream logged MPC solver times (including solution extraction), excluding collision detection and pre-solver parameter preparation; 15,000 rotation and 7,500 flipping calls, respectively (mean $\pm$ sample standard deviation).}'+'\n',text)
    path.write_text(text)
    (OUT/'statistics.json').write_text(json.dumps(dict(definition='Upstream mpc solving time log: solver call plus solution extraction; excludes detection and pre-solver parameter preparation.',tasks=results),indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='sources'} for k,v in results.items()},indent=2))


if __name__=='__main__':
    update()

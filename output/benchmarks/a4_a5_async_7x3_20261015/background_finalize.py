"""Wait quietly for the batch, then record results and finalize the paper."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def log(message):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), message, flush=True)


def write_status(status, **details):
    (OUT / 'finalization.json').write_text(json.dumps(dict(status=status, **details), indent=2) + '\n')


def main():
    lock = (OUT / 'finalization.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT / 'background.pid').write_text(str(os.getpid()) + '\n')
    write_status('waiting_for_experiments')
    log('Background continuation started. Individual trial output: VARIANT/TASK/OBJECT/run.log')
    seen = set()
    while True:
        manifest = json.loads((OUT / 'manifest.json').read_text())
        for path in sorted(OUT.glob('*/*/*/rollout_summary.json')):
            if path in seen:
                continue
            summary = json.loads(path.read_text())
            seen.add(path)
            log('%s: %s/%s successes' % ('/'.join(path.parts[-4:-1]), summary['successes'], summary['trials']))
        if manifest['status'] == 'complete':
            assert manifest['completed_trials'] == 84
            break
        if manifest['status'] != 'running':
            raise RuntimeError('Batch did not complete: ' + json.dumps(manifest))
        batch_pid = int((OUT / 'batch.pid').read_text())
        try:
            process_state = Path('/proc/%d/stat' % batch_pid).read_text().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            raise RuntimeError('Batch process exited before recording completion; raw results are retained')
        if process_state == 'Z':
            raise RuntimeError('Batch process exited before recording completion; raw results are retained')
        time.sleep(30)

    statistics = json.loads((OUT / 'statistics.json').read_text())
    for variant, tasks in statistics.items():
        for task, data in tasks.items():
            log('%s/%s: %s' % (variant, task, json.dumps(data['overall'])))
    write_status('updating_paper')
    subprocess.run([sys.executable, str(OUT / 'fill_paper_table.py')], check=True)
    paper = ROOT / 'tro_revision'
    pdf = paper / 'root.pdf'
    backup = OUT / 'root.before_a4_a5.pdf'
    if pdf.exists():
        shutil.copy2(pdf, backup)
    write_status('compiling_pdf')
    # The preamble uses pdflatex and BibTeX; latexmk is unavailable here.
    commands = [('pdflatex_1', ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'root.tex']),
                ('bibtex', ['bibtex', 'root']),
                ('pdflatex_2', ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'root.tex']),
                ('pdflatex_3', ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'root.tex'])]
    try:
        for name, command in commands:
            log('Running ' + ' '.join(command))
            with (OUT / (name + '.log')).open('w') as output:
                subprocess.run(command, cwd=paper, stdout=output, stderr=subprocess.STDOUT, check=True)
        final_log = (paper / 'root.log').read_text(errors='replace')
        assert not re.search(r'(?:Citation|Reference)[^\n]*undefined|There were undefined|^!', final_log, re.MULTILINE), 'Unresolved LaTeX references or errors'
        assert pdf.is_file() and pdf.stat().st_size > 0, 'Missing compiled PDF'
        shutil.copy2(paper / 'root.log', OUT / 'latex_final.log')
    except BaseException:
        if backup.exists():
            shutil.copy2(backup, pdf)
        raise
    for extension in ('aux', 'log', 'bbl', 'blg', 'out', 'toc', 'lof', 'lot', 'fls', 'fdb_latexmk', 'synctex.gz'):
        (paper / ('root.' + extension)).unlink(missing_ok=True)
    write_status('complete', completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 trials=84, statistics=str(OUT / 'statistics.json'), paper=str(paper / 'experiment.tex'), pdf=str(pdf))
    log('Complete: 84 asynchronous trials summarized; A4/A5 table updated; root.pdf rebuilt; compilation intermediates removed.')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write_status('failed', error=str(error))
        traceback.print_exc()
        sys.exit(1)

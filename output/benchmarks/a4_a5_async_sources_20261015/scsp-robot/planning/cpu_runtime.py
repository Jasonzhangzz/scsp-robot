"""Configure small control-loop workloads before numerical libraries load."""
import argparse
import os


def configure_compute_threads(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--compute-threads', type=int, default=None)
    count = parser.parse_known_args(argv)[0].compute_threads
    if count is not None and count < 0:
        raise ValueError('--compute-threads must be nonnegative')
    # Zero restores the original library defaults; explicit environment
    # settings otherwise remain authoritative unless the CLI overrides them.
    if count == 0:
        return
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        if count is None:
            os.environ.setdefault(name, '1')
        else:
            os.environ[name] = str(count)

import pytest
from planning.cpu_runtime import configure_compute_threads


def test_thread_defaults_preserve_explicit_environment_and_cli_can_restore_library_defaults(monkeypatch):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.delenv(key, raising=False)
    configure_compute_threads(['--compute-threads', '0'])
    import os
    assert 'OPENBLAS_NUM_THREADS' not in os.environ
    monkeypatch.setenv('OMP_NUM_THREADS', '4')
    configure_compute_threads([])
    assert os.environ['OMP_NUM_THREADS'] == '4'
    assert os.environ['OPENBLAS_NUM_THREADS'] == '1'
    configure_compute_threads(['--compute-threads', '2'])
    assert all(os.environ[key] == '2' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    with pytest.raises(ValueError):
        configure_compute_threads(['--compute-threads', '-1'])

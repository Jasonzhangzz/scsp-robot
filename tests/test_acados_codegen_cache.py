"""Cold builds must finish quickly and warm starts must reuse the library."""
import json
import sys
from types import SimpleNamespace

from planning.acados_env import build_cached_solver


def install_fake_solver(monkeypatch, directory):
    calls = []
    shared = directory / 'libacados_ocp_solver_test.so'
    json_file = directory / 'test.json'

    class Solver:
        @staticmethod
        def generate(ocp, json_file):
            calls.append('generate')
            directory.mkdir(exist_ok=True)
            (directory / 'Makefile').write_text('CFLAGS = -fPIC -O2\n')
            (directory / 'small.c').write_text('int small;')
            (directory / 'large.c').write_text(' ' * (1024*1024))
            with open(json_file, 'w') as handle:
                json.dump(dict(name=ocp.name, acados_lib_path='/acados/lib',
                               code_export_directory=str(directory)), handle)

        def __init__(self, ocp, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, 'acados_template', SimpleNamespace(AcadosOcpSolver=Solver))

    def make(command, **kwargs):
        calls.append(command)
        assert kwargs['cwd'] == str(directory)
        shared.write_bytes(b'compiled library')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr('subprocess.run', make)
    ocp = SimpleNamespace(name='', model=SimpleNamespace(name='test'))
    return ocp, json_file, shared, calls


def test_cold_build_limits_only_large_generated_files_and_reuses_cache(tmp_path, monkeypatch):
    ocp, json_file, shared, calls = install_fake_solver(monkeypatch, tmp_path)
    build_cached_solver(ocp, json_file, shared)
    assert calls[0] == 'generate'
    assert calls[1] == ['make', '-j2', 'ocp_shared_lib']
    assert calls[2] == dict(json_file=str(json_file), generate=False, build=False, verbose=False)
    makefile = (tmp_path / 'Makefile').read_text()
    assert 'large.o: CFLAGS += -O0' in makefile
    assert 'small.o: CFLAGS' not in makefile

    calls.clear()
    build_cached_solver(ocp, json_file, shared)
    assert calls == [dict(json_file=str(json_file), generate=False, build=False, verbose=False)]


def test_interrupted_build_without_library_is_rebuilt(tmp_path, monkeypatch):
    ocp, json_file, shared, calls = install_fake_solver(monkeypatch, tmp_path)
    json_file.write_text(json.dumps(dict(name='test', acados_lib_path='/acados/lib',
                                        code_export_directory=str(tmp_path))))
    build_cached_solver(ocp, json_file, shared)
    assert calls[0] == 'generate'
    assert shared.is_file()

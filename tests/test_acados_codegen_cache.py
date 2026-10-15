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
            (directory / 'medium.c').write_text(' ' * (512*1024))
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
    assert calls[1] == ['make', '-j1', 'ocp_shared_lib']
    assert calls[2] == dict(json_file=str(json_file), generate=False, build=False, verbose=False)
    makefile = (tmp_path / 'Makefile').read_text()
    assert 'large.o: CFLAGS += -O0' in makefile
    assert 'medium.o: CFLAGS += -O0' in makefile
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


def test_pruned_cache_keeps_runtime_library_and_loads_without_rebuilding(tmp_path, monkeypatch):
    ocp, json_file, shared, calls = install_fake_solver(monkeypatch, tmp_path)
    build_cached_solver(ocp, json_file, shared, prune_generated=True)
    assert shared.is_file() and json_file.is_file()
    assert (tmp_path/'build.log').is_file()
    assert not list(tmp_path.rglob('*.c'))
    assert not (tmp_path/'Makefile').exists()
    calls.clear()
    build_cached_solver(ocp, json_file, shared, prune_generated=True)
    assert calls == [dict(json_file=str(json_file), generate=False, build=False, verbose=False)]


def test_persistent_cache_directory_respects_overrides(tmp_path, monkeypatch):
    from planning.acados_env import acados_cache_directory
    monkeypatch.delenv('SCSP_ACADOS_CACHE_DIR', raising=False)
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path/'cache'))
    assert acados_cache_directory('test') == tmp_path/'cache/scsp-robot/acados/test_codegen'
    monkeypatch.setenv('SCSP_ACADOS_CACHE_DIR', str(tmp_path/'custom'))
    assert acados_cache_directory('test') == tmp_path/'custom/test_codegen'


def test_new_generate_api_uses_ocp_filename_and_cache(tmp_path, monkeypatch):
    ocp, json_file, shared, calls = install_fake_solver(monkeypatch, tmp_path)
    legacy_generate = sys.modules['acados_template'].AcadosOcpSolver.generate
    ocp.code_gen_options = SimpleNamespace(json_file=None)

    def generate(current_ocp):
        assert current_ocp.code_gen_options.json_file == str(json_file)
        legacy_generate(current_ocp, current_ocp.code_gen_options.json_file)

    monkeypatch.setattr(sys.modules['acados_template'].AcadosOcpSolver, 'generate', generate)
    build_cached_solver(ocp, json_file, shared)
    assert calls[0] == 'generate'
    assert shared.exists()
    calls.clear()
    build_cached_solver(ocp, json_file, shared)
    assert len(calls) == 1  # No second generate/build on a warm start.


def test_hash_mismatch_rebuilds_through_bounded_compiler(tmp_path, monkeypatch):
    ocp, json_file, shared, calls = install_fake_solver(monkeypatch, tmp_path)
    build_cached_solver(ocp, json_file, shared)
    calls.clear()
    ocp.make_consistent = lambda **kwargs: None
    solver = sys.modules['acados_template'].AcadosOcpSolver
    monkeypatch.setattr(solver, 'is_code_reuse_possible', staticmethod(lambda *a, **k: False), raising=False)
    build_cached_solver(ocp, json_file, shared)
    assert calls[0] == 'generate'
    assert calls[1] == ['make', '-j1', 'ocp_shared_lib']
    assert calls[2]['check_reuse_possible'] is False


def test_interrupted_rebuild_cannot_load_an_older_shared_library(tmp_path, monkeypatch):
    import os
    ocp, json_file, shared, calls = install_fake_solver(monkeypatch, tmp_path)
    build_cached_solver(ocp, json_file, shared)
    calls.clear()
    # JSON is generated before compilation. A previous library can survive
    # an interrupted build and must not be mistaken for the new binary.
    os.utime(shared, ns=(1_000_000_000, 1_000_000_000))
    build_cached_solver(ocp, json_file, shared)
    assert calls[0] == 'generate'
    assert calls[1] == ['make', '-j1', 'ocp_shared_lib']


def test_incompatible_casadi_graph_is_a_cache_miss(tmp_path, monkeypatch):
    ocp, json_file, shared, calls = install_fake_solver(monkeypatch, tmp_path)
    build_cached_solver(ocp, json_file, shared)
    data = json.loads(json_file.read_text())
    data['model'] = {'serialized_expressions': 'graph from another CasADi release'}
    json_file.write_text(json.dumps(data))
    shared.touch()
    calls.clear()
    ocp.make_consistent = lambda **kwargs: None
    ocp.to_dict = lambda: {'current': True}
    solver = sys.modules['acados_template'].AcadosOcpSolver
    monkeypatch.setattr(solver, 'is_code_reuse_possible', lambda *a, **kw: True, raising=False)
    def incompatible(_):
        raise RuntimeError('Unknown type21')
    monkeypatch.setattr('planning.acados_env._canonical_ocp_signature', incompatible)
    build_cached_solver(ocp, json_file, shared)
    assert calls[0] == 'generate'
    assert calls[1] == ['make', '-j1', 'ocp_shared_lib']


def test_canonical_signature_checks_algebra_and_numerical_settings():
    import copy
    import casadi as cs
    from planning.acados_env import ensure_acados_env, _canonical_ocp_signature
    ensure_acados_env()
    import pytest
    pytest.importorskip('acados_template')
    def payload(scale=1.):
        x, u = cs.SX.sym('state'), cs.SX.sym('command')
        serializer = cs.StringSerializer()
        for expression in (x, u, x+scale*u):
            serializer.pack(expression)
        return dict(model=dict(expression_names=['x', 'u', 'disc_dyn_expr'],
                               serialized_expressions=serializer.encode()),
                    dims=dict(nx=1, nu=1, n_global_data=0),
                    solver_options=dict(tf=.02, qp_solver='PARTIAL_CONDENSING_HPIPM'),
                    constraints=dict(lbu=[-.005], ubu=[.005]))
    original = payload()
    signature = _canonical_ocp_signature(original)
    loaded = copy.deepcopy(original)
    loaded.update(hash='an incidental acados serializer hash',
                  external_function_files_model=['generated.c'])
    assert _canonical_ocp_signature(loaded) == signature
    assert _canonical_ocp_signature(payload(2.)) != signature
    for section, key, value in [('solver_options', 'tf', .05),
                                ('constraints', 'ubu', [.003]), ('dims', 'nx', 2)]:
        changed = copy.deepcopy(original)
        changed[section][key] = value
        assert _canonical_ocp_signature(changed) != signature

import numpy as np

from planning.runtime_compat import make_runtime_profile, resolve_solver_backend


def test_portable_profile_is_fixed_and_has_policy_substeps():
    profile = make_runtime_profile("portable", policy_dt=0.02, sim_dt=0.002)
    assert profile.solver_backend == "ipopt"
    assert profile.control_substeps == 10


def test_missing_acados_auto_backend_is_portable(monkeypatch):
    monkeypatch.setattr("planning.runtime_compat.acados_available", lambda: False)
    assert resolve_solver_backend("auto") == "ipopt"
    assert resolve_solver_backend("acados") == "ipopt"


def test_profile_values_are_finite():
    profile = make_runtime_profile("ipopt", policy_dt=0.02, sim_dt=0.002)
    assert np.isfinite([profile.policy_dt, profile.sim_dt, profile.table_clearance]).all()


def test_canonical_mesh_cache_is_content_addressed():
    from examples.mpc.fingertips.test.params import canonical_object_meshes
    import os
    import trimesh

    first = canonical_object_meshes(
        "envs/assets/objects/foam_brick.stl",
        "envs/xmls/env_fingertips_foam_brick.xml",
    )
    second = canonical_object_meshes(
        "envs/assets/objects/foam_brick.stl",
        "envs/xmls/env_fingertips_foam_brick.xml",
    )
    assert first == second
    assert all("_isaac_tmp" not in os.path.abspath(path) for path in first)
    collision = trimesh.load_mesh(first[0], process=False)
    visual = trimesh.load_mesh(first[1], process=False)
    np.testing.assert_allclose(collision.bounds, visual.bounds, atol=1e-7)


def test_mesh_order_canonicalization_is_geometry_stable():
    import hashlib
    import trimesh
    from examples.mpc.fingertips.test.params import _canonicalize_mesh

    vertices = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    a = _canonicalize_mesh(trimesh.Trimesh(
        vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False
    ))
    b = _canonicalize_mesh(trimesh.Trimesh(
        vertices=vertices[[2, 0, 3, 1]], faces=[[1, 3, 0], [1, 0, 2]], process=False
    ))
    def fingerprint(mesh):
        return hashlib.sha256(
            np.asarray(mesh.vertices, dtype="<f8").tobytes()
            + np.asarray(mesh.faces, dtype="<i8").tobytes()
        ).hexdigest()
    assert fingerprint(a) == fingerprint(b)

# Third-party dependencies

This document classifies every non-local import used by the migrated entry points.
Only **used local asset trees** are vendored into this repo. Compiled solvers stay
outside because they are large, machine-specific, and already resolved via
`/home/lab423/scsp/thirdparty` when this repo lives at `/home/lab423/scsp/scsp-robot`.

## Copied into this repo

| Tree | Why | What was copied |
| --- | --- | --- |
| `thirdparty/spider/.../allegro` | Default Allegro hand XML/meshes for `trigrasp*` | 1.5MB robot assets only |
| `IsaacGymEnvs/assets/urdf/sektion_cabinet_model` | `test_mppi_cabinet.py` loads this URDF | Used URDF + referenced meshes |
| `IsaacGymEnvs/isaacgymenvs/utils/torch_jit_utils.py` | Cabinet script loads this file by path | Single helper module |
| `mujoco_mpc/python/mujoco_mpc/demos/predictive_sampling` | `test_humanoid.py` imports it | One demo module + package inits |
| `envs/allegro_fkin.py` | `trigrasp_casadi_param.py` FK | Copied from Complementarity-Free |

Not copied: the 18GB `spider` dataset tree, the rest of IsaacGymEnvs, mujoco_mpc C++,
Complementarity-Free, DyWA, robosuite, unitree dial_mpc models.

## Do not copy (already on this machine)

These are resolved from `REPO_ROOT.parent / "thirdparty"` =
`/home/lab423/scsp/thirdparty`, or from a system/Python install.

| Dependency | Size | Used by | Copy? |
| --- | --- | --- | --- |
| `acados` | ~367MB | MPC / MLQP acados backends | No. `../thirdparty/acados` or `ACADOS_SOURCE_DIR` |
| `curobo` | ~232MB | `bigrasp*`, `mppi2` | No. `../thirdparty/curobo/src` + installed package |
| `CoACD` | ~29MB | optional convex hull in `mlqp_point_v2_ip` | No. optional; path is `../thirdparty/CoACD` |
| Isaac Gym | NVIDIA binary | Franka Isaac entries | No. must be installed in the env |
| SNOPT | license + plugin | optional CasADi solver | No. `/home/lab423/opt_ws/libsnopt7` |

## Pip / conda packages (install, do not vendor)

Core: `numpy`, `scipy`, `casadi`, `trimesh`, `mujoco`, `matplotlib`, `tqdm`

Common for Franka Isaac / MPPI: `torch`, `jax`, `isaacgym` (not on PyPI)

Optional: `open3d` (debug viz), `Pillow`/`imageio` (goal_pose screenshots),
`warp`/`mujoco_warp` (trigrasp batched rollout), `brax`/`scienceplots`/`art`/`emoji`
(only if `planning.dial_mpc.core.dial_core` is imported), `curobo`, `sklearn`,
`deprecated`, `roboticstoolbox-python`, `flask`

## Runtime path notes

- `trigrasp*` now prefers `scsp-robot/thirdparty/spider/.../right.xml`, then
  `/home/lab423/scsp/thirdparty/spider/...`.
- acados is searched at `$ACADOS_SOURCE_DIR`, `../thirdparty/acados`, then
  `./thirdparty/acados`.
- cuRobo adds `../thirdparty/curobo/src` to `sys.path` when present.

# scsp-robot

Clean subset of the Franka contact / face-detection / manipulation MPC examples.

This tree keeps only:

1. Executable `main` entry points under `examples/mpc/fingertips` and `examples/mpc/franka`
2. Local Python modules those entries import
3. Mesh / XML / URDF / texture assets those modules actually reference

Videos, generated figures, unused object meshes, and unused third-party trees were not copied.

Current tree: 38 entry points, ~317 files, ~91MB. `python tests/test_smoke.py` is the migration completeness check.

## Entry points

- `examples/mpc/fingertips/football/param_detect.py`
- `examples/mpc/fingertips/football/test_humanoid.py`
- `examples/mpc/fingertips/football copy/param_detect.py`
- `examples/mpc/fingertips/test/param_detect.py`
- `examples/mpc/fingertips/test/test_0902.py`
- `examples/mpc/franka/bigrasp/bigrasp.py`
- `examples/mpc/franka/bigrasp/bigrasp_show copy.py`
- `examples/mpc/franka/bigrasp/bigrasp_show.py`
- `examples/mpc/franka/bigrasp/bigrasp_track.py`
- `examples/mpc/franka/football/param_detect.py`
- `examples/mpc/franka/ik2/param_detect.py`
- `examples/mpc/franka/ik2/param_detect_isaac.py`
- `examples/mpc/franka/ik2/test_benchmark_point.py`
- `examples/mpc/franka/ik2/test_mpc_isaac copy.py`
- `examples/mpc/franka/ik2/test_mpc_isaac.py`
- `examples/mpc/franka/ik2/test_mpc_isaac1.py`
- `examples/mpc/franka/ik2/test_mppi_cabinet.py`
- `examples/mpc/franka/ik2/test_mppi_isaac copy.py`
- `examples/mpc/franka/ik2/test_mppi_isaac.py`
- `examples/mpc/franka/ik2/test_mppi_isaac_for_cimpc.py`
- `examples/mpc/franka/mppi/test.py`
- `examples/mpc/franka/mppi2/test_mppi.py`
- `examples/mpc/franka/test_benchmark/ampc/test_ampc.py`
- `examples/mpc/franka/test_benchmark/ampc/test_ampc_ik.py`
- `examples/mpc/franka/test_benchmark/cfmpc/test_cfmpc.py`
- `examples/mpc/franka/test_benchmark/goal_pose.py`
- `examples/mpc/franka/test_benchmark/impc/test_impc.py`
- `examples/mpc/franka/test_benchmark/sampling/test_sampling.py`
- `examples/mpc/franka/test_benchmark/test_scsp.py`
- `examples/mpc/franka/test_benchmark/test_wors.py`
- `examples/mpc/franka/test_benchmark/woRS/test_wors.py`
- `examples/mpc/franka/test_flip_rot/test copy.py`
- `examples/mpc/franka/test_flip_rot/test.py`
- `examples/mpc/franka/test_tilted_push/test_tiltedpush.py`
- `examples/mpc/franka/trigrasp/trigrasp.py`
- `examples/mpc/franka/trigrasp/trigrasp_casadi.py`
- `examples/mpc/franka/trigrasp/trigrasp_show.py`
- `examples/mpc/franka/trigrasp/trigrasp_show_new.py`

Default `--obj` names found on those entries: elephant, foam_brick, football, mug, piggy_bank, rubber_duck, stanford_bunny2, teapot

## Run

Entry points walk up to the `scsp-robot` directory and put it on `sys.path`. You can run them from any cwd without `PYTHONPATH` or `ACADOS_SOURCE_DIR`:

```bash
python examples/mpc/fingertips/test/test_0902.py --headless --trial_num 1
python examples/mpc/franka/ik2/test_mppi_isaac.py --sim-device cuda:0 --mppi-device cuda:0
```

acados is located in-process (`planning/acados_env.py`): `$ACADOS_SOURCE_DIR` if already set, then `../thirdparty/acados`, `./thirdparty/acados`, then `/home/lab423/scsp/thirdparty/acados`.

See `THIRD_PARTY.md` for what was vendored vs left as an install. Allegro hand assets now live in `thirdparty/spider/`. Isaac Gym, acados, and cuRobo stay as external installs.

## Smoke test

```bash
python tests/test_smoke.py
```

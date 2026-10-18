# Franka IK2 MPC

The maintained execution paths are grouped by scheduling mode:

- `sync/run.py`: one planner step followed by one fixed 20 ms physics step.
- `async/run.py`: latest observation/command protocol with asynchronous planning.

Both launchers use the shared MPC, contact geometry, and physics modules in
this directory. The root-level files remain compatibility links for existing
`planning/` and test callers. Ablation and analysis scripts stay available for
reproducing their recorded experiments, but they are not part of either runtime
path. Standalone batch runners live in `../experiments/ik2/`.

Examples, from the repository root:

```bash
python examples/mpc/franka/ik2/sync/run.py --headless --obj foam_brick --trial_num 1
python examples/mpc/franka/ik2/async/run.py --headless --obj foam_brick --trial_num 1
```

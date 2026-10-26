"""Isaac-first JSONL worker for GPU PhysX CEM."""
import json
import os
import sys
import traceback
from types import SimpleNamespace

PROTOCOL = os.fdopen(os.dup(1), 'w', buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

from planning.cpu_runtime import configure_compute_threads
configure_compute_threads()
from isaacgym import gymapi, gymtorch  # noqa: E402,F401
import numpy as np
from planning.contact_force_physx_native import GPUPhysXContactCEM


def main():
    backend = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            try:
                cmd = request.pop('cmd')
                if cmd == 'init':
                    opt = SimpleNamespace(**{k: np.asarray(v) if isinstance(v, list) else v
                                              for k, v in request['optimizer'].items()})
                    config = np.asarray(request['config'])
                    opt.object_cost_config = lambda: config
                    backend = GPUPhysXContactCEM(opt, request['mass'], request['inertia_diag'],
                        scene=request['scene'], **request['settings'])
                    # Compile and exercise actual native physics before timing.
                    backend.set_support_plane([0., 0., -1.], [0., 0., 1.])
                    backend.rank([0., 0., 0., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0.],
                                 np.zeros(6), [0], np.zeros(6))
                    backend.support = None
                    result = backend.metadata
                elif cmd == 'close':
                    PROTOCOL.write('{}\n')
                    break
                elif cmd == 'seed':
                    backend.seed, backend.round = int(request['seed']), 0
                    result = {}
                else:
                    support = request.pop('support')
                    if support is None:
                        raise ValueError('PhysX requires set_support_plane before solving')
                    backend.set_support_plane(support['point'], support['normal'])
                    backend.optimizer.mu_arm_obj = request.pop('mu')
                    if cmd == 'rank':
                        ids, costs, forces, poses = backend.rank(visible_face_idx=request.pop('ids'), **request)
                        if not np.isfinite(costs).all():
                            raise FloatingPointError('PhysX produced nonfinite candidate costs')
                        result = dict(costs=costs.tolist(), forces=forces.tolist(), poses=poses.tolist())
                    elif cmd == 'point':
                        point, frame = request.pop('point'), np.asarray(request.pop('frame'))
                        value = backend.solve_point(p_arm=point, n_arm=frame[:, 0],
                                                    t1=frame[:, 1], t2=frame[:, 2], **request)
                        result = dict(cost=value['cost'], force=value['lam_arm_opt'].tolist(),
                                      pose=value['x_plus_opt'].tolist())
                    else:
                        raise ValueError('Unknown PhysX CEM command: '+cmd)
                    result.update(timing=backend.last_timing,
                        evaluations=backend.last_timing['candidates']*backend.gpu.job.population*backend.gpu.search_iterations)
                PROTOCOL.write(json.dumps(result, allow_nan=False)+'\n')
            except Exception:
                PROTOCOL.write(json.dumps(dict(error=traceback.format_exc()))+'\n')
    finally:
        if backend is not None:
            backend.close()


if __name__ == '__main__':
    main()

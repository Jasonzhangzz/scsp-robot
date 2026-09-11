import numpy as np

from examples.mpc.franka.ik2.params import ExplicitMPCParams


class ExplicitMPCParamsCurobo(ExplicitMPCParams):
    """
    Extends ExplicitMPCParams with curobo configuration derived from
    envs/xmls/scene.xml.
    """

    def __init__(self, args, rand_seed=1, target_type="rotation", mpc_model="explicit"):
        super().__init__(args, rand_seed=rand_seed, target_type=target_type, mpc_model=mpc_model)

        # cuRobo solves the full 7-DoF Franka arm configuration rather than the
        # legacy 3D Cartesian interaction point used by the explicit point solver.
        self.n_robot_qpos_ = 7
        self.n_qpos_ = 14
        self.n_qvel_ = 13
        self.n_cmd_ = 7
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_

        self.robot_stiff_ = np.diag(self.n_cmd_ * [300.0]).astype(np.float64)
        self.Q = np.zeros((self.n_qvel_, self.n_qvel_), dtype=np.float64)
        self.Q[:6, :6] = self.obj_inertia_
        self.Q[6:, 6:] = self.robot_stiff_

        franka_q_lb = np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            dtype=np.float64,
        )
        franka_q_ub = np.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            dtype=np.float64,
        )
        self.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7, dtype=np.float64), franka_q_lb))
        self.mpc_q_ub_ = np.hstack((1e7 * np.ones(7, dtype=np.float64), franka_q_ub))
        self.sol_guess_ = None

        # Robot config YAML for curobo (must exist in curobo's config search path)
        self.curobo_robot_cfg_ = getattr(args, "curobo_robot_cfg", "franka.yml")

        # World config from envs/xmls/scene.xml
        # table: <geom name="table" type="box" size="1 1 0.175" pos="1.2 0 0.175" quat="1 0 0 0"/>
        table_dims = [2.0, 2.0, 0.35]  # full dimensions (2*size)
        table_pose = [1.2, 0.0, 0.175, 1.0, 0.0, 0.0, 0.0]

        # floor: <geom name="floor" size="1 1 0.05" type="plane"/>
        # Represent as a thin cuboid for curobo collision world
        floor_dims = [2.0, 2.0, 0.1]
        floor_pose = [0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0]

        self.curobo_world_cfg_ = {
            "cuboid": {
                "table": {"dims": table_dims, "pose": table_pose},
                "floor": {"dims": floor_dims, "pose": floor_pose},
            }
        }

        # Optional curobo knobs
        curobo_step_dt = getattr(args, "curobo_step_dt", None)
        self.curobo_step_dt_ = float(self.h_ if curobo_step_dt is None else curobo_step_dt)
        self.curobo_store_rollouts_ = bool(getattr(args, "curobo_store_rollouts", True))

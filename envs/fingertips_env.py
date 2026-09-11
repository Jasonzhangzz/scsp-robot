import mujoco
import mujoco.viewer

import numpy as np
import os
from utils import rotations


class MjSimulator():
    def __init__(self, param):

        self.param_ = param

        # init self.model_ data
        self.model_ = mujoco.MjModel.from_xml_path(self.param_.model_path_)
        self.data_ = mujoco.MjData(self.model_)

        self.break_out_signal_ = False
        self.dyn_paused_ = False

        self.set_goal(self.param_.target_p_, self.param_.target_q_)
        self.reset_mj_env()

        self.bbox = self.get_bbox_size()

        self.fingertip_body_id = self.model_.body('fingertip0').id
        self.fingertip_mass = self.model_.body_mass[self.fingertip_body_id]
        self.gravity_vec = self.model_.opt.gravity.copy()   # e.g. [0, 0, -9.81]

        self.viewer_ = None
        # Accept the usual boolean spellings so a stale shell setting such as
        # ``MUJOCO_HEADLESS=true`` cannot unexpectedly launch a GLFW window.
        headless_value = os.environ.get('MUJOCO_HEADLESS', '0').strip().lower()
        self.headless_ = headless_value in {'1', 'true', 'yes', 'on'}
        if not self.headless_:
            self.viewer_ = mujoco.viewer.launch_passive(self.model_, self.data_, key_callback=self.keyboardCallback)
            self.viewer_.cam.distance = 1
            self.viewer_.cam.azimuth = 0
            self.viewer_.cam.elevation = -45

    def keyboardCallback(self, keycode):
        if chr(keycode) == ' ':
            self.dyn_paused_ = not self.dyn_paused_
            if self.dyn_paused_:
                print('simulation paused!')
            else:
                print('simulation resumed!')
        elif chr(keycode) == 'Ā':
            self.break_out_signal_ = True

    def reset_mj_env(self):
        self.data_.qpos[:] = np.copy(np.hstack((self.param_.init_obj_qpos_, self.param_.init_robot_qpos_)))
        self.data_.qvel[:] = np.copy(np.array(self.param_.n_qvel_ * [0]))

        mujoco.mj_forward(self.model_, self.data_)

    def step(self, fts_pos_cmd):
        curr_q = self.get_state()
        feasible_fts_cmd = fts_pos_cmd

        # calculate the graviety
        fullM = np.ndarray(shape=(self.param_.n_qvel_, self.param_.n_qvel_), dtype=np.float64, order="C")
        # MuJoCo <=3.1 exposed the dense mass matrix buffer as ``qM``;
        # newer releases renamed it to ``M``.  Support both APIs.
        mass_buffer = getattr(self.data_, 'qM', None)
        if mass_buffer is None:
            # MuJoCo 3.x takes (model, data, destination) directly.
            try:
                mujoco.mj_fullM(self.model_, self.data_, fullM)
            except TypeError:
                mass_buffer = self.data_.M
                mujoco.mj_fullM(self.model_, fullM, mass_buffer)
        else:
            mujoco.mj_fullM(self.model_, fullM, mass_buffer)
        fingertipM = fullM[-self.param_.n_cmd_:, :][:, -self.param_.n_cmd_:]

        desired_fts_pos = (curr_q[7:] + feasible_fts_cmd).copy()
        fts_dpos = []
        for _ in range(self.param_.frame_skip_):
            # gravity compensation
            self.data_.xfrc_applied[:] = 0
            self.data_.xfrc_applied[self.fingertip_body_id, :3] = - self.fingertip_mass * self.gravity_vec
            self.data_.xfrc_applied[self.fingertip_body_id, 3:] = 0

            curr_q = self.get_state()
            dpos = curr_q[7:] - desired_fts_pos
            dvel = self.data_.qvel[6:]
            control = -100 * dpos - 2 * dvel
            self.data_.ctrl[:] = control
            mujoco.mj_step(self.model_, self.data_, nstep=1)
            if self.viewer_ is not None:
                self.viewer_.sync()
            fts_dpos.append(dpos)

    def get_state(self):
        return self.data_.qpos.flatten().copy()

    def set_goal(self, goal_pos=None, goal_quat=None):
        if goal_pos is not None:
            self.model_.body('goal').pos = goal_pos
        if goal_quat is not None:
            self.model_.body('goal').quat = goal_quat
        mujoco.mj_forward(self.model_, self.data_)
        pass

    def get_bbox_size(self):
        return self.model_.geom('obj').size.copy()
    
    def show_target(self, goal_pos=None):
        # Marker updates are only visible in the interactive viewer.  In
        # headless rollouts ``mj_forward`` here used to duplicate the forward
        # pass performed by contact detection, adding measurable latency to
        # every control cycle.
        if self.headless_ or self.viewer_ is None:
            return
        if goal_pos is not None:
            self.model_.body('marker').pos = goal_pos
        # Defer mj_forward until show_best_contact(), which is called
        # immediately afterwards by the fingertip demo.

    def show_best_contact(self, point=None):
        """Update the yellow surface marker used by the fingertip demos."""
        if self.headless_ or self.viewer_ is None:
            return
        if point is not None:
            try:
                self.model_.body('marker_best').pos = np.asarray(point, dtype=float)
            except (KeyError, ValueError):
                # Keep compatibility with XMLs that do not define the extra
                # marker body.
                pass
        mujoco.mj_forward(self.model_, self.data_)

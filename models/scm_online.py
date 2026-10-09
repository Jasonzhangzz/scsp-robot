"""On-policy supervised soft-SCM identification, following ContactSDF V-B.

No upstream code is copied. Data come from executed physics windows, never from
queued planner actions. Bounded log parameters, a replay buffer and a disjoint
validation subset prevent an individual changing contact from driving the model.
"""
from collections import deque, Counter
import json
from pathlib import Path
import time

import casadi as cs
import numpy as np
from models.scm_parameters import SCM_LOG_LOWER, SCM_LOG_UPPER, SCM_PARAMETER_NAMES


class ExecutionTransitionBuffer:
    """Keep 20 ms windows available even when latest-only IPC drops observations.

    u is the final OSC position reference minus the tip at the start of the
    window. The target path mean is also recorded for diagnostics. This matches
    the position-increment input of the quasi-dynamic predictor; tracking error
    is intentionally part of the effective model being identified.
    """
    def __init__(self, trial, dt, capacity=32):
        self.trial, self.dt = int(trial), float(dt)
        self.records = deque(maxlen=capacity)
        self.next_id = 0
        self.pending = None

    def begin(self, q, fields):
        self.pending = dict(q=np.asarray(q, dtype=float).copy(),
                            phi=np.asarray(fields['phi_vec'], dtype=float).copy(),
                            jac=np.asarray(fields['jac_mat'], dtype=float).copy(),
                            contact=bool(fields['if_contact']), targets=[])

    def record_target(self, target):
        if self.pending is not None:
            self.pending['targets'].append(np.asarray(target, dtype=float).copy())

    def finish(self, q_plus, fields):
        p = self.pending
        self.pending = None
        if p is None or not p['targets']:
            return
        targets = np.asarray(p.pop('targets'))
        p.update(q_plus=np.asarray(q_plus, dtype=float).copy(),
                 u=targets[-1]-p['q'][7:10], target_mean=targets.mean(axis=0),
                 jac_plus=np.asarray(fields['jac_mat'], dtype=float).copy(),
                 phi_plus=np.asarray(fields['phi_vec'], dtype=float).copy(),
                 contact_plus=bool(fields['if_contact']), dt=len(targets)*self.dt,
                 id=self.next_id, trial=self.trial)
        self.next_id += 1
        self.records.append(p)

    def snapshot(self):
        return list(self.records)


class OnlineSCMIdentifier:
    def __init__(self, model, min_samples=32, update_every=32, buffer_size=512,
                 batch_size=64, gradient_steps=6, learning_rate=.03, log_path=None):
        if model.contact_model != 'soft-scm':
            raise ValueError('OnlineSCMIdentifier requires an adaptive soft-SCM model')
        if model.contact_dynamics != 'quasi-dynamic':
            raise ValueError('Executed-position calibration requires quasi-dynamic SCM')
        if min_samples < 10 or update_every < 1 or batch_size < 10 or gradient_steps < 1:
            raise ValueError('Invalid SCM replay/update configuration')
        self.model = model
        self.theta = np.zeros(6)
        self.buffer = deque(maxlen=buffer_size)
        self.min_samples, self.update_every = int(min_samples), int(update_every)
        self.batch_size, self.gradient_steps = int(batch_size), int(gradient_steps)
        self.lr = float(learning_rate)
        self.seen = {}
        self.accepted_samples = self.accepted_updates = self.rejected_updates = 0
        self.rejections = Counter()
        self.since_update = 0
        self.log_path = Path(log_path) if log_path else None
        self.history = []
        self.last_metrics = {}
        q = cs.SX.sym('q', model.param_.n_qpos_)
        u = cs.SX.sym('u', model.param_.n_cmd_)
        phi = cs.SX.sym('phi', model.param_.max_ncon_*4)
        j = cs.SX.sym('j', model.param_.max_ncon_*4, model.param_.n_qvel_)
        target = cs.SX.sym('target', model.param_.n_qpos_)
        theta = cs.SX.sym('theta', 6)
        pred = model.step_learn_fn(q, u, phi, j, theta)
        # Sign-invariant quaternion chord loss approximates squared angle/4.
        dot = cs.dot(pred[3:7], target[3:7]/cs.norm_2(target[3:7]))
        pos = cs.sumsqr((pred[:3]-target[:3])/.005)
        rot = (1-dot*dot)/(.05*.05/4)
        # Object pose is the supervised output. Actual tip motion is recorded,
        # but fitting the arm as a free mass would corrupt contact parameters.
        loss = pos + rot
        self.loss_fn = cs.Function('scm_identification_loss', [q,u,phi,j,target,theta],
                                   [loss, cs.gradient(loss,theta), pred])

    def _reject(self, reason):
        self.rejections[reason] += 1
        return False

    def add(self, sample):
        trial, sid = int(sample['trial']), int(sample['id'])
        if sid <= self.seen.get(trial, -1):
            return False
        self.seen[trial] = sid
        arrays = [np.asarray(sample[k], dtype=float) for k in ('q','u','phi','jac','q_plus','jac_plus','phi_plus')]
        if not all(np.isfinite(x).all() for x in arrays):
            return self._reject('nonfinite')
        if not np.isclose(float(sample['dt']), self.model.param_.h_, rtol=1e-3, atol=1e-6):
            return self._reject('duration')
        if not sample.get('contact') or not sample.get('contact_plus'):
            return self._reject('contact_entry_exit')
        q,u,phi,j,target,j_end,phi_end = arrays
        nv, nr = self.model.param_.n_qvel_, self.model.param_.max_ncon_*4
        if q.shape != (self.model.param_.n_qpos_,) or target.shape != q.shape or j.shape != (nr,nv) or j_end.shape != j.shape or phi.shape != (nr,) or phi_end.shape != phi.shape or u.shape != (self.model.param_.n_cmd_,):
            return self._reject('shape')
        if abs(np.linalg.norm(q[3:7])-1) > .01 or abs(np.linalg.norm(target[3:7])-1) > .01:
            return self._reject('quaternion')
        # Normal comparison is in world coordinates; the angular columns use
        # the object's body frame. Match contact sets without manifold ordering.
        def normals(jac):
            b = jac.reshape(-1,4,nv)
            robot = np.any(np.abs(b[:,:,6:])>1e-8, axis=(1,2))
            n = b[robot,:,:3].mean(axis=1)
            return n / np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
        n0,n1 = normals(j),normals(j_end)
        if len(n0) == 0 or len(n0) != len(n1):
            return self._reject('contact_set')
        if np.min(np.max(n0@n1.T,axis=1)) < .8 or np.min(np.max(n1@n0.T,axis=1)) < .8:
            return self._reject('normal_jump')
        def body_rotation(quat):
            w,x,y,z = quat
            return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                             [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                             [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        def arms(jac, rotation):
            blocks = jac.reshape(-1,4,nv)
            robot = np.any(np.abs(blocks[:,:,6:])>1e-8, axis=(1,2))
            means = blocks[robot].mean(axis=1)
            # n x (r x n) recovers the tangential component of the lever arm.
            return np.cross(means[:,:3]@rotation,means[:,3:6])
        r0,r1=arms(j,body_rotation(q[3:7])),arms(j_end,body_rotation(target[3:7]))
        match=np.argmax(n0@n1.T,axis=1)
        if np.max(np.linalg.norm(r0-r1[match],axis=1)) > .01:
            return self._reject('contact_point_jump')
        # Exclude teleports and invalid intervals; keep stationary contacts too
        # since prediction drift at rest is useful negative evidence.
        if np.linalg.norm(target[:3]-q[:3]) > .02 or np.linalg.norm(u) > .03:
            return self._reject('jump')
        if np.linalg.norm(u) < 1e-5 and np.linalg.norm(target[:3]-q[:3]) < 1e-6:
            return self._reject('unexcited')
        entry = dict(zip(('q','u','phi','jac','q_plus'), (q.copy(),u.copy(),phi.copy(),j.copy(),target.copy())))
        entry['prediction'] = np.asarray(self.model.step_learn_fn(q,u,phi,j,self.theta)).reshape(-1)
        entry['prediction_base'] = np.asarray(self.model.step_learn_fn(q,u,phi,j,np.zeros(6))).reshape(-1)
        entry['theta_at_prediction'] = self.theta.copy()
        entry['trial'] = np.asarray(trial)
        entry['id'] = np.asarray(sid)
        self.buffer.append(entry)
        self.accepted_samples += 1
        self.since_update += 1
        return True

    def _evaluate(self, samples, theta):
        # CasADi maps all samples in one call; no Torch/CUDA dependency.
        n = len(samples)
        fn = self.loss_fn.map(n)
        result = fn(np.stack([s['q'] for s in samples],axis=1),
                    np.stack([s['u'] for s in samples],axis=1),
                    np.stack([s['phi'] for s in samples],axis=1),
                    np.hstack([s['jac'] for s in samples]),
                    np.stack([s['q_plus'] for s in samples],axis=1),theta)
        return float(np.asarray(result[0]).mean()), np.asarray(result[1]).mean(axis=1)

    def update(self):
        if len(self.buffer) < self.min_samples or self.since_update < self.update_every:
            return None
        self.since_update = 0
        t0 = time.perf_counter()
        data = list(self.buffer)[-self.batch_size:]
        # Disjoint rolling holdout. This guards updates on available contacts;
        # it is not a claim of independent future-trajectory generalization.
        train = [s for i,s in enumerate(data) if i%5]
        valid = [s for i,s in enumerate(data) if not i%5]
        old_loss,_ = self._evaluate(valid,self.theta)
        candidate = self.theta.copy()
        m,v = np.zeros(6),np.zeros(6)
        for k in range(1,self.gradient_steps+1):
            _,g = self._evaluate(train,candidate)
            g += .02*candidate
            if not np.isfinite(g).all():
                break
            m=.9*m+.1*g; v=.999*v+.001*g*g
            step=self.lr*(m/(1-.9**k))/(np.sqrt(v/(1-.999**k))+1e-8)
            candidate = np.clip(candidate-step, SCM_LOG_LOWER,SCM_LOG_UPPER)
            candidate = np.clip(candidate,self.theta-.08,self.theta+.08)
        new_loss,_ = self._evaluate(valid,candidate)
        accepted = bool(np.isfinite(new_loss) and new_loss < old_loss*(1-1e-4))
        if accepted:
            self.theta = candidate
            self.accepted_updates += 1
        else:
            self.rejected_updates += 1
        record = dict(samples=self.accepted_samples, buffer=len(self.buffer),
                      accepted=accepted, validation_loss_before=old_loss,
                      validation_loss_proposed=new_loss, validation_loss_after=new_loss if accepted else old_loss,
                      update_ms=(time.perf_counter()-t0)*1000,
                      scales=dict(zip(SCM_PARAMETER_NAMES,np.exp(self.theta).tolist())),
                      theta=self.theta.tolist(), rejections=dict(self.rejections))
        self.history.append(record)
        self.last_metrics = record
        if self.log_path:
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            with self.log_path.open('a') as f:
                f.write(json.dumps(record)+'\n')
        return record

    def observe(self, samples, learn=True):
        self.learning_enabled = bool(learn)
        for sample in samples:
            self.add(sample)
        update = self.update() if learn else None
        return update

    def save(self, path):
        data = list(self.buffer)
        if not data:
            return
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(str(path), **{k:np.stack([s[k] for s in data]) for k in data[0]},
                            theta=self.theta, Q=self.model.param_.Q,
                            h=self.model.param_.h_, tau=self.model.soft_scm_response_time,
                            zeta=self.model.soft_scm_damping_ratio,
                            mu=self.model.param_.mu_object_, obj_mass=self.model.param_.obj_mass_,
                            robot_stiff=self.model.param_.robot_stiff_, gravity=self.model.param_.gravity_,
                            contact_dynamics=self.model.contact_dynamics)
        metadata = dict(parameter_names=list(SCM_PARAMETER_NAMES), theta=self.theta.tolist(),
                        scales=dict(zip(SCM_PARAMETER_NAMES,np.exp(self.theta).tolist())),
                        accepted_samples=self.accepted_samples, accepted_updates=self.accepted_updates,
                        rejections=dict(self.rejections))
        path.with_name('parameters.json').write_text(json.dumps(metadata,indent=2)+'\n')

    def load(self, path):
        from models.scm_parameters import validate_theta
        with np.load(path,allow_pickle=False) as data:
            expected = dict(Q=self.model.param_.Q, h=self.model.param_.h_,
                tau=self.model.soft_scm_response_time, zeta=self.model.soft_scm_damping_ratio,
                mu=self.model.param_.mu_object_, obj_mass=self.model.param_.obj_mass_,
                robot_stiff=self.model.param_.robot_stiff_, gravity=self.model.param_.gravity_)
            for key,value in expected.items():
                if key not in data or data[key].shape != np.asarray(value).shape or not np.allclose(data[key],value):
                    raise ValueError('SCM checkpoint has incompatible '+key)
            if str(data['contact_dynamics']) != self.model.contact_dynamics:
                raise ValueError('SCM checkpoint has incompatible dynamics convention')
            self.theta = validate_theta(data['theta'])

    def diagnostics(self):
        return {**self.last_metrics, 'enabled': bool(getattr(self, 'learning_enabled', self.model.online_scm)),
                'samples': self.accepted_samples, 'updates': self.accepted_updates,
                'rejected_updates': self.rejected_updates, 'rejections': dict(self.rejections),
                'theta': self.theta.tolist()}

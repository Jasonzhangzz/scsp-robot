"""Object velocity error versus DART contact dynamics + MuJoCo collision.

LCP uses a deterministic NumPy port of DART's ContactConstraint and boxed PGS
at h=0.001 s, with MuJoCo collision. softSCM, M1, comfree and ContactSDF step at
h=0.002 s and are compared with two GT substeps. The hard diagonal SCM is omitted.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import quoteattr

ROOT = Path(__file__).resolve().parents[4]
SCM = ROOT / "examples/mpc/franka/scm"
for path in (ROOT, SCM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from runtime_env import activate_dependencies

activate_dependencies()

import mujoco
import numpy as np

from contact_dynamics import (CpuContactEngine, dart_pgs_solve,
                              DART_ERP, DART_MAX_ERV, DART_CFM)

OBJECTS = (
    "foam_brick", "mug", "rubber_duck", "elephant", "piggy_bank",
    "stanford_bunny2", "teapot",
)
CONDITIONS = ("transition", "sliding", "sticking", "rollout")
H_MODEL = 0.002
H_LCP = 0.001
H_GT = 0.001
GT_SUBSTEPS = int(round(H_MODEL / H_GT))
SOFT_TAU = 0.008
SOFT_ZETA = 0.25


def mesh_asset(shape):
    path = ROOT / "envs/assets/objects" / f"{shape}.stl"
    scene = ROOT / "envs/xmls" / f"env_fingertips_{shape}.xml"
    geom = ET.parse(scene).find(".//geom[@name='obj']")
    if not path.is_file() or geom is None:
        raise FileNotFoundError(shape)
    return path, geom.get("quat", "1 0 0 0")


def mesh_world_vertices(model, data):
    geom = model.geom("object").id
    mesh = int(model.geom_dataid[geom])
    adr, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    return (model.mesh_vert[adr:adr + count] @ data.geom_xmat[geom].reshape(3, 3).T
            + data.geom_xpos[geom])


def make_model(shape, friction, timestep, mass):
    path, quat = mesh_asset(shape)
    xml = f'''<mujoco model="velocity_{shape}">
      <option timestep="{timestep}" integrator="Euler" cone="pyramidal" iterations="50" tolerance="1e-8" gravity="0 0 -9.81"/>
      <default><joint damping="0"/><geom condim="3" friction="{friction} .005 .0001"/></default>
      <asset><mesh name="object_mesh" file={quoteattr(str(path))}/></asset>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <body name="object" pos="0 0 .05"><freejoint/>
          <geom name="object" type="mesh" mesh="object_mesh" quat={quoteattr(quat)} mass="{mass}"/>
        </body>
      </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    shift = .002 - float(mesh_world_vertices(model, data)[:, 2].min())
    model.qpos0[2] += shift
    model.body_pos[1, 2] += shift
    mujoco.mj_setConst(model, data)
    return model


def set_state(model, data, qpos, qvel, wrench):
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.xfrc_applied[1] = wrench
    mujoco.mj_forward(model, data)


def world_velocity(data, rotation):
    """Angular velocity in the pre-step world frame."""
    return np.r_[data.qvel[:3], rotation @ data.qvel[3:6]]


def _cpu_velocity(model, data, law, h, sigma=120.):
    probe = mujoco.MjData(model)
    mujoco.mj_copyData(probe, model, data)
    rotation = data.xmat[1].reshape(3, 3).copy()
    CpuContactEngine(model, law, sigma).step(probe, h)
    return world_velocity(probe, rotation)


def dart_lcp_velocity(model, data, friction, h):
    """DART v6.16 boxed contact law using MuJoCo collision and smooth dynamics.

    Friction comes from MuJoCo's resolved contact parameters, not the override.
    The PGS solver is a deterministic NumPy port, not a native DART world.
    """
    return _cpu_velocity(model, data, "dart", h)


def dart_window(model, qpos, qvel, wrench, friction, h, steps):
    """Integrate DART LCP for ``steps`` and return velocity in the initial world frame."""
    data = mujoco.MjData(model)
    set_state(model, data, qpos, qvel, wrench)
    rotation0 = data.xmat[1].reshape(3, 3).copy()
    nxt = None
    for i in range(steps):
        world = dart_lcp_velocity(model, data, friction, h)
        rotation = data.xmat[1].reshape(3, 3)
        data.qvel[:3] = world[:3]
        data.qvel[3:] = rotation.T @ world[3:]
        mujoco.mj_integratePos(model, data.qpos, data.qvel, h)
        data.xfrc_applied[1] = wrench
        mujoco.mj_forward(model, data)
        nxt = (data.qpos.copy(), data.qvel.copy())
    final = np.r_[data.qvel[:3], rotation0 @ data.qvel[3:6]]
    return final, nxt


def lse_velocity(model, data, friction, h, sigma=120.0):
    """ContactSDF dynamic LSE extension, Q=M/h²; not pretrained quasi-dynamics."""
    return _cpu_velocity(model, data, "contactsdf", h, sigma)


def step_mujoco(model, qpos, qvel, wrench, steps):
    data = mujoco.MjData(model)
    set_state(model, data, qpos, qvel, wrench)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data


def contact_mode(model, data, slip=0.01, normal=0.02, force=1e-3):
    loaded_slip, loaded_normal, any_load = [], [], False
    for index in range(data.ncon):
        con = data.contact[index]
        g1, g2 = int(con.geom1), int(con.geom2)
        if model.geom("floor").id not in (g1, g2):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        if abs(wrench[0]) <= force:
            continue
        any_load = True
        jacp = np.zeros((3, model.nv))
        relative = np.zeros((3, model.nv))
        for sign, geom in ((-1., g1), (1., g2)):
            mujoco.mj_jac(model, data, jacp, None, con.pos, int(model.geom_bodyid[geom]))
            relative += sign*jacp
        local = np.asarray(con.frame).reshape(3, 3) @ relative @ data.qvel
        loaded_normal.append(abs(local[0]))
        loaded_slip.append(float(np.linalg.norm(local[1:3])))
    if not any_load:
        return "separation"
    if max(loaded_normal) > normal:
        return "transition"
    return "sliding" if max(loaded_slip) > slip else "sticking"


def sample_state(settled, condition, rng, friction, mass):
    qpos, qvel, wrench = settled.qpos.copy(), np.zeros(6), np.zeros(6)
    yaw = rng.uniform(-np.pi, np.pi)
    qyaw = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    orientation = np.zeros(4)
    mujoco.mju_mulQuat(orientation, qyaw, qpos[3:7])
    qpos[3:7] = orientation
    angle = rng.uniform(-np.pi, np.pi)
    direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    if condition == "transition":
        # Already supported, with a downward speed so this step is an impact.
        qvel[2] = -rng.uniform(0.2, 1.2)
        qvel[:2] = direction[:2] * rng.uniform(0.0, 0.4)
        qvel[3:] = rng.uniform(-1.0, 1.0, 3)
    elif condition == "sliding":
        qvel[:2] = direction[:2] * rng.uniform(0.2, 1.2)
        wrench[:2] = direction[:2] * rng.uniform(0.05, 0.4) * friction * mass * 9.81
    elif condition == "sticking":
        wrench[:2] = direction[:2] * rng.uniform(0.0, 0.15) * friction * mass * 9.81
    else:
        speed = rng.uniform(0.15, 0.7)
        qvel[:2] = direction[:2] * speed
        radius = max(float(settled.xpos[1, 2]), 0.01)
        omega_world = np.cross(np.array([0.0, 0.0, 1.0]), direction) * (speed / radius)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, qpos[3:7])
        qvel[3:] = rotation.reshape(3, 3).T @ omega_world
    return qpos, qvel, wrench


def errors(pred, ref):
    if pred is None or not np.isfinite(pred).all():
        return None, None
    return (float(np.linalg.norm(pred[:3] - ref[:3])),
            float(np.linalg.norm(pred[3:] - ref[3:])))


def step_backend(backend, model, qpos, qvel, wrench):
    data = mujoco.MjData(model)
    set_state(model, data, qpos, qvel, wrench)
    rotation = data.xmat[1].reshape(3, 3).copy()
    backend.reset(data)
    backend.step(data)
    return world_velocity(data, rotation), data


def predict_all(model, qpos, qvel, wrench, friction, backends):
    probe = mujoco.MjData(model)
    set_state(model, probe, qpos, qvel, wrench)
    pre = probe.xmat[1].reshape(3, 3).copy()
    reference, nxt = dart_window(model, qpos, qvel, wrench, friction, H_LCP, GT_SUBSTEPS)
    out = {"LCP": reference, "GT": reference, "next": nxt,
           "measured_mode": contact_mode(model, probe)}
    m1 = step_mujoco(model, qpos, qvel, wrench, 1)
    out["M1"] = world_velocity(m1, pre)
    for name, engine in (("softSCM", "scm_soft"), ("softSCM_corrected", "scm_soft_corrected"), ("comfree", "comfree"), ("diagonalContact", "diagonal_contact")):
        backend = backends.get(engine)
        if backend is None:
            out[name] = None
            continue
        try:
            out[name], _ = step_backend(backend, model, qpos, qvel, wrench)
        except (RuntimeError, FloatingPointError, ValueError):
            out[name] = None
    probe = mujoco.MjData(model)
    set_state(model, probe, qpos, qvel, wrench)
    try:
        out["ContactSDF"] = lse_velocity(model, probe, friction, H_MODEL)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        out["ContactSDF"] = None
    return out


def settle(model, steps=400):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if data.time > 0.3 and np.max(np.abs(data.qvel)) < 0.02:
            break
    data.qvel[:] = 0
    data.time = 0
    mujoco.mj_forward(model, data)
    return data


def stats(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "std": None}
    return {"n": int(len(values)), "mean": float(values.mean()), "std": float(values.std())}


def fmt(block):
    if not block or block["n"] == 0 or block["mean"] is None:
        return "--"
    return "%.4f ± %.4f (n=%d)" % (block["mean"], block["std"], block["n"])


def make_backends(model, settled):
    from benchmark_utils import Backend
    common = dict(device="cuda:0", nconmax=512, njmax=4096, stiffness=0.2, damping=0.001,
                  scm_response_time=SOFT_TAU, scm_damping_ratio=SOFT_ZETA)
    backends = {}
    for engine in ("scm_soft", "scm_soft_corrected", "comfree", "diagonal_contact"):
        try:
            backends[engine] = Backend(engine, model, settled, **common)
        except Exception as exc:
            print(engine, "unavailable:", exc, flush=True)
    return backends


def run(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out = out / ("velocity_error_" + stamp)
    out.mkdir(parents=True)
    log_path = out / "velocity_error.log"
    rows = []
    rng = np.random.default_rng(args.seed)
    models = ("LCP", "M1", "softSCM", "softSCM_corrected", "comfree", "diagonalContact", "ContactSDF")
    with log_path.open("w") as log:
        log.write("Reference is the deterministic DART boxed PGS NumPy port with MuJoCo collision, %d steps of h=0.001 s.\n" % GT_SUBSTEPS)
        log.write("M1, softSCM, comfree and ContactSDF use one step of h=0.002 s.\n")
        log.write("LCP error is zero because it is the reference.\n")
        log.write("Error is ||v_model - v_GT|| in the pre-step world frame. Linear m/s, angular rad/s.\n")
        log.write("softSCM: tau=%.3f s, zeta=%.2f. Hard diagonal SCM is omitted.\n" % (SOFT_TAU, SOFT_ZETA))
        for shape in args.objects:
            print("object", shape, flush=True)
            model_h = make_model(shape, args.friction, H_MODEL, args.mass)
            settled = settle(model_h)
            backends = make_backends(model_h, settled)
            for condition in CONDITIONS:
                for index in range(args.samples):
                    qpos, qvel, wrench = sample_state(settled, condition, rng, args.friction, args.mass)
                    horizon = args.rollout_steps if condition == "rollout" else 1
                    for step in range(horizon):
                        pred = predict_all(model_h, qpos, qvel, wrench, args.friction, backends)
                        record = {"object": shape, "condition": condition, "sample": index, "step": step,
                                  "measured_mode": pred["measured_mode"]}
                        for name in models:
                            lin, ang = errors(pred[name], pred["GT"])
                            record[name + "_linear"] = lin
                            record[name + "_angular"] = ang
                        rows.append(record)
                        if condition == "rollout":
                            qpos, qvel = pred["next"]
                    if (index + 1) % 25 == 0:
                        print("  %s %d/%d" % (condition, index + 1, args.samples), flush=True)
                        log.flush()
            summary = aggregate(rows, models, shape_filter=shape)
            log.write("\n[%s]\n" % shape)
            log.write(render(summary, models))
            log.flush()
        summary = aggregate(rows, models, shape_filter=None)
        text = render(summary, models)
        log.write("\n[all objects]\n")
        log.write(text)
        print(text, flush=True)
    payload = {"objects": list(args.objects), "samples": args.samples, "seed": args.seed,
               "rollout_steps": args.rollout_steps, "h_model": H_MODEL,
               "h_lcp": H_LCP, "h_gt": H_GT, "friction": args.friction, "mass": args.mass,
               "reference": "dart_pgs_lcp_2x0.001",
               "lcp": "DART v6.16 deterministic NumPy boxed PGS port; MuJoCo collision",
               "soft_scm": {"tau": SOFT_TAU, "zeta": SOFT_ZETA},
               "contact_dimensions": 3,
               "diagonal_contact": "one ordered normal/tangent pass; physical diagonal Delassus; box friction; no fitted gains or history",
               "soft_scm_corrected": "same diag Delassus; one global quadratic amplitude correction",
               "contact_sdf": "dynamic extension of upstream LSE; Q=M/h^2; sigma=120",
               "collision": "MuJoCo CPU for DART/ContactSDF; vendored MuJoCo Warp for GPU models",
               "summary": aggregate(rows, models, None),
               "rows": rows}
    (out / "velocity_error.json").write_text(json.dumps(payload) + "\n")
    print("wrote", log_path, flush=True)


def aggregate(rows, models, shape_filter):
    selected = [r for r in rows if shape_filter is None or r["object"] == shape_filter]
    report = {}
    for condition in CONDITIONS:
        report[condition] = {}
        group = [r for r in selected if r["condition"] == condition]
        for model in models:
            report[condition][model] = {
                "linear_m_s": stats(r[model + "_linear"] for r in group),
                "angular_rad_s": stats(r[model + "_angular"] for r in group),
            }
    return report


def render(summary, models):
    lines = ["condition | " + " | ".join(models)]
    for condition in CONDITIONS:
        cells = []
        for model in models:
            block = summary[condition][model]
            cells.append("v %s ; w %s" % (fmt(block["linear_m_s"]), fmt(block["angular_rad_s"])))
        lines.append(condition + " | " + " | ".join(cells))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", default=list(OBJECTS))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--mass", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="output/diagnostics")
    args = parser.parse_args()
    if args.samples < 1 or any(name not in OBJECTS for name in args.objects):
        parser.error("samples >= 1 and objects must be the seven manipulation meshes")
    run(args)


if __name__ == "__main__":
    main()

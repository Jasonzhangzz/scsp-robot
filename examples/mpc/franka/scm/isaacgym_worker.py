"""PhysX worker process for ``IsaacGymProxy`` (run with the Isaac Gym interpreter).

Protocol: one JSON object per line on stdin, one JSON reply per line on stdout.
    {"cmd": "init", "spec": {...}, "headless": true}
    {"cmd": "reset", "qpos": [7], "qvel": [6]}
    {"cmd": "step", "qpos": [7], "qvel": [6], "wrench": [6]} -> {"qpos", "qvel", "seconds"}
    {"cmd": "close"}
Isaac Gym is imported first (before NumPy-heavy modules) as the SDK requires.
"""
import json
import os
import sys

# Isaac Gym and PhysX print banners on file descriptor 1. Keep a private copy
# of the original stdout for the protocol and send everything else to stderr.
_PROTOCOL = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import isaacgym  # noqa: F401,E402  (must precede other imports)
import numpy as np  # noqa: E402

from isaacgym_backend import IsaacGymBackend  # noqa: E402


def main():
    backend = None
    out = _PROTOCOL
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            cmd = request["cmd"]
            if cmd == "init":
                backend = IsaacGymBackend(request["spec"], headless=bool(request.get("headless", True)))
                reply = {"ok": True, "fallback_geometry": backend.fallback_geometry,
                         "collision": backend.collision, "physx_mass": backend.physx_mass}
            elif cmd == "reset":
                backend._write_state(np.asarray(request["qpos"]), np.asarray(request["qvel"]))
                reply = {"ok": True}
            elif cmd == "step":
                qpos, qvel, seconds = backend.predict(np.asarray(request["qpos"]),
                                                      np.asarray(request["qvel"]),
                                                      np.asarray(request["wrench"]))
                reply = {"qpos": qpos.tolist(), "qvel": qvel.tolist(), "seconds": seconds}
            elif cmd == "close":
                if backend is not None:
                    backend.close()
                out.write(json.dumps({"ok": True}) + "\n")
                out.flush()
                return
            else:
                reply = {"error": f"unknown command {cmd}"}
        except Exception as exc:  # report, keep serving
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()

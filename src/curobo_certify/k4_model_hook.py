"""Make cuRobo TimeOpt plan with a K4 model contract (e.g. the identified Gaz et al. 2019 model).

Runtime-only, no cuRobo source change: before TrajectoryTimeOpt builds its Dynamics object, every link's
mass, centre of mass and CoM inertia (ixx iyy izz ixy ixz iyz, link frame) in TimeOpt's KinematicsParams are
overwritten with the contract's values (rounded to float32, as cuRobo stores parameters), and TimeOpt's
gravity is set to the contract's gravity. Link frames / joint axes are NOT changed: the identified contracts
reuse contract-v2 kinematics, which equal cuRobo's (pinned URDF). Values are read back after construction and
returned as an audit; `verify` scripts compare TimeOpt's own GPU torques with the contract's interval RNEA.

Torque limits are unchanged (FER tau_max 87/12 N m, scaled by TimeOpt's limit_scale). Friction is not part of
the rigid contract and cannot be represented in cuRobo's RNEA; use certification with --friction-bound.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from fractions import Fraction as Q


def contract_inertials(contract: dict) -> dict:
    m = contract["model"]
    links = {e["name"]: {"mass": float(Q(e["mass"])), "com": [float(Q(v)) for v in e["com"]],
                         "inertia": [float(Q(v)) for v in e["inertia"]]} for e in m["links"]}
    return {"links": links, "gravity": [float(Q(v)) for v in m["gravity"]]}


def install_timeopt_model_hook(contract: dict):
    """Patch TrajectoryTimeOpt.__init__ (idempotent w.r.t. earlier hooks: always wraps the ORIGINAL init).
    Returns the audit list, appended to on every TimeOpt construction."""
    import torch
    import curobo._src.motion.motion_trajectory_timeopt as mt
    P = contract_inertials(contract)
    audit = []
    orig = getattr(mt.TrajectoryTimeOpt, "_k4_original_init", mt.TrajectoryTimeOpt.__init__)
    mt.TrajectoryTimeOpt._k4_original_init = orig
    sha = contract["contract_sha256"]

    def init(self, config, *a, **kw):
        kin = config.robot.kinematics_config
        idx = kin.link_name_to_idx_map
        missing = [n for n in P["links"] if n not in idx]
        if missing:
            raise RuntimeError(f"K4 model hook: links not in cuRobo kinematics: {missing}")
        with torch.no_grad():
            for name, v in P["links"].items():
                i = idx[name]
                kin.link_masses_com[i, :3] = torch.tensor(v["com"], dtype=kin.link_masses_com.dtype)
                kin.link_masses_com[i, 3] = v["mass"]
                kin.link_inertias[i, :6] = torch.tensor(v["inertia"], dtype=kin.link_inertias.dtype)
        config = dataclasses.replace(config, gravity=tuple(P["gravity"]))
        orig(self, config, *a, **kw)
        rb = {n: {"masses_com": kin.link_masses_com[idx[n]].tolist(), "inertia": kin.link_inertias[idx[n], :6].tolist()}
              for n in P["links"]}
        ok = all(abs(rb[n]["masses_com"][3] - v["mass"]) <= 1e-6 * max(1, v["mass"]) for n, v in P["links"].items())
        audit.append({"contract_sha256": sha, "gravity": list(self._dynamics.config.gravity) if self._dynamics is not None
                      else list(config.gravity), "links": rb, "masses_match": ok})
    mt.TrajectoryTimeOpt.__init__ = init
    return audit


def uninstall():
    import curobo._src.motion.motion_trajectory_timeopt as mt
    if hasattr(mt.TrajectoryTimeOpt, "_k4_original_init"):
        mt.TrajectoryTimeOpt.__init__ = mt.TrajectoryTimeOpt._k4_original_init


def planning_contract(base: dict, payload: str | None) -> dict:
    """The exact contract TimeOpt is made to plan with (identified base + optional hypothetical payload)."""
    if payload is None:
        return base
    from .k4_payload import with_payload
    return with_payload(base, payload)


def contract_digest(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract["model"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()

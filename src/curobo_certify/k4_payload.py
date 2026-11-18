"""K4 HYPOTHETICAL payload family, preregistered BEFORE any payload solve (SYNTHETIC_PAYLOAD).

No real payload object is sourced. The user approved a clearly labelled hypothetical payload:
a uniform solid cylinder (radius 0.04 m, height 0.10 m), axis along the panda_hand z axis, centre of mass
at the Franka Hand TCP offset (0, 0, 0.1034) m in the panda_hand frame (attached_object frame = panda_hand,
identity transform, franka.yml extra_links), i.e. an object held centred between the fingers.
  PAYLOAD_SYNTH_3KG_CYL  3 kg  (= Panda rated payload)
  PAYLOAD_SYNTH_5KG_CYL  5 kg  ABOVE_RATED_PAYLOAD (exceeds the 3 kg rating; stress condition)
Inertia about the CoM (exact): Ixx = Iyy = m (3 r^2 + h^2) / 12, Izz = m r^2 / 2.
Payload collision geometry is NOT modelled (K4 is a torque contract only).
The payload replaces the attached_object default inertial (0.01 kg) in BOTH the cuRobo-model contract (v2)
and the identified-model contracts, and is injected into cuRobo TimeOpt's dynamics at construction time
(runtime-only hook; read back and audited) so the native planner plans WITH the payload.
"""
from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction as Q

R, H, TCP_Z = Q(4, 100), Q(10, 100), Q(1034, 10000)
PAYLOADS = {"PAYLOAD_SYNTH_3KG_CYL": Q(3), "PAYLOAD_SYNTH_5KG_CYL": Q(5)}
LABELS = {"PAYLOAD_SYNTH_3KG_CYL": "SYNTHETIC_PAYLOAD_RATED", "PAYLOAD_SYNTH_5KG_CYL": "SYNTHETIC_PAYLOAD_ABOVE_RATED"}


def inertial(name):
    m = PAYLOADS[name]
    ixx = m * (3 * R * R + H * H) / 12
    izz = m * R * R / 2
    return {"mass": m, "com": [Q(0), Q(0), TCP_Z], "inertia": [ixx, ixx, izz, Q(0), Q(0), Q(0)]}


def with_payload(contract: dict, name: str) -> dict:
    """New contract dict with attached_object inertial replaced by the payload (exact decimal strings)."""
    p = inertial(name)
    c = copy.deepcopy(contract)
    links = c["model"]["links"]
    idx = [e["name"] for e in links].index("attached_object")
    links[idx]["mass"] = str(p["mass"])
    links[idx]["com"] = [str(v) for v in p["com"]]
    links[idx]["inertia"] = [str(v) for v in p["inertia"]]
    links[idx]["inertial_source"] = f"{name} ({LABELS[name]}), preregistered hypothetical uniform cylinder"
    c["model"]["scope"] = c["model"]["scope"] + "+" + name
    c["model"]["payload"] = {"name": name, "label": LABELS[name], "radius_m": "0.04", "height_m": "0.1",
                             "tcp_offset_m": "0.1034", "collision": "NOT_MODELLED"}
    body = json.dumps(c["model"], sort_keys=True, separators=(",", ":")).encode()
    c["contract_sha256"] = hashlib.sha256(body).hexdigest()
    for k in ("float32_crosscheck_vs_curobo", "curobo_float32_rotation_non_orthonormality", "supersedes_contract_sha256"):
        c.pop(k, None)
    return c


def install_timeopt_payload_hook(name: str):
    """Runtime-only: before cuRobo TrajectoryTimeOpt builds its Dynamics, overwrite the attached_object
    mass/CoM/inertia in its KinematicsParams (float32, as cuRobo stores parameters). Returns an audit list
    filled with the values read back after construction."""
    import torch
    import curobo._src.motion.motion_trajectory_timeopt as mt
    p = inertial(name)
    audit = []
    orig = getattr(mt.TrajectoryTimeOpt, "_k4_original_init", mt.TrajectoryTimeOpt.__init__)
    mt.TrajectoryTimeOpt._k4_original_init = orig

    def init(self, config, *a, **kw):
        kin = config.robot.kinematics_config
        i = kin.link_name_to_idx_map["attached_object"]
        with torch.no_grad():
            kin.link_masses_com[i, :3] = torch.tensor([float(v) for v in p["com"]], dtype=kin.link_masses_com.dtype)
            kin.link_masses_com[i, 3] = float(p["mass"])
            kin.link_inertias[i, :6] = torch.tensor([float(v) for v in p["inertia"]], dtype=kin.link_inertias.dtype)
        orig(self, config, *a, **kw)
        audit.append({"link_index": int(i), "masses_com": kin.link_masses_com[i].tolist(),
                      "inertias": kin.link_inertias[i, :6].tolist(), "payload": name})
    mt.TrajectoryTimeOpt.__init__ = init
    return audit


# ---------------- preregistered payload population (same frozen source queries as K4-Panda-Dev40) -------------
COHORT = "K4-Panda-PayloadDev40"
K, ARMS = 40, (("primary", "0.95"), ("secondary", "0.99"))


def freeze_population(base_rows: list[dict]) -> list[dict]:
    """For each payload condition: the K4-Panda-Dev40 pool goals (primary repeat-0 rows, frozen order),
    arms 0.95 / 0.99; new source ids; first-K = 40 per payload on the 0.95 arm (rule as in k4_population)."""
    from .margin_campaign import margin_label
    from .query_pipeline import _canonical_hash, verify_query
    base = sorted((r for r in base_rows if r["arm"] == "primary" and r["repeat"] == 0), key=lambda r: r["pool_order"])
    out = []
    for name in PAYLOADS:
        tag = "p3" if "3KG" in name else "p5"
        for b in base:
            for arm, margin in ARMS:
                label = margin_label(margin)
                row = {k: b[k] for k in ("schema", "robot", "scene_id", "scene_stratum", "magnitude_stratum", "seed",
                                         "query_index", "goal_delta_rad", "start_joint_positions_rad",
                                         "goal_joint_positions_rad", "query_resolution", "planner", "timeopt",
                                         "source_geometry_contract", "pool_order")}
                row.update({"protocol": "k4_panda_payload_v1", "cohort": COHORT, "payload": name,
                            "payload_label": LABELS[name], "arm": arm, "repeat": 0, "cohort_k_eligible": K,
                            "pair_id": f"k4-payload-{tag}-{b['pool_order']:04d}", "solver_condition": label,
                            "planning_margin": margin, "source_id": f"k4-payload-{tag}-{b['pool_order']:04d}-{label}",
                            "base_source_id": b["source_id"]})
                row["source_query_sha256"] = _canonical_hash(row)
                if not verify_query(row):
                    raise AssertionError("freeze")
                out.append(row)
    return out

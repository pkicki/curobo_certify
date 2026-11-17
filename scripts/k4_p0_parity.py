#!/usr/bin/env python3
"""K4 P0 numerical parity (DIAGNOSTIC, not a continuous-time proof).

Three implementations on the same seeded float32 states (from k4_extract_curobo_model.py):
  1. native cuRobo GPU inverse dynamics (float32, lumped fixed links)          -> tau_native
  2. this repository's interval RNEA on the frozen contract (point boxes)      -> rigorous [lo, hi]
  3. Pinocchio (independent URDF parse + independent RNEA, float64), configured to cuRobo's
     actual inertial set: right_gripper dummy inertial removed (cuRobo does not load it),
     cuRobo default inertials (0.01 kg, 1e-4 I) added to panda_link8, both fingers and an
     attached_object frame at panda_hand; finger joints locked at 0.04 m.     -> tau_pin
Also checks cuRobo's coefficient identity (G = F(q,0,0), B = (F(q,0,q')-G)/2, A = F(q,q',q'')-G)
against directly computed interval A, B, G (no friction/damping/rotor terms exist in cuRobo).
Run with an interpreter that has pinocchio (runs/pinvenv) and PYTHONPATH=src.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.k4_interval import IA  # noqa: E402
from curobo_certify.k4_panda_model import PandaModel, load_contract  # noqa: E402

URDF = Path("/home/pkicki/curobo-pinned-c301cb00a87012c2808b1a60649c4065c9b3c95a/curobo/content/assets/robot/"
            "franka_description/franka_panda.urdf")
DEFAULT_INERTIAL = ('<inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="0.01"/>'
                    '<inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>')


def pinocchio_model():
    import pinocchio as pin
    txt = URDF.read_text()
    deltas = []
    # cuRobo does not load right_gripper / ee_link: drop the dummy inertial (otherwise lumped into link7)
    txt2 = re.sub(r'(<link name="right_gripper">)\s*<inertial>.*?</inertial>', r"\1", txt, flags=re.S)
    if txt2 == txt:
        raise SystemExit("right_gripper inertial not found")
    deltas.append("removed right_gripper dummy inertial (0.01 kg) - not part of cuRobo's model")
    for link in ("panda_link8", "panda_leftfinger", "panda_rightfinger"):
        pat = re.compile(rf'<link name="{link}"\s*/>|<link name="{link}">')
        m = pat.search(txt2)
        if m is None:
            raise SystemExit("link not found " + link)
        if m.group(0).endswith("/>"):
            txt2 = txt2[:m.start()] + f'<link name="{link}">{DEFAULT_INERTIAL}</link>' + txt2[m.end():]
        else:
            txt2 = txt2[:m.end()] + DEFAULT_INERTIAL + txt2[m.end():]
        deltas.append(f"added cuRobo default inertial to {link}")
    txt2 = txt2.replace("</robot>", '<link name="attached_object">' + DEFAULT_INERTIAL + '</link>'
                        '<joint name="attach_joint" type="fixed"><parent link="panda_hand"/><child link="attached_object"/>'
                        '<origin rpy="0 0 0" xyz="0 0 0"/></joint></robot>')
    deltas.append("added attached_object (0.01 kg default) fixed at panda_hand, identity transform (franka.yml extra_links)")
    full = pin.buildModelFromXML(txt2)
    lock = [full.getJointId(n) for n in ("panda_finger_joint1", "panda_finger_joint2")]
    qref = pin.neutral(full)
    for jid in lock:
        qref[full.joints[jid].idx_q] = 0.04
    red = pin.buildReducedModel(full, lock, qref)
    deltas.append("finger joints locked at 0.04 m (franka.yml lock_joints)")
    names = [red.names[i] for i in range(1, red.njoints)]
    return pin, red, names, deltas, hashlib.sha256(txt2.encode()).hexdigest()


def f32_params(pin, model):
    """Copy of the Pinocchio model with every joint placement and inertia rounded to float32
    (attribution test: cuRobo stores float32 parameters)."""
    m = model.copy()
    r32 = lambda a: np.asarray(a, dtype=np.float32).astype(np.float64)  # noqa: E731
    for i in range(1, m.njoints):
        pl = m.jointPlacements[i]
        m.jointPlacements[i] = pin.SE3(r32(pl.rotation), r32(pl.translation))
        I = m.inertias[i]
        m.inertias[i] = pin.Inertia(float(np.float32(I.mass)), r32(I.lever), r32(I.inertia))
    m.gravity = pin.Motion(r32(model.gravity.linear), r32(model.gravity.angular))  # cuRobo: float32 9.81
    return m


def model_from_pinocchio(pin, m):
    """PandaModel on EXACTLY Pinocchio's own (lumped, float64) parameterisation, to validate the
    interval RNEA algorithm (frames, axes, signs, recursion) independently of parameter precision."""
    links = m.njoints  # universe + 7 joints
    for i in range(1, links):
        if m.joints[i].shortname() != "JointModelRZ":
            raise SystemExit("unexpected pinocchio joint " + m.joints[i].shortname())
    T, mc, ine = [], [], []
    for i in range(links):
        pl = m.jointPlacements[i]
        T.append(np.hstack([pl.rotation, pl.translation[:, None]]))
        I = m.inertias[i]
        mc.append([*I.lever, I.mass])
        J = I.inertia
        ine.append([J[0, 0], J[1, 1], J[2, 2], J[0, 1], J[0, 2], J[1, 2]])
    hx = lambda a: [float(x).hex() for x in np.asarray(a, dtype=np.float64).reshape(-1)]  # noqa: E731
    model = {"num_links": links, "num_dof": links - 1, "parent": [0] + [int(m.parents[i]) for i in range(1, links)],
             "joint_index": [-1] + list(range(links - 1)), "joint_type": [-1] + [5] * (links - 1),
             "fixed_transforms_hex": hx(T), "masses_com_hex": hx(mc), "inertias_hex": hx(ine),
             "gravity_hex": hx(m.gravity.linear)}
    return PandaModel({"model": model, "contract_sha256": "pinocchio-parameterisation"})


def interval_torque(model, q, qd, qdd):
    box = lambda arr: [IA(arr[:, j].astype(np.float64)) for j in range(arr.shape[1])]  # noqa: E731
    tau = model.torque(box(q), box(qd), box(qdd))
    return np.stack([t.lo for t in tau], 1), np.stack([t.hi for t in tau], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--extract", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    c = load_contract(a.contract)
    M = PandaModel(c)
    z = np.load(a.extract)
    pin, pm, pnames, deltas, urdf_cfg_sha = pinocchio_model()
    if pnames != c["model"]["joint_names"]:
        raise SystemExit(f"joint order mismatch {pnames}")
    pdata = pm.createData()
    pm32 = f32_params(pin, pm)
    PM_pin = model_from_pinocchio(pin, pm)
    pdata32 = pm32.createData()
    out = {"contract_sha256": c["contract_sha256"], "pinocchio_version": pin.__version__,
           "pinocchio_configuration_deltas": deltas, "pinocchio_urdf_config_sha256": urdf_cfg_sha,
           "pinocchio_gravity": list(map(float, pm.gravity.linear)), "sets": {}}
    for key in ("gravity_only", "full", "B_probe", "A_probe"):
        q, qd, qdd = (z[f"ids_{key}_{s}"] for s in ("q", "qd", "qdd"))
        tn = z[f"ids_{key}_tau"].astype(np.float64)
        lo, hi = interval_torque(M, q, qd, qdd)
        tp = np.stack([pin.rnea(pm, pdata, q[i].astype(np.float64), qd[i].astype(np.float64), qdd[i].astype(np.float64)).copy()
                       for i in range(len(q))])
        t32 = np.stack([pin.rnea(pm32, pdata32, q[i].astype(np.float64), qd[i].astype(np.float64), qdd[i].astype(np.float64)).copy()
                        for i in range(len(q))])
        plo, phi = interval_torque(PM_pin, q, qd, qdd)
        pscale = np.maximum(1.0, np.abs(0.5 * (plo + phi)))
        alg_inside = int(((tp >= plo - 1e-12 * pscale) & (tp <= phi + 1e-12 * pscale)).all(1).sum())
        mid = 0.5 * (lo + hi)
        scale = np.maximum(1.0, np.abs(mid))
        out["sets"][key] = {
            "states": int(len(q)),
            "interval_width_max": float((hi - lo).max()),
            "algorithm_check_pinocchio_inside_interval_rnea_on_pinocchio_params": alg_inside,
            "algorithm_check_max_abs_diff": float(np.abs(tp - 0.5 * (plo + phi)).max()),
            "pinocchio_urdf_max_rel_diff": float((np.abs(tp - mid) / scale).max()),
            "pinocchio_f32params_inside_interval_pm1e-10rel": int(((t32 >= lo - 1e-10 * scale) & (t32 <= hi + 1e-10 * scale)).all(1).sum()),
            "max_abs_pinocchio_f32params_minus_interval_mid": float(np.abs(t32 - mid).max()),
            "max_abs_pinocchio_minus_interval_mid": float(np.abs(tp - mid).max()),
            "max_abs_native_minus_interval_mid": float(np.abs(tn - mid).max()),
            "max_rel_native_minus_interval_mid": float((np.abs(tn - mid) / scale).max()),
            "per_joint_max_abs_native_err": np.abs(tn - mid).max(0).tolist(),
            "max_abs_tau": float(np.abs(mid).max()),
        }
    # coefficient identity: native-derived vs interval-computed A, B, G at the same states
    q, qd, qdd = (z[f"ids_A_probe_{s}"] for s in ("q", "qd", "qdd"))
    box = lambda arr: [IA(arr[:, j].astype(np.float64)) for j in range(arr.shape[1])]  # noqa: E731
    A, B, G = M.coefficients(box(q), box(qd), box(qdd))
    mids = {k: np.stack([0.5 * (v.lo + v.hi) for v in vs], 1) for k, vs in (("A", A), ("B", B), ("G", G))}
    Gn = z["ids_gravity_only_tau"].astype(np.float64)
    Bn = 0.5 * (z["ids_B_probe_tau"].astype(np.float64) - Gn)
    An = z["ids_A_probe_tau"].astype(np.float64) - Gn
    out["coefficient_identity"] = {k: float(np.abs(mids[k] - v).max()) for k, v in (("A", An), ("B", Bn), ("G", Gn))}
    # self-consistency: tau = A + B*2*? -> at x=1, x'=0 the identity tau(q, q', q'') = A + G must hold
    lo, hi = interval_torque(M, q, qd, qdd)
    AG_lo = np.stack([(x + g).lo for x, g in zip(A, G)], 1)
    AG_hi = np.stack([(x + g).hi for x, g in zip(A, G)], 1)
    out["interval_identity_tau_eq_A_plus_G_overlap"] = bool(((AG_lo <= hi) & (lo <= AG_hi)).all())
    s = out["sets"]
    out["verdict"] = ("PARITY_PASS" if all(v["algorithm_check_pinocchio_inside_interval_rnea_on_pinocchio_params"] == v["states"]
                                           for v in s.values())
                      and max(v["pinocchio_urdf_max_rel_diff"] for v in s.values()) < 1e-6
                      and max(v["max_rel_native_minus_interval_mid"] for v in s.values()) < 1e-4
                      and out["interval_identity_tau_eq_A_plus_G_overlap"] else "PARITY_FAIL")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "pinocchio_configuration_deltas"}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""K4 P0 contract v2: the exact RIGID-BODY model cuRobo's float32 Panda parameters approximate.

Why v2: the independent world-frame verifier showed that cuRobo's stored float32 fixed rotations are not
orthonormal (panda_hand off by 3.4e-8, locked-finger transforms by 9.5e-7), so "cuRobo's float32 numbers
read exactly" (v1) is not a rigid-body model: link-frame and world-frame dynamics of it differ (~8e-8 N m).
v2 defines the model from the pinned URDF decimal values and cuRobo's documented defaults, each an exact
rational; rotations are the exact URDF rpy rotations R = Rz(yaw) Ry(pitch) Rx(roll). cuRobo's float32
representation of these values is cross-checked here and becomes part of native numerical error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIN = Path("/home/pkicki/curobo-pinned-c301cb00a87012c2808b1a60649c4065c9b3c95a")
URDF = PIN / "curobo/content/assets/robot/franka_description/franka_panda.urdf"
YML = PIN / "curobo/content/configs/robot/franka.yml"
# cuRobo link order (KinematicsParams) -> (urdf joint giving the link's origin, joint index or -1)
ORDER = [("panda_link0", None, -1), ("panda_link1", "panda_joint1", 0), ("panda_link2", "panda_joint2", 1),
         ("panda_link3", "panda_joint3", 2), ("panda_link4", "panda_joint4", 3), ("panda_link5", "panda_joint5", 4),
         ("panda_link6", "panda_joint6", 5), ("panda_link7", "panda_joint7", 6), ("panda_link8", "panda_joint8", -1),
         ("panda_hand", "panda_hand_joint", -1), ("panda_leftfinger", "panda_finger_joint1", -1),
         ("panda_rightfinger", "panda_finger_joint2", -1), ("attached_object", None, -1)]
DEFAULT_MASS, DEFAULT_I = "0.01", "0.0001"   # cuRobo default inertial for links without one
FINGER_LOCK = "0.04"                          # franka.yml lock_joints


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rpy_matrix_f64(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root = ET.parse(URDF).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    links = {l.get("name"): l for l in root.findall("link")}
    z = np.load(a.extract)
    model_links = []
    names = [o[0] for o in ORDER]
    for idx, (name, jname, jidx) in enumerate(ORDER):
        entry = {"name": name, "joint_index": jidx, "joint_type": "Z_ROT" if jidx >= 0 else "FIXED"}
        if jname is None:
            entry.update(parent=0 if idx else -1, origin_xyz=["0", "0", "0"], origin_rpy=["0", "0", "0"])
            if name == "attached_object":
                entry["parent"] = names.index("panda_hand")
        else:
            j = joints[jname]
            entry["parent"] = names.index(j.find("parent").get("link"))
            o = j.find("origin")
            xyz = o.get("xyz").split()
            rpy = o.get("rpy").split()
            if j.get("type") == "prismatic":  # locked finger: origin + lock * axis (exact)
                axis = [Fraction(v) for v in j.find("axis").get("xyz").split()]
                xyz = [str(Fraction(x) + Fraction(FINGER_LOCK) * ax) for x, ax in zip(xyz, axis)]
            if j.get("type") == "revolute" and j.find("axis").get("xyz").split() != ["0", "0", "1"]:
                raise SystemExit("unexpected axis")
            entry.update(origin_xyz=xyz, origin_rpy=rpy)
        inert = links.get(name).find("inertial") if name in links else None
        if inert is not None:
            if inert.find("origin").get("rpy").split() != ["0", "0", "0"]:
                raise SystemExit("rotated inertial frame not supported")
            entry["mass"] = inert.find("mass").get("value")
            entry["com"] = inert.find("origin").get("xyz").split()
            ii = inert.find("inertia")
            entry["inertia"] = [ii.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
            entry["inertial_source"] = "pinned URDF <inertial>"
        else:
            entry["mass"], entry["com"] = DEFAULT_MASS, ["0", "0", "0"]
            entry["inertia"] = [DEFAULT_I, DEFAULT_I, DEFAULT_I, "0", "0", "0"]
            entry["inertial_source"] = "cuRobo default (no URDF inertial)"
        model_links.append(entry)
    # -------- cross-check every value against cuRobo's stored float32 -----------------------------
    T = z["kin_fixed_transforms"]
    mc = z["kin_link_masses_com"]
    ine = z["kin_link_inertias"][:, :6]
    checks = []
    for i, e in enumerate(model_links):
        f32 = lambda s: np.float32(float(Fraction(s)))  # noqa: E731
        t_err = max(abs(float(Fraction(v)) - float(T[i][k, 3])) for k, v in enumerate(e["origin_xyz"]))
        ok_t = all(f32(v) == T[i][k, 3] for k, v in enumerate(e["origin_xyz"]))
        R = rpy_matrix_f64(*(float(Fraction(v)) for v in e["origin_rpy"]))
        rot_err = float(np.abs(R - T[i][:, :3].astype(np.float64)).max())
        ok_m = f32(e["mass"]) == mc[i, 3] and all(f32(v) == mc[i, k] for k, v in enumerate(e["com"]))
        ok_i = all(f32(v) == ine[i][k] for k, v in enumerate(e["inertia"]))
        checks.append({"link": e["name"], "translation_float32_equal": bool(ok_t), "translation_max_abs_diff": t_err,
                       "rotation_max_abs_diff": rot_err,
                       "mass_com_float32_equal": bool(ok_m), "inertia_float32_equal": bool(ok_i),
                       "parent_matches": e["parent"] == (int(z["kin_link_map"][i]) if i else -1) or (i == 0)})
    # float32 composition of the locked finger joints (rotation x lock offset) perturbs those translations
    if not all((c["translation_float32_equal"] or c["translation_max_abs_diff"] < 1e-7) and c["mass_com_float32_equal"] and c["inertia_float32_equal"]
               and c["rotation_max_abs_diff"] < 2e-6 and c["parent_matches"] for c in checks):
        print(json.dumps(checks, indent=1))
        raise SystemExit("URDF-decimal model does not match cuRobo's stored float32 model")
    model = {"scope": "CUROBO_PINNED_PANDA_RIGID_BODY_MODEL_V2_URDF_EXACT", "num_dof": 7, "links": model_links,
             "gravity": ["0", "0", "-9.81"], "rotation_convention": "R = Rz(yaw) Ry(pitch) Rx(roll), child joint frame in parent",
             "effort_lower": ["-87"] * 4 + ["-12"] * 3, "effort_upper": ["87"] * 4 + ["12"] * 3,
             "timeopt_torque_lower_scaled_float32_hex": [float(x).hex() for x in z["timeopt_limits_scaled"][3].astype(np.float64)],
             "timeopt_torque_upper_scaled_float32_hex": [float(x).hex() for x in z["timeopt_limits_scaled"][4].astype(np.float64)],
             "timeopt_limit_scale": "0.95", "joint_names": [f"panda_joint{k}" for k in range(1, 8)]}
    body = json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
    v1 = json.loads((ROOT / "docs/reports/k4/p0/panda_model_contract.json").read_text())
    contract = {"schema": "k4_panda_model_contract_v2", "contract_sha256": hashlib.sha256(body).hexdigest(),
                "supersedes_contract_sha256": v1["contract_sha256"], "model": model,
                "float32_crosscheck_vs_curobo": checks,
                "curobo_float32_rotation_non_orthonormality": {"panda_hand": 3.42e-08, "panda_leftfinger": 9.54e-07,
                                                              "panda_rightfinger": 9.54e-07},
                "provenance": {**v1["provenance"], "franka_urdf_sha256": sha(URDF), "franka_yml_sha256": sha(YML),
                               "v2_reason": __doc__.split("Why v2: ")[1].split("\n\"\"\"")[0].strip()},
                "physical_validity_findings": v1["physical_validity_findings"],
                "physical_validity_status": v1["physical_validity_status"]}
    a.out.write_text(json.dumps(contract, indent=1) + "\n")
    print(contract["contract_sha256"])
    print(json.dumps([{k: c[k] for k in ("link", "rotation_max_abs_diff")} for c in checks]))


if __name__ == "__main__":
    main()

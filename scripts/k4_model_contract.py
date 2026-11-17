#!/usr/bin/env python3
"""K4 P0: build the immutable, hash-identified Panda dynamics model contract.

Input: the extraction NPZ/JSON of scripts/k4_extract_curobo_model.py (pinned cuRobo TimeOpt's own
KinematicsParams). Output: docs/reports/k4/p0/panda_model_contract.json with
  * model: every parameter as exact float32 hex (parent map, joint map/type, fixed transforms,
    masses/CoMs, CoM inertias, gravity, effort and TimeOpt planning torque limits);
  * contract_sha256 over the canonical model JSON;
  * provenance: pinned cuRobo SHA, franka.yml / URDF SHA-256, link-by-link origin of each inertial;
  * physical_validity_findings: positive-definiteness / triangle inequalities / duplicates /
    cuRobo-injected default inertials. Findings are REPORTED, never "repaired".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.k4_panda_model import f32_hex  # noqa: E402

PIN = Path("/home/pkicki/curobo-pinned-c301cb00a87012c2808b1a60649c4065c9b3c95a")
URDF = PIN / "curobo/content/assets/robot/franka_description/franka_panda.urdf"
YML = PIN / "curobo/content/configs/robot/franka.yml"
LINKS = ["panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5",
         "panda_link6", "panda_link7", "panda_link8", "panda_hand", "panda_leftfinger", "panda_rightfinger",
         "attached_object"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    z = np.load(a.extract)
    meta = json.loads(a.extract.with_suffix(".json").read_text())
    n = z["kin_link_map"].shape[0]
    if n != len(LINKS):
        raise SystemExit("unexpected link count")
    ine = z["kin_link_inertias"][:, :6]
    model = {
        "scope": "CUROBO_PINNED_PANDA_RIGID_BODY_MODEL",
        "num_links": int(n), "num_dof": int(meta["num_dof"]), "link_names_assumed_from_urdf_order": LINKS,
        "joint_names": meta["joint_names"],
        "parent": [int(x) for x in z["kin_link_map"]], "joint_index": [int(x) for x in z["kin_joint_map"]],
        "joint_type": [int(x) for x in z["kin_joint_map_type"]],
        "mimic_scale_offset_hex": f32_hex(z["kin_joint_offset_map"]),
        "fixed_transforms_hex": f32_hex(z["kin_fixed_transforms"]),
        "masses_com_hex": f32_hex(z["kin_link_masses_com"]), "inertias_hex": f32_hex(ine),
        "gravity_hex": f32_hex(np.array(meta["gravity"], dtype=np.float32)),
        "effort_lower_hex": f32_hex(z["limit_effort"][0]), "effort_upper_hex": f32_hex(z["limit_effort"][1]),
        "timeopt_torque_lower_scaled_hex": f32_hex(z["timeopt_limits_scaled"][3]),
        "timeopt_torque_upper_scaled_hex": f32_hex(z["timeopt_limits_scaled"][4]),
        "timeopt_limit_scale": meta["limit_scale"],
        "payload": {"attached_object_mass_kg_float32": float(z["kin_link_masses_com"][12, 3]),
                    "note": "cuRobo default inertial for links without URDF inertial (0.01 kg, 1e-4 I); "
                            "no sourced 3 kg / 5 kg payload geometry, CoM, inertia or attachment"},
    }
    for i in range(n):
        if model["joint_type"][i] == 5 and not (z["kin_joint_offset_map"][2 * i] == 1 and z["kin_joint_offset_map"][2 * i + 1] == 0):
            raise SystemExit("mimic scale/offset on a movable joint is not supported")
    body = json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
    findings = []
    for i, name in enumerate(LINKS):
        m = float(z["kin_link_masses_com"][i, 3])
        r = ine[i].astype(np.float64)
        Ic = np.array([[r[0], r[3], r[4]], [r[3], r[1], r[5]], [r[4], r[5], r[2]]])
        ev = np.linalg.eigvalsh(Ic)
        tri = [r[0] + r[1] >= r[2], r[1] + r[2] >= r[0], r[0] + r[2] >= r[1]]
        issues = []
        if ev.min() <= 0:
            issues.append(f"CoM inertia not positive definite (eigenvalues {ev.round(6).tolist()})")
        if not all(tri):
            issues.append("principal-moment triangle inequality violated")
        for k in range(i):
            if np.array_equal(ine[k], ine[i]) and ine[i].any():
                issues.append(f"inertia tensor identical to {LINKS[k]}")
        if np.isclose(m, 0.01) and np.allclose(ine[i], [1e-4, 1e-4, 1e-4, 0, 0, 0]):
            issues.append("cuRobo default inertial (no URDF inertial): mass 0.01 kg, I = 1e-4")
        affects_torque = i >= 1
        findings.append({"link": name, "mass": m, "affects_joint_torque": affects_torque, "issues": issues})
    contract = {
        "schema": "k4_panda_model_contract_v1",
        "contract_sha256": hashlib.sha256(body).hexdigest(),
        "model": model,
        "provenance": {
            "upstream_curobo_sha": "c301cb00a87012c2808b1a60649c4065c9b3c95a",
            "franka_yml_sha256": sha(YML), "franka_urdf_sha256": sha(URDF),
            "extraction_npz_sha256": meta["arrays_sha256"],
            "extraction": "scripts/k4_extract_curobo_model.py: TimeOpt optimizer KinematicsParams after a warm-up plan",
            "inertials": "panda_link1..7 and panda_hand: <inertial> of the pinned franka_panda.urdf (origin of values "
                         "not documented in the repository); panda_link8, fingers, attached_object: cuRobo defaults",
            "effort_limits": "URDF <limit effort> 87 N m (joints 1-4) / 12 N m (joints 5-7), symmetric; "
                             "TimeOpt enforces limit_scale x effort at its constraint grid",
            "gravity": "DynamicsCfg default [0, 0, -9.81] world, base frame = world (fixed base, identity)",
            "friction_damping_rotor_inertia": "NOT MODELLED by cuRobo; not part of this contract",
        },
        "physical_validity_findings": findings,
        "physical_validity_status": ("PHYSICALLY_INCONSISTENT_PARAMETERS"
                                     if any(f["issues"] and f["affects_joint_torque"] and
                                            any("positive definite" in s or "triangle" in s for s in f["issues"])
                                            for f in findings) else "NO_INCONSISTENCY_DETECTED"),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(contract, indent=1) + "\n")
    print(contract["contract_sha256"], contract["physical_validity_status"])
    for f in findings:
        if f["issues"]:
            print(f["link"], f["issues"])


if __name__ == "__main__":
    main()

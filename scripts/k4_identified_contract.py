#!/usr/bin/env python3
"""K4: identified Panda dynamics contract (Gaz et al., RA-L 2019, feasible parameters p_hat).

Source: C. Gaz, M. Cognetti, A. Oliva, P. Robuffo Giordano, A. De Luca, "Dynamic Identification of the
Franka Emika Panda Robot With Retrieval of Feasible Parameters Using Penalty-Based Optimization", IEEE RA-L
4(4), 2019; supplementary material Tables VIII (masses, CoMs) and IX (inertia tensors), column p_hat,
from github.com/marcocognetti/FrankaEmikaPandaDynModel (GPL-3.0; only the published numeric values are
cited here, no code is copied). Kinematics: Franka modified-DH frames == the pinned URDF link frames
(checked algebraically and by the oracle parity below), so link origins/rpy are taken from contract v2.
Gravity 9.80665 (the authors' model). c1z is a "don't care" in the paper (no influence on the dynamics);
it is set to 0. Variants:
  identified           links 1-7 only (no hand / fingers / payload)
  identified_plus_hand links 1-7 + the pinned URDF panda_hand inertial (sensitivity: whether the
                       identified link 7 already contains the Franka Hand is not stated in the sources read)
Whether p_hat inertias are about the CoM is decided EMPIRICALLY against the authors' own published
M(q) and g(q) functions (scripts/k4_gaz_oracle.py); the convention that matches is recorded.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Table VIII / IX, column p_hat (exact decimal strings as printed)
M_ = ["4.970684", "0.646926", "3.228604", "3.587895", "1.225946", "1.666555", "7.35522e-01"]
C_ = [["3.875e-03", "2.081e-03", "0"], ["-3.141e-03", "-2.872e-02", "3.495e-03"],
      ["2.7518e-02", "3.9252e-02", "-6.6502e-02"], ["-5.317e-02", "1.04419e-01", "2.7454e-02"],
      ["-1.1953e-02", "4.1065e-02", "-3.8437e-02"], ["6.0149e-02", "-1.4117e-02", "-1.0517e-02"],
      ["1.0517e-02", "-4.252e-03", "6.1597e-02"]]
I_ = [  # xx, xy, xz, yy, yz, zz
    ["7.0337e-01", "-1.3900e-04", "6.7720e-03", "7.0661e-01", "1.9169e-02", "9.1170e-03"],
    ["7.9620e-03", "-3.9250e-03", "1.0254e-02", "2.8110e-02", "7.0400e-04", "2.5995e-02"],
    ["3.7242e-02", "-4.7610e-03", "-1.1396e-02", "3.6155e-02", "-1.2805e-02", "1.0830e-02"],
    ["2.5853e-02", "7.7960e-03", "-1.3320e-03", "1.9552e-02", "8.6410e-03", "2.8323e-02"],
    ["3.5549e-02", "-2.1170e-03", "-4.0370e-03", "2.9474e-02", "2.2900e-04", "8.6270e-03"],
    ["1.9640e-03", "1.0900e-04", "-1.1580e-03", "4.3540e-03", "3.4100e-04", "5.4330e-03"],
    ["1.2516e-02", "-4.2800e-04", "-1.1960e-03", "1.0027e-02", "-7.4100e-04", "4.8150e-03"]]


def to_com(m, c, I_origin):
    """Inertia about CoM from inertia about the link origin (parallel axis), exact rationals."""
    m = Fraction(m)
    c = [Fraction(v) for v in c]
    xx, xy, xz, yy, yz, zz = (Fraction(v) for v in I_origin)
    cc = c[0] ** 2 + c[1] ** 2 + c[2] ** 2
    return [xx - m * (cc - c[0] ** 2), yy - m * (cc - c[1] ** 2), zz - m * (cc - c[2] ** 2),
            xy + m * c[0] * c[1], xz + m * c[0] * c[2], yz + m * c[1] * c[2]]


def build(convention: str, with_hand: bool):
    v2 = json.loads((ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json").read_text())
    model = copy.deepcopy(v2["model"])
    for i, e in enumerate(model["links"]):
        if 1 <= i <= 7:
            k = i - 1
            e["mass"], e["com"] = M_[k], C_[k]
            xx, xy, xz, yy, yz, zz = I_[k]
            ine = [xx, yy, zz, xy, xz, yz]  # contract order: ixx iyy izz ixy ixz iyz (about CoM)
            if convention == "origin":
                ine = [str(v) for v in to_com(M_[k], C_[k], [xx, xy, xz, yy, yz, zz])]
            e["inertia"] = ine
            e["inertial_source"] = f"Gaz et al. RA-L 2019 suppl. Tables VIII/IX p_hat (inertia convention: {convention})"
        elif i == 9 and with_hand:
            pass  # keep the pinned URDF panda_hand inertial
        elif i >= 8:
            e["mass"], e["com"], e["inertia"] = "0", ["0", "0", "0"], ["0", "0", "0", "0", "0", "0"]
            e["inertial_source"] = "not part of the identified model (massless)"
    model["gravity"] = ["0", "0", "-9.80665"]
    model["scope"] = "IDENTIFIED_PANDA_GAZ2019" + ("_PLUS_URDF_HAND" if with_hand else "")
    body = json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
    return {"schema": "k4_panda_model_contract_v2", "contract_sha256": hashlib.sha256(body).hexdigest(), "model": model}


def oracle_parity(contract, oracle_dir: Path, n=60, seed=7):
    import k4_gaz_oracle as O
    from curobo_certify.k4_interval import IA
    from curobo_certify.k4_panda_model import PandaModel
    M = PandaModel(contract)
    G = O.load(oracle_dir / "get_GravityVector.m")
    Mm = O.load(oracle_dir / "get_MassMatrix.m")
    rng = np.random.default_rng(seed)
    eg, em = 0.0, 0.0
    for _ in range(n):
        q = rng.uniform(-2.5, 2.5, 7)
        qi = [IA(v) for v in q]
        g = np.array([t.mid() for t in M.rnea(qi, None, None, gravity=True)], dtype=float)
        eg = max(eg, float(np.abs(g - np.array(G(list(q)))).max()))
        Mo = np.array(Mm(list(q))).reshape(7, 7)
        cols = []
        for j in range(7):
            e = [IA(1.0 if k == j else 0.0) for k in range(7)]
            cols.append([float(t.mid()) for t in M.rnea(qi, None, e, gravity=False)])
        em = max(em, float(np.abs(np.array(cols).T - Mo).max()))
    return {"gravity_max_abs_diff": eg, "mass_matrix_max_abs_diff": em, "states": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs/reports/k4/identified")
    a = ap.parse_args()
    res = {}
    for conv in ("com", "origin"):
        res[conv] = oracle_parity(build(conv, False), a.oracle_dir)
        print(conv, res[conv])
    best = min(res, key=lambda k: res[k]["mass_matrix_max_abs_diff"])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for with_hand in (False, True):
        c = build(best, with_hand)
        # physical consistency audit of links 1-7 (CoM inertia positive definite + triangle inequalities)
        audit = []
        for i in range(1, 8):
            x = [float(Fraction(v)) for v in c["model"]["links"][i]["inertia"]]
            Ic = np.array([[x[0], x[3], x[4]], [x[3], x[1], x[5]], [x[4], x[5], x[2]]])
            ev = np.linalg.eigvalsh(Ic)
            audit.append({"link": i, "eigenvalues": ev.tolist(), "positive_definite": bool(ev.min() > 0),
                          "triangle": bool(ev[0] + ev[1] >= ev[2] - 1e-12)})
        c["oracle_parity"] = {"convention_selected": best, "by_convention": res,
                              "oracle": "authors' published get_GravityVector.m / get_MassMatrix.m (diagnostic)"}
        c["physical_consistency_links_1_7"] = audit
        c["provenance"] = {
            "citation": "Gaz, Cognetti, Oliva, Robuffo Giordano, De Luca, IEEE RA-L 4(4):4147-4154, 2019",
            "values": "supplementary material Tables VIII and IX, column p_hat",
            "supplement_sha256": hashlib.sha256((a.oracle_dir / "supp.pdf").read_bytes()).hexdigest(),
            "repository": "github.com/marcocognetti/FrankaEmikaPandaDynModel (GPL-3.0); numeric values cited, no code copied",
            "kinematics": "contract v2 link frames (pinned URDF == Franka modified DH)",
            "gravity": "9.80665 (authors' model)", "friction": "Gaz sigmoid model is NOT part of the rigid contract; a "
            "conservative |tau_F| <= A_i bound is applied separately as a limit tightening",
            "friction_A_Nm": [0.54615, 0.87224, 0.64068, 1.2794, 0.83904, 0.30301, 0.56489],
            "hand": "identified_plus_hand variant adds the pinned URDF panda_hand inertial (sensitivity)",
            "effort_limits": "unchanged: FER tau_max 87/12 N m"}
        name = "identified_plus_hand" if with_hand else "identified"
        (a.out_dir / f"panda_contract_{name}.json").write_text(json.dumps(c, indent=1) + "\n")
        print(name, c["contract_sha256"], all(x["positive_definite"] and x["triangle"] for x in audit))


if __name__ == "__main__":
    main()

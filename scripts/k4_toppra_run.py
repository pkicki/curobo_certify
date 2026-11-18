#!/usr/bin/env python3
"""TOPP-RA baseline on cuRobo geometric paths (run in runs/toppravenv: toppra 0.6.3, MIT).

Same geometry (exact cuRobo spline spans of the frozen output), same planning-model contract and the SAME limits
TimeOpt enforces: joint velocity and acceleration (NPZ limits x 0.95, float32 as cuRobo) and torque (contract's
0.95-scaled float32 limits). TOPP-RA cannot represent cuRobo's jerk limit (documented difference). The torque
constraint is TOPP-RA's canonical second-order form a(s) sdd + b(s) sd^2 + c(s) in [lo, hi] with a = 2B, b = A,
c = G from the compiled K4 coefficient routine (exact model, float64 point evaluation at TOPP-RA's gridpoints).
Grid conditions: 'auto' (toppra default: propose_gridpoints, max_err 1e-3, >= 100 points) and uniform N.
Output per (path, grid): return code, gridpoints and sd (float64 hex), solve seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--payload", default=None)
    ap.add_argument("--grids", nargs="+", default=["auto", "50", "100", "200", "500", "1000", "2000"])
    ap.add_argument("--trim", action="store_true", help="parametrize only the moving part (drop cuRobo's stationary padding)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import toppra as ta
    import toppra.algorithm as algo
    import toppra.constraint as constraint
    from toppra.constraint import DiscretizationType
    from toppra.constraint.linear_second_order import SecondOrderConstraint, canlinear_colloc_to_interpolate
    from toppra.interpolator import AbstractGeometricPath
    ta.setup_logging("WARNING")
    from curobo_certify import k4_compiled as KC
    from curobo_certify.k4_interval import IA
    from curobo_certify.k4_panda_model import PandaModel
    from curobo_certify.k4_sampling import SpanTable
    from curobo_certify.spline_polynomials import extract_timeopt_spans
    c = json.loads(a.contract.read_text())
    if a.payload:
        from curobo_certify.k4_payload import with_payload
        c = with_payload(c, a.payload)
    M = PandaModel(c)
    tlo = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_lower_scaled_float32_hex"]])
    thi = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_upper_scaled_float32_hex"]])
    tmp = tempfile.TemporaryDirectory()
    build = KC.build(Path(tmp.name) / "p.so")
    CC = KC.CompiledCoefficients(M, Path(tmp.name) / "p.so")

    class SplinePath(AbstractGeometricPath):
        def __init__(self, table):
            self.T = table

        def __call__(self, ss, order=0):
            s = np.atleast_1d(np.asarray(ss, float))
            q, q1, q2, _, _ = self.T.evaluate(s)
            r = (q, q1, q2)[order]
            return r[0] if np.ndim(ss) == 0 else r

        @property
        def dof(self):
            return 7

        @property
        def path_interval(self):
            return np.array([0.0, 1.0])

    class ExactTorque(SecondOrderConstraint):
        def __init__(self, table):
            dof = 7
            F = np.vstack((np.eye(dof), -np.eye(dof)))
            g = np.r_[thi, -tlo]
            super().__init__(None, lambda _: F, lambda _: g, dof, None, DiscretizationType.Interpolation)
            self.T = table

        def compute_constraint_params(self, path, gridpoints):
            q, q1, q2, _, _ = self.T.evaluate(np.asarray(gridpoints, float))
            box = lambda x: [IA(x[:, j].copy()) for j in range(7)]  # noqa: E731
            A, B, G = CC.coefficients(box(q), box(q1), box(q2))
            mid = lambda L: np.stack([0.5 * (v.lo + v.hi) for v in L], 1)  # noqa: E731
            a_vec, b_vec, c_vec = 2 * mid(B), mid(A), mid(G)
            n = len(gridpoints)
            F_vec = np.repeat(self.constraint_F(None)[None], n, 0)
            g_vec = np.repeat(self.constraint_g(None)[None], n, 0)
            return canlinear_colloc_to_interpolate(a_vec, b_vec, c_vec, F_vec, g_vec, None, None, gridpoints)

    rows = []
    for f in a.npz:
        with np.load(f) as z:
            g, tc = z["effective_geometry"], z["timing_controls"]
            vl = (np.minimum(np.abs(z["velocity_limits"][0]), np.abs(z["velocity_limits"][1])).astype(np.float32) * np.float32(0.95)).astype(float)
            al = (np.minimum(np.abs(z["acceleration_limits"][0]), np.abs(z["acceleration_limits"][1])).astype(np.float32) * np.float32(0.95)).astype(float)
            native_T = float(z["native_duration"][0])
        base = extract_timeopt_spans(g, tc)
        if a.trim:
            from curobo_certify.k4_toppra import trimmed
            base, _ = trimmed(base)
        T = SpanTable(base)
        path = SplinePath(T)
        for gc in a.grids:
            grid = None if gc == "auto" else np.linspace(0.0, 1.0, int(gc) + 1)
            cons = [constraint.JointVelocityConstraint(np.c_[-vl, vl]),
                    constraint.JointAccelerationConstraint(np.c_[-al, al], discretization_scheme=DiscretizationType.Interpolation),
                    ExactTorque(T)]
            t0 = time.perf_counter()
            try:
                inst = algo.TOPPRA(cons, path, gridpoints=grid, solver_wrapper="seidel", parametrizer="ParametrizeConstAccel")
                sdd, sd, v, K = inst.compute_parameterization(0, 0, return_data=True)
                ok = sd is not None and not np.isnan(sd).any()
                rec = {"return_code": str(inst.problem_data.return_code), "ok": bool(ok),
                       "gridpoints_hex": [float(x).hex() for x in inst.gridpoints],
                       "sd_hex": [float(max(x, 0.0)).hex() for x in sd] if ok else None}
            except Exception as e:  # retained
                rec = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            rec.update({"path": f.stem, "grid": gc, "seconds": time.perf_counter() - t0, "native_duration_s": native_T})
            rows.append(rec)
            print(json.dumps({k: rec.get(k) for k in ("path", "grid", "ok", "seconds")} |
                             {"n": len(rec.get("gridpoints_hex") or [])}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"scope": "TOPPRA_BASELINE_PARAMETRIZATIONS", "toppra": __import__("importlib.metadata").metadata.version("toppra"), "build": build,
                                 "contract_sha256": c["contract_sha256"], "payload": a.payload, "trim": a.trim,
                                 "limits": "velocity/acceleration NPZ x0.95 (float32), torque contract 0.95 float32; no jerk",
                                 "rows": rows}) + "\n")


if __name__ == "__main__":
    main()

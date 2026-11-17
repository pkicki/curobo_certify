"""K4 P6: certification-aware retiming of a FIXED geometry (squared-speed timing spline only).

x(s) = sum_k r_k N_k(s) is linear in the 32 timing controls r. On every leaf of the common q/x partition,
with r >= 0, these are SUFFICIENT linear constraints (bounds from exact Bernstein hulls and interval
coefficient enclosures; never necessary):
  velocity   q'_j(s)^2 x(s) <= vmax_j^2                -> sum_k r_k sup(q'^2 N_k) <= vmax^2
  accel      |q''_j x + 1/2 q'_j x'| <= amax_j         -> sum_k r_k sup/inf(q'' N_k + q' N'_k / 2)
  torque     lo_j <= G_j + A_j x + B_j x' <= hi_j       -> G.hi + sum_k r_k sup(A N_k + B N'_k) <= hi, etc.
Objective: the TimeOpt duration T(r) = int_0^1 x^-1/2 ds (convex in r), evaluated by Gauss quadrature
(objective only). Jerk (nonlinear) and every other K3/K4 obligation are NOT trusted from the program:
the candidate is rounded to float32 controls and must pass the complete MODEL_K3 (compiled producer +
verifier) and MODEL_K4 (tm1 + independent verifier) proofs; otherwise r = r_native + lam (r_opt - r_native)
is bisected towards the native (certified) controls. A candidate that is not fully certified is never
returned. Duration of the accepted controls is reported as the rigorous time-law bracket (time_law_grid).
"""
from __future__ import annotations

from fractions import Fraction as Q

import numpy as np

from .k4_reference import bernstein, dpoly
from .spline_polynomials import extract_timeopt_spans


def _hull(poly_coeffs):
    b = bernstein(tuple(poly_coeffs))
    return float(min(b)), float(max(b))


def basis_spans(geometry):
    """Per common span: Bernstein-hull data of each timing basis function N_k and N'_k (exact)."""
    out = []
    for k in range(32):
        e = np.zeros(32, dtype=np.float32)
        e[k] = 1.0
        out.append(extract_timeopt_spans(geometry, e))
    return out


def build_constraints(geometry, cache, vlim, alim, tlo, thi, depth=1):
    """Linear inequalities G r <= h (sufficient) on leaves of uniform depth `depth` per span."""
    from .k4_reuse import _leaf_bounds, _sub
    spans0 = basis_spans(geometry)
    nsp = len(spans0[0])
    rows, rhs = [], []
    keys = [(si, p) for si in range(nsp) for p in _paths(depth)]
    cache.ensure(keys)
    d = cache.dof
    for si, path in keys:
        sp = spans0[0][si]
        h = sp.right - sp.left
        qp = [bernstein(dpoly(p, 1, h)) for p in sp.q]
        qpp = [bernstein(dpoly(p, 2, h)) for p in sp.q]
        qp_l = [_sub(b, path) for b in qp]
        qpp_l = [_sub(b, path) for b in qpp]
        Nb, Npb = [], []
        for k in range(32):
            xk = spans0[k][si]
            Nb.append(_sub(bernstein(tuple(xk.x)), path))
            Npb.append(_sub(bernstein(dpoly(xk.x, 1, h)), path))
        A, B, G = cache.store[(si, path)][:3] if cache.mode == "box" else _box_from_tm1(cache.store[(si, path)])
        for j in range(d):
            q1 = (float(min(qp_l[j])), float(max(qp_l[j])))
            q2 = (float(min(qpp_l[j])), float(max(qpp_l[j])))
            q1sq_hi = max(q1[0] ** 2, q1[1] ** 2) * (1 + 1e-12)
            v_row, a_up, a_lo, t_up, t_lo = [], [], [], [], []
            for k in range(32):
                n = (float(min(Nb[k])), float(max(Nb[k])))
                npr = (float(min(Npb[k])), float(max(Npb[k])))
                v_row.append(q1sq_hi * n[1])
                acc = _imul(q2, n) + _imul((0.5 * q1[0], 0.5 * q1[1]), npr)
                a_up.append(acc[1]); a_lo.append(acc[0])
                tq = _imul((float(A[j].lo), float(A[j].hi)), n) + _imul((float(B[j].lo), float(B[j].hi)), npr)
                t_up.append(tq[1]); t_lo.append(tq[0])
            pad = 1 + 1e-9
            rows.append(v_row); rhs.append(float(vlim[j]) ** 2 / pad)
            rows.append(a_up); rhs.append(float(alim[j]) / pad)
            rows.append([-x for x in a_lo]); rhs.append(float(alim[j]) / pad)
            rows.append(t_up); rhs.append((float(thi[j]) - float(G[j].hi)) / pad)
            rows.append([-x for x in t_lo]); rhs.append((float(G[j].lo) - float(tlo[j])) / pad)
    return np.array(rows), np.array(rhs)


class _Pair(tuple):
    def __add__(self, o):
        return _Pair((self[0] + o[0], self[1] + o[1]))


def _imul(a, b):
    p = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return _Pair((min(p), max(p)))


def _paths(depth):
    out = [""]
    for _ in range(depth):
        out = [p + c for p in out for c in "LR"]
    return out


def _box_from_tm1(e):
    A0, B0, G0, A1, B1, G1, H = e
    from .k4_interval import IA
    Hb = IA(-float(H) * (1 + 1e-12), float(H) * (1 + 1e-12))
    return ([A0[j] + A1[j] * Hb for j in range(len(A0))], [B0[j] + B1[j] * Hb for j in range(len(B0))],
            [G0[j] + G1[j] * Hb for j in range(len(G0))])


def duration_objective(geometry, n_gauss=8):
    """Callable T(r) and gradient by Gauss-Legendre on the 29 timing spans (objective only)."""
    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
    s = np.concatenate([(m + 0.5 * (nodes + 1)) / 29 for m in range(29)])
    w = np.concatenate([0.5 * weights / 29 for _ in range(29)])
    from scipy.interpolate import BSpline
    knots = np.r_[[0.0] * 3, np.arange(30) / 29.0, [1.0] * 3]
    Nmat = np.stack([BSpline(knots, np.eye(32)[k], 3)(s) for k in range(32)], 1)

    def T(r):
        x = Nmat @ r
        return float(np.sum(w / np.sqrt(np.maximum(x, 1e-12))))

    def dT(r):
        x = Nmat @ r
        return -0.5 * (w / np.maximum(x, 1e-12) ** 1.5) @ Nmat
    return T, dT


def optimize_timing(geometry, r_native, Gm, h, *, r_min_frac=0.25):
    from scipy.optimize import LinearConstraint, minimize
    T, dT = duration_objective(geometry)
    r0 = np.asarray(r_native, float)
    lb = np.maximum(r0 * r_min_frac, 1e-6)
    feasible0 = bool((Gm @ r0 <= h + 1e-9).all())
    res = minimize(T, r0 if feasible0 else lb, jac=dT, method="trust-constr",
                   constraints=[LinearConstraint(Gm, -np.inf, h)],
                   bounds=[(l_, None) for l_ in lb], options={"maxiter": 400, "verbose": 0})
    return res.x, {"native_feasible_in_sufficient_program": feasible0, "T_native_obj": T(r0), "T_opt_obj": T(res.x),
                   "optimizer_status": int(res.status), "constraint_violation": float(np.max(Gm @ res.x - h))}

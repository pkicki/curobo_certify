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


# ---------------------------------------------------------------------------------------------------
# Practical optimiser (HEURISTIC constraint model; rigour comes ONLY from the post-hoc K3+K4 proofs).
# Dense sampled constraints (TOPP-style) with a safety factor, solved by sequential linear programming
# (HiGHS) on the convex duration objective with a multiplicative trust region and exact line search.
# ---------------------------------------------------------------------------------------------------
def sampled_constraints(geometry, model, vlim, alim, tlo, thi, *, jlim=None, per_span=16, safety=0.98):
    from scipy.interpolate import BSpline
    from .k4_interval import IA
    from .k4_reference import eval_poly
    spans = extract_timeopt_spans(geometry, np.zeros(32, dtype=np.float32) + np.float32(1.0))
    s_all, qp_all, qpp_all, q_all, qppp_all = [], [], [], [], []
    for sp in spans:
        h = sp.right - sp.left
        for t in range(per_span):
            u = Q(t, per_span - 1)
            s_all.append(float(sp.left + h * u))
            q_all.append([float(eval_poly(p, u)) for p in sp.q])
            qp_all.append([float(eval_poly(dpoly(p, 1, h), u)) for p in sp.q])
            qpp_all.append([float(eval_poly(dpoly(p, 2, h), u)) for p in sp.q])
            qppp_all.append([float(eval_poly(dpoly(p, 3, h), u)) for p in sp.q])
    s = np.array(s_all); q = np.array(q_all); qp = np.array(qp_all); qpp = np.array(qpp_all); qppp = np.array(qppp_all)
    knots = np.r_[[0.0] * 3, np.arange(30) / 29.0, [1.0] * 3]
    N = np.stack([BSpline(knots, np.eye(32)[k], 3)(s) for k in range(32)], 1)
    Np = np.stack([BSpline(knots, np.eye(32)[k], 3).derivative()(s) for k in range(32)], 1)
    Npp = np.stack([BSpline(knots, np.eye(32)[k], 3).derivative(2)(s) for k in range(32)], 1)
    A, B, G = model.coefficients([IA(q[:, j]) for j in range(7)], [IA(qp[:, j]) for j in range(7)],
                                 [IA(qpp[:, j]) for j in range(7)])
    rows, rhs = [], []
    for j in range(7):
        a_, b_, g_ = A[j].mid(), B[j].mid(), G[j].mid()
        rows.append((qp[:, j:j + 1] ** 2) * N); rhs.append(np.full(len(s), safety * vlim[j] ** 2))
        acc = qpp[:, j:j + 1] * N + 0.5 * qp[:, j:j + 1] * Np
        rows.append(acc); rhs.append(np.full(len(s), safety * alim[j]))
        rows.append(-acc); rhs.append(np.full(len(s), safety * alim[j]))
        tq = a_[:, None] * N + b_[:, None] * Np
        rows.append(tq); rhs.append(safety * thi[j] - g_)
        rows.append(-tq); rhs.append(g_ - safety * tlo[j])
    jerk = {"N": N, "L": [qppp[:, j:j + 1] * N + 1.5 * qpp[:, j:j + 1] * Np + 0.5 * qp[:, j:j + 1] * Npp for j in range(7)],
            "bound": [(safety * float(jl)) ** 2 for jl in jlim]} if jlim is not None else None
    return np.vstack(rows), np.concatenate(rhs), jerk


def _jerk_ok(jerk, r):
    if jerk is None:
        return True
    x = jerk["N"] @ r
    return all((x * (L @ r) ** 2 <= b).all() for L, b in zip(jerk["L"], jerk["bound"]))


def _jerk_rows(jerk, r):
    """Linearisation of g_j(r) = x(r) L_j(r)^2 <= bound at r (SLP rows: grad . r' <= bound - g + grad . r)."""
    x = jerk["N"] @ r
    rows, rhs = [], []
    for L, b in zip(jerk["L"], jerk["bound"]):
        l_ = L @ r
        g = x * l_ ** 2
        grad = (l_ ** 2)[:, None] * jerk["N"] + (2 * x * l_)[:, None] * L
        rows.append(grad); rhs.append(b - g + grad @ r)
    return np.vstack(rows), np.concatenate(rhs)


def slp_optimize(geometry, r0, Gm, h, *, jerk=None, iters=40, trust=0.2, r_floor_frac=0.25):
    from scipy.optimize import linprog
    T, dT = duration_objective(geometry)
    r = np.asarray(r0, float).copy()
    floor = np.maximum(r * r_floor_frac, 1e-6)
    hist = [T(r)]
    feasible0 = bool((Gm @ r <= h + 1e-9).all()) and _jerk_ok(jerk, r)
    for _ in range(iters):
        g = dT(r)
        lb = np.maximum(floor, r * (1 - trust))
        ub = r * (1 + trust)
        A_ub, b_ub = Gm, h
        if jerk is not None:
            Jr, Jh = _jerk_rows(jerk, r)
            A_ub, b_ub = np.vstack([Gm, Jr]), np.concatenate([h, Jh])
        res = linprog(g, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)), method="highs")
        if res.status != 0:
            break
        d = res.x - r
        best, bt = None, T(r)
        for step in (1.0, 0.5, 0.25, 0.125):
            cand = r + step * d
            if (Gm @ cand <= h + 1e-9).all() and _jerk_ok(jerk, cand) and T(cand) < bt - 1e-12:
                best, bt = cand, T(cand)
                break
        if best is None:
            trust *= 0.5
            if trust < 1e-3:
                break
            continue
        r = best
        hist.append(bt)
    return r, {"native_feasible_in_sampled_program": feasible0, "T_obj_native": hist[0], "T_obj_opt": hist[-1],
               "iterations": len(hist) - 1}

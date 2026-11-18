"""Sampled torque checks (BASELINES) and a rigorous max-utilization bracket (ground truth) for K4.

Evidence classes, kept distinct:
  SAMPLED   torque of the exact contract model at N points (float64 inputs, compiled interval RNEA midpoint):
            what a careful sampled checker would see. Says nothing between samples.
  RIGOROUS  max-utilization bracket [lb, ub] over the whole continuous path:
            lb = rigorous point enclosure (exact rational path evaluation, interval RNEA) at the densest-sample
                 argmax -> the true max is >= lb;
            ub = smallest limit scale f at which the compiled K4 producer CERTIFIES -> the true max is <= f.
Utilization = max over joints of max(tau / upper, tau / lower) for the given (lower < 0 < upper) limits.
"""
from __future__ import annotations

import time
from fractions import Fraction as Q

import numpy as np

from .k4_interval import IA
from .k4_path import _point_boxes, torque_enclosure
from .k4_reference import dpoly


class SpanTable:
    """float64 power-basis coefficients of q, q', q'' (w.r.t. global s), x, x' per span."""

    def __init__(self, spans, dof=7):
        self.dof = dof
        self.left = np.array([float(sp.left) for sp in spans])
        self.right = np.array([float(sp.right) for sp in spans])
        self.h = self.right - self.left
        pad = lambda p, n: [float(c) for c in p] + [0.0] * (n - len(p))  # noqa: E731
        self.cq = np.array([[pad(p, 4) for p in sp.q] for sp in spans])                     # [S,dof,4]
        self.cq1 = np.array([[pad(dpoly(p, 1, sp.right - sp.left), 4) for p in sp.q] for sp in spans])
        self.cq2 = np.array([[pad(dpoly(p, 2, sp.right - sp.left), 4) for p in sp.q] for sp in spans])
        self.cx = np.array([pad(sp.x, 4) for sp in spans])                                 # [S,4]
        self.cx1 = np.array([pad(dpoly(sp.x, 1, sp.right - sp.left), 4) for sp in spans])
        self.spans = spans

    def locate(self, s):
        s = np.clip(np.asarray(s, float), 0.0, 1.0)
        k = np.clip(np.searchsorted(self.right, s, side="left"), 0, len(self.right) - 1)
        u = np.clip((s - self.left[k]) / self.h[k], 0.0, 1.0)
        return k, u

    @staticmethod
    def _horner(c, u):
        out = np.zeros(c.shape[:-1])
        for i in range(c.shape[-1] - 1, -1, -1):
            out = out * u + c[..., i]
        return out

    def evaluate(self, s):
        k, u = self.locate(s)
        U = u[:, None]
        q = self._horner(self.cq[k], U)
        q1 = self._horner(self.cq1[k], U)
        q2 = self._horner(self.cq2[k], U)
        x = self._horner(self.cx[k], u)
        x1 = self._horner(self.cx1[k], u)
        return q, q1, q2, x, x1


def sampled_torque(cc, table: SpanTable, s):
    """SAMPLED: exact-model torque at path parameters s (float64 inputs), via compiled interval RNEA."""
    q, q1, q2, x, x1 = table.evaluate(s)
    box = lambda a: [IA(a[:, j].copy()) for j in range(table.dof)]  # noqa: E731
    A, B, G = cc.coefficients(box(q), box(q1), box(q2))
    X, X1 = IA(x.copy()), IA(x1.copy())
    tau = [G[j] + A[j] * X + B[j] * X1 for j in range(table.dof)]
    return np.stack([0.5 * (t.lo + t.hi) for t in tau], 1)


def utilization(tau, lo, hi):
    return np.maximum(tau / hi, tau / lo)                   # [N, dof]


def grid_uniform_s(n):
    return np.linspace(0.0, 1.0, n)


def grid_uniform_t(n, times, progress):
    """N uniform-time samples mapped to s through the native progress table (cuRobo's own time law)."""
    t = np.linspace(0.0, float(times[-1]), n)
    return np.interp(t, np.asarray(times, float), np.asarray(progress, float))


def rigorous_point_util(model, table: SpanTable, s_float, lo, hi):
    """RIGOROUS lower bound on max utilization: exact rational evaluation at s (rounded to a dyadic) + interval RNEA."""
    s = Q(s_float)
    k = int(table.locate(np.array([s_float]))[0][0])
    sp = table.spans[k]
    h = sp.right - sp.left
    u = min(max((s - sp.left) / h, Q(0)), Q(1))
    raw = [tuple(p) for p in sp.q] + [dpoly(p, 1, h) for p in sp.q] + [dpoly(p, 2, h) for p in sp.q] + [tuple(sp.x), dpoly(sp.x, 1, h)]
    pq, pqp, pqpp, px, pxp, ex = _point_boxes([raw], [u], table.dof)
    if ex[0][3 * table.dof] < 0:
        return None
    tau = torque_enclosure(model, pq, pqp, pqpp, px, pxp)
    lo_t = np.array([float(np.asarray(t.lo).ravel()[0]) for t in tau])
    hi_t = np.array([float(np.asarray(t.hi).ravel()[0]) for t in tau])
    # lower bound of max(tau/hi, tau/lo): use the enclosure end that minimizes each ratio
    lb = np.maximum(lo_t / hi, hi_t / lo)          # min over the enclosure of max(tau/hi, tau/lo) when 0 is excluded
    lb = np.nextafter(lb, -np.inf)                  # absorb the float division rounding (outward = downward)
    j = int(np.argmax(lb))
    return float(lb[j]), {"span": k, "u": str(u), "s": float(s), "joint": j,
                          "torque_enclosure": [float(lo_t[j]), float(hi_t[j])]}


def util_upper_bound(producer, geometry, timing, lo, hi, start, tol=1e-7, max_iter=40, threads=8):
    """RIGOROUS upper bound: smallest scale f (to tol) with K4 CERTIFIED at limits f*[lo, hi], bisection from start.
    Returns (f_ub, f_violated_or_None, n_calls). UNKNOWN results are treated as 'not certified' (conservative)."""
    calls = 0

    def cert(f):
        nonlocal calls
        calls += 1
        return producer.certify(geometry, timing, lo * f, hi * f, threads=threads)["status"] == "CERTIFIED"
    a, b = start, start
    while not cert(b):
        a, b = b, b * 1.01 + 1e-3
        if calls > max_iter:
            return None, a, calls
    if b == start:   # certified at the starting point: walk down to find a non-certified lower end
        a = start * 0.999
        while cert(a) and calls < max_iter:
            b, a = a, a * 0.999
    while b - a > tol and calls < max_iter:
        m = 0.5 * (a + b)
        if cert(m):
            b = m
        else:
            a = m
    return b, a, calls


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return r, time.perf_counter() - t0

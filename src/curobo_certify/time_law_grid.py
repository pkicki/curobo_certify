"""Rigorous high-resolution enclosure of the exact TimeOpt time law T(s)=∫_0^s dσ/√x(σ).

Scope: EXACT_MODEL time law for the serialized clamped cubic squared-speed spline
x(s) of the 32 float32 timing controls (exact rational coefficients). Produces, on
the dyadic grid s_k = k/N (k = 0..N):

    T_lo[k] <= T(s_k) <= T_hi[k]          (all k)
    fmin[k] <= 1/√x(s) <= fmax[k]          for s in [s_k, s_{k+1}]

Method (all binary64 with explicit outward rounding by np.nextafter):
  * smooth cells: midpoint rule  ∫ = h f(c) + R,  |R| <= h^3/24 sup|f''|,
    f'' = 3/4 x^{-5/2} x'^2 - 1/2 x^{-3/2} x''  bounded with interval Horner on the cell;
  * cells containing an interior timing knot: zeroth-order hull bound h*[fmin, fmax];
  * prefix sums: recursive summation, error <= gamma_{n} * sum|terms| added explicitly.
Requires x > 0 on [0,1] (checked; otherwise status UNKNOWN_POSITIVITY). Zero-speed
endpoints are handled separately in time_law_enclosures (root substitution).
This module bounds the MATHEMATICAL inverse only; native inversion is audited
against it by scripts/native_time_inversion_audit.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .native_error_bound import exact_timing_table

INF = np.inf
SCOPE = "EXACT_MODEL_TIME_LAW_GRID_ENCLOSURE"


def up(x):
    return np.nextafter(x, INF)


def dn(x):
    return np.nextafter(x, -INF)


def iadd(a, b):
    return dn(a[0] + b[0]), up(a[1] + b[1])


def imul(a, b):
    p = np.stack([a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]])
    return dn(p.min(0)), up(p.max(0))


def frac_iv(v: Fraction) -> tuple[float, float]:
    f = float(v)
    return (f, f) if Fraction(f) == v else (float(dn(f)), float(up(f)))


def span_polys(timing_controls) -> list[list[tuple[float, float]]]:
    """Outward intervals of exact x coefficients on each timing span, ascending powers of tau=s-m/29."""
    t = exact_timing_table()  # [m][P = 3 - p][k]
    out = []
    for m in range(29):
        coeffs = []
        for p in range(4):
            v = sum((t[m][3 - p][k] * Fraction(float(timing_controls[m + k])) for k in range(4)), Fraction(0))
            coeffs.append(frac_iv(v))
        out.append(coeffs)
    return out


def _horner(coeffs, tau):
    """Interval Horner of ascending-power interval coefficients at interval tau (arrays)."""
    n = len(coeffs)
    acc = (np.full_like(tau[0], coeffs[-1][0]), np.full_like(tau[0], coeffs[-1][1]))
    for p in range(n - 2, -1, -1):
        acc = iadd(imul(acc, tau), (coeffs[p][0], coeffs[p][1]))
    return acc


def _deriv(coeffs):
    return [(dn(min(c[0] * k, c[1] * k)), up(max(c[0] * k, c[1] * k))) for k, c in enumerate(coeffs) if k > 0]


def _inv_sqrt(lo, hi):
    """Interval of 1/sqrt(x) for 0 < lo <= x <= hi."""
    return dn(1.0 / up(np.sqrt(up(hi)))), up(1.0 / dn(np.sqrt(dn(lo))))


@dataclass
class TimeGrid:
    n_cells: int
    T_lo: np.ndarray
    T_hi: np.ndarray
    fmin: np.ndarray
    fmax: np.ndarray
    status: str
    summation_error: float
    max_cell_remainder: float
    scope: str = SCOPE

    @property
    def duration(self):
        return float(self.T_lo[-1]), float(self.T_hi[-1])


def build_grid(timing_controls, log2_cells: int = 22) -> TimeGrid:
    N = 1 << log2_cells
    polys = span_polys(timing_controls)
    k = np.arange(N, dtype=np.float64)
    s0, s1 = k / N, (k + 1) / N            # exact dyadic doubles
    m0 = np.minimum((k * 29) // N, 28).astype(np.int64)
    m1 = np.minimum(((k + 1) * 29 - 1) // N, 28).astype(np.int64)   # span of points just below s1
    straddle = m1 != m0
    h = 1.0 / N
    lo_int = np.empty(N); hi_int = np.empty(N); fmin = np.empty(N); fmax = np.empty(N)
    worst_R = 0.0
    for m in range(29):
        sel = np.nonzero((m0 == m) & ~straddle)[0]
        if sel.size == 0:
            continue
        knot = frac_iv(Fraction(m, 29))
        c = polys[m]
        d1, = [_deriv(c)]
        d2 = _deriv(d1)
        tau_cell = (dn(s0[sel] - knot[1]), up(s1[sel] - knot[0]))
        sm = (s0[sel] + s1[sel]) * 0.5     # exact (dyadic)
        tau_mid = (dn(sm - knot[1]), up(sm - knot[0]))
        x_cell = _horner(c, tau_cell)
        if np.any(x_cell[0] <= 0):
            return TimeGrid(N, None, None, None, None, "UNKNOWN_POSITIVITY", 0.0, 0.0)
        x_mid = _horner(c, tau_mid)
        x1 = _horner(d1, tau_cell)
        x2 = _horner(d2, tau_cell)
        f_mid = _inv_sqrt(x_mid[0], x_mid[1])
        fl, fh = _inv_sqrt(x_cell[0], x_cell[1])
        xmin = x_cell[0]
        a1 = up(np.maximum(np.abs(x1[0]), np.abs(x1[1])))
        a2 = up(np.maximum(np.abs(x2[0]), np.abs(x2[1])))
        inv_s = up(1.0 / dn(np.sqrt(dn(xmin))))           # >= x^{-1/2}
        inv3 = up(inv_s * up(inv_s * inv_s))              # >= x^{-3/2}
        inv5 = up(inv3 * up(inv_s * inv_s))               # >= x^{-5/2}
        F2 = up(up(0.75 * up(inv5 * up(a1 * a1))) + up(0.5 * up(inv3 * a2)))
        R = up(F2 * (h * h * h / 24.0))                    # h^3/24 exact (dyadic h)
        worst_R = max(worst_R, float(R.max()))
        lo_int[sel] = dn(dn(h * f_mid[0]) - R)
        hi_int[sel] = up(up(h * f_mid[1]) + R)
        fmin[sel], fmax[sel] = fl, fh
    for idx in np.nonzero(straddle)[0]:
        xs_lo, xs_hi = [], []
        for m in (m0[idx], m1[idx]):
            knot = frac_iv(Fraction(int(m), 29))
            tau = (np.array([dn(s0[idx] - knot[1])]), np.array([up(s1[idx] - knot[0])]))
            xc = _horner(polys[m], tau)
            xs_lo.append(xc[0][0]); xs_hi.append(xc[1][0])
        xl, xh = min(xs_lo), max(xs_hi)
        if xl <= 0:
            return TimeGrid(N, None, None, None, None, "UNKNOWN_POSITIVITY", 0.0, 0.0)
        fl, fh = _inv_sqrt(np.array([xl]), np.array([xh]))
        lo_int[idx], hi_int[idx] = dn(h * fl[0]), up(h * fh[0])
        fmin[idx], fmax[idx] = fl[0], fh[0]
    if np.any(lo_int <= 0) or np.any(~np.isfinite(hi_int)):
        return TimeGrid(N, None, None, None, None, "UNKNOWN_NUMERICAL", 0.0, worst_R)
    # Recursive summation error bound: |fl(S_k) - S_k| <= gamma_{N} * sum_i |t_i| (u = 2^-53).
    u = 2.0 ** -53
    gamma = up(N * u / (1 - N * u))
    E = float(up(up(gamma * float(np.sum(hi_int)) * (1 + 1e-9)) + 1e-300))
    T_lo = np.concatenate([[0.0], dn(np.cumsum(lo_int) - E)])
    T_hi = np.concatenate([[0.0], up(np.cumsum(hi_int) + E)])
    T_lo[1:] = np.maximum(T_lo[1:], 0.0)
    return TimeGrid(N, T_lo, T_hi, fmin, fmax, "BOUNDED_EXACT_MODEL", E, worst_R)


def inverse_bracket(grid: TimeGrid, t: float) -> tuple[float, float]:
    """Rigorous [s_lo, s_hi] containing s*(t) for one t (CPU reference of the GPU audit).

    For t beyond the model duration the continuous trajectory holds its end state: s* = 1."""
    N = grid.n_cells
    if t <= 0:
        return 0.0, 0.0
    k = int(np.searchsorted(grid.T_hi, t, side="right")) - 1       # largest k with T_hi[k] <= t
    j = int(np.searchsorted(grid.T_lo, t, side="left")) - 1         # largest j with T_lo[j] < t
    if k >= N:
        lo = 1.0
    else:
        lo = min(float(dn(k / N + float(dn(dn(t - grid.T_hi[k]) / grid.fmax[k])))), (k + 1) / N)
        lo = max(lo, k / N)
    if j >= N:
        hi = 1.0
    else:
        hi = min(float(up(j / N + float(up(up(t - grid.T_lo[j]) / grid.fmin[j])))), (j + 1) / N, 1.0)
    return lo, hi

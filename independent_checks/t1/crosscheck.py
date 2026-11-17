"""Cross-checks of the code under review against the independent checker.

Only this module touches the reviewed code (curobo_certify.time_law_grid and, for a
model-identity comparison only, native_error_bound.exact_timing_table). Every decision is
made with exact rational comparisons against the independent rigorous enclosures of
taylor_time_law; outcomes are PROVED (containment shown), VIOLATED (non-containment shown)
or UNDECIDED (independent enclosure not tight enough to decide).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as Q

from . import exact_spline as ES
from .taylor_time_law import TimeLaw, inv_sqrt_bracket


def F(v) -> Q:
    """Exact rational of a binary64 value (Fraction(float) is exact)."""
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError("non-finite")
    return Q(v)


@dataclass
class Tally:
    proved: int = 0
    violated: int = 0
    undecided: int = 0
    examples: list = field(default_factory=list)

    def add(self, outcome: str, info=None):
        setattr(self, outcome, getattr(self, outcome) + 1)
        if outcome != "proved" and len(self.examples) < 20:
            self.examples.append((outcome, info))

    def as_dict(self):
        return {"proved": self.proved, "violated": self.violated, "undecided": self.undecided,
                "examples": [(o, repr(i)) for o, i in self.examples]}


def model_identity_vs_repo(timing_controls) -> int:
    """Number of exact coefficient mismatches between the independent de Boor model and the
    repo's exact_timing_table extraction (0 expected)."""
    from curobo_certify.native_error_bound import exact_timing_table
    mine = ES.model_from_controls(timing_controls)
    table = exact_timing_table()
    ctrl = [ES.f32_to_rational(v) for v in timing_controls]
    bad = 0
    for m in range(ES.SPANS):
        for p in range(4):
            theirs = sum((table[m][3 - p][k] * ctrl[m + k] for k in range(4)), Q(0))
            if theirs != mine[m][p]:
                bad += 1
    return bad


def check_grid_tables(grid, tl: TimeLaw, ks) -> dict:
    """Containment T_lo[k] <= T(k/N) <= T_hi[k] and fmin[k] <= x^{-1/2} <= fmax[k] on cell k."""
    import numpy as np
    N = grid.n_cells
    out = {"T_lo": Tally(), "T_hi": Tally(), "fmin": Tally(), "fmax": Tally()}
    struct = {"finite": bool(np.all(np.isfinite(grid.T_lo)) and np.all(np.isfinite(grid.T_hi))
                             and np.all(np.isfinite(grid.fmin)) and np.all(np.isfinite(grid.fmax))),
              "T_lo_nondecreasing": bool(np.all(np.diff(grid.T_lo) >= 0)),
              "T_hi_nondecreasing": bool(np.all(np.diff(grid.T_hi) >= 0)),
              "T_lo_le_T_hi": bool(np.all(grid.T_lo <= grid.T_hi)),
              "fmin_le_fmax": bool(np.all(grid.fmin <= grid.fmax)),
              "fmin_positive": bool(np.all(grid.fmin > 0)),
              "T_zero_at_0": bool(grid.T_lo[0] == 0 and grid.T_hi[0] == 0)}
    for k in ks:
        k = int(k)
        s = Q(k, N)
        lo, hi = tl.T_bounds(s)
        a, b = F(grid.T_lo[k]), F(grid.T_hi[k])
        out["T_lo"].add("proved" if a <= lo else ("violated" if a > hi else "undecided"), (k, float(a - lo)))
        out["T_hi"].add("proved" if b >= hi else ("violated" if b < lo else "undecided"), (k, float(hi - b)))
        if k < N:
            xl, xh = tl.x_range_on(s, Q(k + 1, N))
            if xl <= 0:
                out["fmax"].add("undecided", (k, "x lower bound <= 0"))
                out["fmin"].add("undecided", (k, "x lower bound <= 0"))
                continue
            # 1/sqrt(xh) >= flo_ ; 1/sqrt(xl) <= fhi_
            flo_, _ = inv_sqrt_bracket(xh.numerator, xh.denominator)
            _, fhi_ = inv_sqrt_bracket(xl.numerator, xl.denominator)
            fm, fM = F(grid.fmin[k]), F(grid.fmax[k])
            out["fmin"].add("proved" if fm <= flo_ else "undecided", (k, float(fm - flo_)))
            out["fmax"].add("proved" if fM >= fhi_ else "undecided", (k, float(fhi_ - fM)))
    res = {k: v.as_dict() for k, v in out.items()}
    res["structure"] = struct
    return res


def check_inverse_bracket(grid, tl: TimeLaw, ts, inverse_bracket) -> dict:
    """For each time t (float), decide  lo <= s*(t) <= hi  for (lo,hi)=inverse_bracket(grid,t)."""
    lower, upper, empty = Tally(), Tally(), 0
    for t in ts:
        tq = F(t)
        lo, hi = inverse_bracket(grid, float(t))
        lo_q, hi_q = F(lo), F(hi)
        if lo_q > hi_q:
            empty += 1
        # lower: lo <= s*
        if tq <= 0:
            lower.add("proved" if lo_q <= 0 else "violated", (t, lo))
            upper.add("proved" if hi_q >= 0 else "violated", (t, hi))
            continue
        if lo_q <= 0:
            lower.add("proved")
        else:
            a, b = tl.T_bounds(min(lo_q, tl.end))
            lower.add("proved" if b <= tq else ("violated" if a > tq else "undecided"), (t, lo))
        if hi_q >= tl.end:
            upper.add("proved")
        else:
            a, b = tl.T_bounds(hi_q)
            upper.add("proved" if a >= tq else ("violated" if b < tq else "undecided"), (t, hi))
    return {"lower": lower.as_dict(), "upper": upper.as_dict(), "empty_brackets": empty}


def certified_error_at(tl: TimeLaw, t, s_native) -> tuple[Q, str]:
    """Independent rigorous upper bound on |s_native - s*(t)| (t, s_native exact binary values)."""
    lo, hi, status = tl.inverse(F(t))
    s = F(s_native)
    return max(s - lo, hi - s, Q(0)), status

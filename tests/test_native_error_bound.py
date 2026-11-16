"""Tests for the rigorous native FP32 evaluation error bound.

These tests emulate the pinned native float32 expressions on the CPU in several
legal arithmetic variants (no FMA, FMA-contracted, worst-case 2-ulp division)
and compare against exact rational values. They test the bound implementation;
the GPU falsification script tests it against the real compiled kernels.
"""
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify import native_error_bound as NEB  # noqa: E402
from curobo_certify.boundary_reference import Boundaries  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402

F32 = np.float32
TABLE = np.load(ROOT / "fixtures/timeopt_timing_polynomial_c301cb.npy")


def ulp_shift(x, k):
    x = F32(x)
    for _ in range(abs(int(k))):
        x = np.nextafter(x, F32(np.inf) if k > 0 else F32(-np.inf))
    return x


class Arith:
    """float32 emulation; fma=True contracts a*b+c, div_ulps perturbs divisions."""

    def __init__(self, fma=False, div_ulps=0):
        self.fma, self.div_ulps = fma, div_ulps

    def mul(self, a, b):
        return F32(F32(a) * F32(b))

    def add(self, a, b):
        return F32(F32(a) + F32(b))

    def madd(self, a, b, c):  # a*b + c
        if self.fma:
            return F32(float(F32(a)) * float(F32(b)) + float(F32(c)))
        return self.add(self.mul(a, b), c)

    def div(self, a, b):
        return ulp_shift(F32(F32(a) / F32(b)), self.div_ulps)


def native_geometry_emul(ar, support, spans, t):
    c = [F32(0)] + [F32(F32(support[k]) - F32(support[0])) for k in range(1, 4)]
    t = F32(t)
    t2 = ar.mul(t, t)
    omt = F32(F32(1) - t)
    vb = [ar.div(ar.mul(ar.mul(F32(-3), omt), omt), 6),
          ar.div(ar.add(ar.mul(9, t2), -ar.mul(12, t)), 6),
          ar.div(ar.add(ar.madd(-9, t2, ar.mul(6, t)), 3), 6),
          ar.div(ar.mul(3, t2), 6)]
    ab = [ar.div(ar.mul(6, omt), 6), ar.div(ar.madd(18, t, F32(-12)), 6),
          ar.div(ar.madd(-18, t, F32(6)), 6), ar.div(ar.mul(6, t), 6)]
    jb = [F32(-1), F32(3), F32(-3), F32(1)]

    def dot(basis):
        r = F32(0)
        for v, b in zip(c, basis):
            r = ar.madd(v, b, r)
        return r
    kdt = ar.div(1, F32(spans))
    kdt2 = ar.mul(kdt, kdt)
    kdt3 = ar.mul(kdt2, kdt)
    return ar.div(dot(vb), kdt), ar.div(dot(ab), kdt2), ar.div(dot(jb), kdt3)


def exact_geometry(support, spans, t):
    s = [Fraction(float(F32(v))) for v in support]
    c = [x - s[0] for x in s]
    t = Fraction(float(F32(t)))
    vb = [-3 * (1 - t) ** 2 / 6, (9 * t * t - 12 * t) / 6, (-9 * t * t + 6 * t + 3) / 6, 3 * t * t / 6]
    ab = [(1 - t), (18 * t - 12) / 6, (-18 * t + 6) / 6, t]
    jb = [-1, 3, -3, 1]
    S = Fraction(spans)
    return (S * sum(a * b for a, b in zip(c, vb)), S ** 2 * sum(a * b for a, b in zip(c, ab)),
            S ** 3 * sum(a * b for a, b in zip(c, jb)))


def synthetic_path(rng, controls=9, dof=3):
    start, goal = rng.normal(size=dof), rng.normal(size=dof)
    inner = [start + (goal - start) * (k + 1) / (controls + 1) + 0.05 * rng.normal(size=dof)
             for k in range(controls - 1)]
    eff = np.array([start] * 3 + inner + [goal] * 3, dtype=np.float32)
    timing = (5 + 2 * rng.random(32)).astype(np.float32)
    return eff, timing


class NativeErrorBoundTests(unittest.TestCase):
    def test_geometry_bound_dominates_emulated_variants(self):
        rng = np.random.default_rng(20261116)
        m = NEB.ArithmeticModel()
        variants = [Arith(), Arith(fma=True), Arith(div_ulps=2), Arith(fma=True, div_ulps=-2)]
        for trial in range(150):
            spans = int(rng.integers(5, 40))
            support = (rng.normal(size=4) * 10 ** rng.uniform(-2, 1)).astype(np.float32)
            t = F32(rng.random())
            exact = exact_geometry(support, spans, t)
            et = NEB.native_geometry(m, support.tolist(), spans, Fraction(float(t)), Fraction(float(t)))
            for ar in variants:
                got = native_geometry_emul(ar, support, spans, t)
                for g, e, b in zip(got, exact, et):
                    self.assertLessEqual(abs(Fraction(float(g)) - e), Fraction(b.e),
                                         f"trial {trial} variant fma={ar.fma} div={ar.div_ulps}")

    def test_timing_bound_dominates_table_horner(self):
        rng = np.random.default_rng(7)
        m = NEB.ArithmeticModel()
        exact_table = NEB.exact_timing_table()
        for trial in range(120):
            controls = (rng.random(32) * 50).astype(np.float32)
            span = int(rng.integers(0, 29))
            local = F32(rng.uniform(-1e-7, 1 / 29 + 1e-7))
            for fma in (False, True):
                ar = Arith(fma=fma)
                a = []
                for P in range(4):
                    acc = F32(0)
                    for k in range(4):
                        acc = ar.madd(TABLE[span, P, span + k], controls[span + k], acc)
                    a.append(acc)
                x0 = ar.madd(ar.madd(ar.madd(a[0], local, a[1]), local, a[2]), local, a[3])
                x1 = ar.madd(ar.madd(ar.mul(3, a[0]), local, ar.mul(2, a[1])), local, a[2])
                x2 = ar.madd(ar.mul(6, a[0]), local, ar.mul(2, a[1]))
                ex = [sum((exact_table[span][P][k] * Fraction(float(controls[span + k])) for k in range(4)), Fraction(0))
                      for P in range(4)]
                L = Fraction(float(local))
                exact = (((ex[0] * L + ex[1]) * L + ex[2]) * L + ex[3], (3 * ex[0] * L + 2 * ex[1]) * L + ex[2],
                         6 * ex[0] * L + 2 * ex[1])
                et = NEB.native_timing(m, TABLE, exact_table, controls.tolist(), span, L, L)
                for g, e, b in zip((x0, x1, x2), exact, et):
                    self.assertLessEqual(abs(Fraction(float(g)) - e), Fraction(b.e), f"trial {trial} fma={fma}")

    def test_table_layout_and_exactness(self):
        # Finding (2026-11-16): the native float32 table is NOT a pure relative
        # rounding of the exact basis; SciPy's binary64 evaluation leaves 19 used
        # entries nonzero (|.| <= 4.3e-14) where the exact coefficient is zero.
        # The bound therefore charges the exact absolute difference per entry.
        exact = NEB.exact_timing_table()
        worst, spurious, spurious_abs = 0.0, 0, 0.0
        for span in range(29):
            for P in range(4):
                for k in range(4):
                    e = exact[span][P][k]
                    n = Fraction(float(TABLE[span, P, span + k]))
                    if e:
                        worst = max(worst, float(abs(n - e) / abs(e)))
                    elif n:
                        spurious += 1
                        spurious_abs = max(spurious_abs, float(abs(n)))
                # the kernel never reads columns outside span..span+3; they are zero anyway
                others = [TABLE[span, P, c] for c in range(32) if not span <= c <= span + 3]
                self.assertTrue(all(v == 0 for v in others))
        self.assertLess(worst, 2 ** -24)
        self.assertEqual(spurious, 19)
        self.assertLess(spurious_abs, 1e-13)
        # The bound charges these differences: a timing ET at an exact-zero entry has e>0.
        et = NEB.native_timing(NEB.ArithmeticModel(), TABLE, exact, [1.0] * 32, 0, Fraction(0), Fraction(0))
        self.assertTrue(all(x.e > 0 for x in et))

    def test_end_to_end_emulation_within_common_span_bounds(self):
        rng = np.random.default_rng(3)
        m = NEB.ArithmeticModel()
        eff, timing = synthetic_path(rng)
        spans = extract_timeopt_spans(eff, timing)
        bounds = NEB.common_span_bounds(m, spans, eff.tolist(), timing.tolist(), TABLE, pieces=4)
        S = len(eff) - 3
        exact_table = NEB.exact_timing_table()
        for fma in (False, True):
            ar = Arith(fma=fma)
            for s in np.concatenate([rng.random(120), [0.0, 1.0] + [k / S for k in range(S)]]).astype(np.float32):
                coord = ar.mul(s, F32(S))
                seg = min(int(coord), S - 1)
                t = F32(coord - F32(seg))
                q1, q2, q3 = native_geometry_emul(ar, eff[seg:seg + 4, 0], S, t)
                ms = min(28, max(0, int(ar.mul(s, F32(29)))))
                local = F32(s - ar.div(F32(ms), F32(29)))
                a = []
                for P in range(4):
                    acc = F32(0)
                    for k in range(4):
                        acc = ar.madd(TABLE[ms, P, ms + k], timing[ms + k], acc)
                    a.append(acc)
                x0 = ar.madd(ar.madd(ar.madd(a[0], local, a[1]), local, a[2]), local, a[3])
                x1 = ar.madd(ar.madd(ar.mul(3, a[0]), local, ar.mul(2, a[1])), local, a[2])
                x2 = ar.madd(ar.mul(6, a[0]), local, ar.mul(2, a[1]))
                speed = F32(np.sqrt(max(x0, F32(0))))
                jerk = ar.mul(speed, ar.madd(ar.mul(F32(0.5), q1), x2, ar.madd(ar.mul(F32(1.5), q2), x1, ar.mul(q3, x0))))
                sg = (Fraction(seg) + Fraction(float(t))) / S
                ci = next(i for i, sp in enumerate(spans) if sp.left <= sg <= sp.right)
                Q = exact_geometry(eff[seg:seg + 4, 0], S, t)
                mm = min(int(sg * 29), 28)
                ex = [sum((exact_table[mm][P][k] * Fraction(float(timing[mm + k])) for k in range(4)), Fraction(0))
                      for P in range(4)]
                L = sg - Fraction(mm, 29)
                X = (((ex[0] * L + ex[1]) * L + ex[2]) * L + ex[3], (3 * ex[0] * L + 2 * ex[1]) * L + ex[2],
                     6 * ex[0] * L + 2 * ex[1])
                J = float(X[0]) ** 0.5 * float(Q[2] * X[0] + Fraction(3, 2) * Q[1] * X[1] + Fraction(1, 2) * Q[0] * X[2])
                self.assertLessEqual(abs(float(jerk) - J), bounds[ci][0].jerk * (1 + 1e-9) + 1e-12,
                                     f"s={s} fma={fma}")

    def test_delta_guard(self):
        rng = np.random.default_rng(4)
        eff, timing = synthetic_path(rng)
        spans = extract_timeopt_spans(eff, timing)
        with self.assertRaises(ValueError):
            NEB.common_span_bounds(NEB.ArithmeticModel(), spans, eff.tolist(), timing.tolist(), TABLE,
                                   delta=Fraction(1, 10 ** 8))

    def test_tightening_changes_marginal_certificates(self):
        rng = np.random.default_rng(5)
        eff, timing = synthetic_path(rng)
        spans = extract_timeopt_spans(eff, timing)
        dof = eff.shape[1]
        targets = Boundaries.from_values(q_start=eff[0].astype(float).tolist(), q_end=eff[-1].astype(float).tolist(),
                                         x_start=spans[0].at("x", 0)[0], x_end=spans[-1].at("x", 1)[0], strict_rest=True)
        bounds = NEB.common_span_bounds(NEB.ArithmeticModel(), spans, eff.tolist(), timing.tolist(), TABLE, pieces=4)
        big = {"velocity": [1e3] * dof, "acceleration": [1e5] * dof, "jerk": [1e8] * dof}
        st, *_ = NEB.certify_native_reported(spans, targets, big, bounds)
        self.assertEqual(st, "CERTIFIED")
        # A limit equal to the error bound itself must fail closed.
        tiny = {"velocity": [1e3] * dof, "acceleration": [1e5] * dof,
                "jerk": [max(b.jerk for row in bounds for b in row if b.joint == j) for j in range(dof)]}
        st, *_ = NEB.certify_native_reported(spans, targets, tiny, bounds)
        self.assertNotEqual(st, "CERTIFIED")

    def test_division_requires_nonzero_divisor(self):
        m = NEB.ArithmeticModel()
        with self.assertRaises(ValueError):
            NEB.et_div(m, NEB.const(1), NEB.ET(-1.0, 1.0, 0.0))
        with self.assertRaises(ValueError):
            NEB.et_div(m, NEB.const(1), NEB.ET(1.0, 2.0, 1.5))


if __name__ == "__main__":
    unittest.main()

"""R0 independent T1 review: tests of the independent checker and cross-checks of the reviewed code.

Run:  PYTHONPATH=src:. python -m unittest tests.test_independent_t1_checker -v
(also picked up by `python -m unittest discover -s tests -p 'test_*.py'` when PYTHONPATH
contains src and the repository root).

The independent checker (independent_checks/t1) uses exact integers/Fractions, integer isqrt
and a Taylor model of (1+u)^{-1/2} with Lagrange remainder; it shares no formulas with
time_law_grid / time_law_enclosures / native_error_bound. Closed forms are checked with exact
rational identities wherever possible; mpmath.iv (outward-rounded interval arithmetic) is used
only where no rational closed form exists.

Grid resolution note: the reviewed grid is cross-checked in FULL (every index) at 2^10, 2^12
and 2^16 here; 2^22 is cross-checked on a sample of ~750 indices per path (all knot-adjacent
and end cells + random) because a full exact check of 4.2M cells takes ~15 min per path.
The 224-path run is independent_checks/t1/run_review.py.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from independent_checks.t1 import exact_spline as ES  # noqa: E402
from independent_checks.t1 import crosscheck as C  # noqa: E402
from independent_checks.t1.taylor_time_law import TimeLaw, inv_sqrt_bracket  # noqa: E402

AUDIT_JSON = ROOT / "runs/margins/time_inversion_audit_224paths.json"
REAL_NPZ = ROOT / "runs/margins/native/margin-20261117-0000-margin_0p900.npz"
FIXTURE = ROOT / "fixtures/timeopt_timing_polynomial_c301cb.npy"


def f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def single_piece(coeffs, left=Q(0), right=Q(1)):
    """TimeLaw from one exact polynomial piece in w = s - left."""
    return TimeLaw.from_pieces([(left, right, tuple(Q(c) for c in coeffs))])


def contains(tl_bounds, exact):
    lo, hi = tl_bounds
    return lo <= exact <= hi


class TestFloat32Decoding(unittest.TestCase):
    def test_random_bits_match_fraction(self):
        rng = np.random.default_rng(0)
        for b in list(rng.integers(0, 0x7F800000, 2000)) + [0, 1, 0x7FFFFF, 0x800000, 0x7F7FFFFF, 0x80000000, 0x80000001]:
            b = int(b)
            self.assertEqual(ES.f32_bits_to_rational(b), Q(float(np.uint32(b).view(np.float32))))

    def test_nonfinite_rejected(self):
        for b in (0x7F800000, 0x7FC00000, 0xFF800000):
            with self.assertRaises(ValueError):
                ES.f32_bits_to_rational(b)


class TestIndependentSpline(unittest.TestCase):
    def test_partition_of_unity(self):
        model = ES.model_from_rationals([Q(1)] * 32)
        for m in range(29):
            self.assertEqual(model[m], (Q(1), Q(0), Q(0), Q(0)))

    def test_c2_and_endpoints_random(self):
        rng = np.random.default_rng(1)
        for _ in range(5):
            ctrl = [Q(int(v), 1024) for v in rng.integers(1, 100000, 32)]
            model = ES.model_from_rationals(ctrl)
            self.assertEqual(ES.continuity_defects(model), [])
            self.assertEqual(model[0][0], ctrl[0])                      # clamped: x(0) = c0
            self.assertEqual(ES.eval_cubic(model[28], Q(1, 29)), ctrl[31])  # x(1) = c31

    def test_matches_upstream_float32_table(self):
        """Independent derivation vs the upstream CubicTimingBasis export (float32, approx)."""
        if not FIXTURE.exists():
            self.skipTest("fixture missing")
        table = np.load(FIXTURE).astype(np.float64)  # [span][power desc][control], coeff of local^(3-P)
        for k in range(32):
            ctrl = [Q(0)] * 32
            ctrl[k] = Q(1)
            model = ES.model_from_rationals(ctrl)
            for m in range(29):
                for P in range(4):
                    exact = float(model[m][3 - P])
                    self.assertLessEqual(abs(table[m][P][k] - exact), 2e-6 * max(1.0, abs(exact)),
                                         (k, m, P, table[m][P][k], exact))

    @unittest.skipUnless(REAL_NPZ.exists(), "real NPZ missing")
    def test_identity_with_repo_exact_model(self):
        tc = np.load(REAL_NPZ)["timing_controls"]
        self.assertEqual(C.model_identity_vs_repo(tc), 0)
        rng = np.random.default_rng(2)
        self.assertEqual(C.model_identity_vs_repo(rng.uniform(1e-3, 1e2, 32).astype(np.float32)), 0)


class TestSqrtEnclosure(unittest.TestCase):
    def test_bracket(self):
        for num, den in ((1, 1), (4, 1), (2, 1), (1, 3), (10 ** 40 + 7, 3), (5, 10 ** 30)):
            lo, hi = inv_sqrt_bracket(num, den)
            r = Q(den, num)
            self.assertLessEqual(lo * lo, r)
            self.assertGreaterEqual(hi * hi, r)
            self.assertLessEqual(hi - lo, Q(1, 2 ** 199))


class TestClosedForms(unittest.TestCase):
    def test_constant(self):
        tl = single_piece([Q(4)])
        for s in (Q(0), Q(1, 3), Q(1, 2), Q(1)):
            self.assertTrue(contains(tl.T_bounds(s), s / 2))
        lo, hi, st = tl.inverse(Q(1, 8))
        self.assertTrue(lo <= Q(1, 4) <= hi)

    def test_affine_rational_points(self):
        # x = 1 + 3s: T(s) = 2(sqrt(1+3s) - 1)/3 ; rational at 1+3s = (p/q)^2
        tl = single_piece([Q(1), Q(3)])
        for s, root in ((Q(1), Q(2)), (Q(5, 12), Q(3, 2)), (Q(7, 27), Q(4, 3))):
            self.assertEqual(1 + 3 * s, root * root)
            self.assertTrue(contains(tl.T_bounds(s), 2 * (root - 1) / 3))
            lo, hi, _ = tl.inverse(2 * (root - 1) / 3)
            self.assertTrue(lo <= s <= hi)

    def test_positive_cubic_rational_points(self):
        # x = (1+s)^3 on [0,3]: T(s) = 2(1 - (1+s)^{-1/2})
        tl = single_piece([Q(1), Q(3), Q(3), Q(1)], Q(0), Q(3))
        for s, root in ((Q(3), Q(2)), (Q(5, 4), Q(3, 2)), (Q(9, 16), Q(5, 4))):
            self.assertEqual(1 + s, root * root)
            exact = 2 * (1 - 1 / root)
            self.assertTrue(contains(tl.T_bounds(s), exact))
            lo, hi, _ = tl.inverse(exact)
            self.assertTrue(lo <= s <= hi)

    def test_quadratic_log_interval(self):
        import mpmath
        iv = mpmath.iv
        iv.prec = 200
        tl = single_piece([Q(1), Q(2), Q(1)])  # (1+s)^2 -> T = ln(1+s)
        for s in (Q(1, 7), Q(1, 2), Q(1)):
            lo, hi = tl.T_bounds(s)
            ref = iv.log(iv.mpf(1) + iv.mpf(s.numerator) / s.denominator)
            self.assertTrue(iv.mpf(lo.numerator) / lo.denominator <= ref.b)
            self.assertTrue(iv.mpf(hi.numerator) / hi.denominator >= ref.a)
            self.assertLess(float(hi - lo), 1e-20)

    def test_near_zero_positive_minimum(self):
        import mpmath
        iv = mpmath.iv
        iv.prec = 200
        eps = Q(1, 10 ** 8)
        # x = eps + (s - 1/2)^2 ; T(s) = asinh((s-1/2)/sqrt(eps)) - asinh(-1/(2 sqrt(eps)))
        tl = single_piece([eps + Q(1, 4), Q(-1), Q(1)])
        self.assertEqual(tl.status, "BOUNDED_EXACT_MODEL")
        se = iv.sqrt(iv.mpf(1) / 10 ** 8)

        def asinh(z):
            return iv.log(z + iv.sqrt(z * z + 1))
        for s in (Q(1, 2), Q(3, 4), Q(1)):
            u = iv.mpf(s.numerator) / s.denominator - iv.mpf(1) / 2
            ref = asinh(u / se) - asinh(-iv.mpf(1) / 2 / se)
            lo, hi = tl.T_bounds(s)
            self.assertTrue(iv.mpf(lo.numerator) / lo.denominator <= ref.b, s)
            self.assertTrue(iv.mpf(hi.numerator) / hi.denominator >= ref.a, s)

    def test_internal_root_fails_closed(self):
        tl = single_piece([Q(1, 4), Q(-1), Q(1)])  # (s-1/2)^2 : double root, nonintegrable
        self.assertEqual(tl.status, "UNKNOWN_POSITIVITY")
        tl = single_piece([Q(-1), Q(3)])          # negative near 0
        self.assertEqual(tl.status, "UNKNOWN_POSITIVITY")


class TestReviewedGridAgainstIndependent(unittest.TestCase):
    """Exact containment of the reviewed time_law_grid tables in the independent enclosure."""

    def _full(self, tc, L):
        from curobo_certify import time_law_grid as TG
        grid = TG.build_grid(tc, L)
        self.assertEqual(grid.status, "BOUNDED_EXACT_MODEL", L)
        tl = TimeLaw.from_timing_controls(tc)
        res = C.check_grid_tables(grid, tl, range(grid.n_cells + 1))
        for fam in ("T_lo", "T_hi", "fmin", "fmax"):
            self.assertEqual(res[fam]["violated"], 0, (L, fam, res[fam]["examples"]))
            self.assertEqual(res[fam]["undecided"], 0, (L, fam, res[fam]["examples"]))
        self.assertTrue(all(res["structure"].values()), res["structure"])
        return grid, tl

    @unittest.skipUnless(REAL_NPZ.exists(), "real NPZ missing")
    def test_real_path_full_2p10_2p12_2p16(self):
        from curobo_certify import time_law_grid as TG
        tc = np.load(REAL_NPZ)["timing_controls"]
        self.assertEqual(TG.build_grid(tc, 8).status, "UNKNOWN_POSITIVITY")  # fails closed at 2^8
        for L in (10, 12, 16):
            self._full(tc, L)

    def test_benign_synthetic_full_2p8(self):
        tc = (2.0 + np.sin(np.arange(32))).astype(np.float32)
        self._full(tc, 8)

    @unittest.skipUnless(REAL_NPZ.exists(), "real NPZ missing")
    def test_real_path_sampled_2p22(self):
        from curobo_certify import time_law_grid as TG
        tc = np.load(REAL_NPZ)["timing_controls"]
        grid = TG.build_grid(tc, 22)
        tl = TimeLaw.from_timing_controls(tc)
        N = grid.n_cells
        ks = set(range(64)) | set(range(N - 63, N + 1))
        for m in range(1, 29):
            ks |= {(m * N) // 29 + d for d in (-1, 0, 1, 2)}
        ks |= set(int(v) for v in np.random.default_rng(3).integers(0, N + 1, 200))
        res = C.check_grid_tables(grid, tl, sorted(ks))
        for fam in ("T_lo", "T_hi", "fmin", "fmax"):
            self.assertEqual(res[fam]["violated"] + res[fam]["undecided"], 0, (fam, res[fam]["examples"]))
        self.assertTrue(all(res["structure"].values()))

    def test_randomized_float32_families(self):
        from curobo_certify import time_law_grid as TG
        rng = np.random.default_rng(20261117)
        for trial in range(6):
            tc = np.exp(rng.uniform(np.log(1e-3), np.log(1e2), 32)).astype(np.float32)
            grid = TG.build_grid(tc, 12)
            if grid.status != "BOUNDED_EXACT_MODEL":
                continue  # fail closed is acceptable
            tl = TimeLaw.from_timing_controls(tc)
            res = C.check_grid_tables(grid, tl, range(grid.n_cells + 1))
            for fam in ("T_lo", "T_hi", "fmin", "fmax"):
                self.assertEqual(res[fam]["violated"], 0, (trial, fam))
                self.assertEqual(res[fam]["undecided"], 0, (trial, fam))
            T1 = float(grid.T_hi[-1])
            ts = [float(np.float32(v)) for v in rng.random(40) * T1 * 1.01]
            inv = C.check_inverse_bracket(grid, tl, ts, TG.inverse_bracket)
            self.assertEqual(inv["lower"]["violated"] + inv["upper"]["violated"], 0)
            self.assertEqual(inv["lower"]["undecided"] + inv["upper"]["undecided"], 0)

    def test_near_zero_internal_knot_controls(self):
        """x ~ 1e-6 at an interior knot (three tiny controls): both must bound or fail closed."""
        from curobo_certify import time_law_grid as TG
        tc = np.full(32, 2.0, np.float32)
        tc[14:17] = np.float32(1e-6)
        tl = TimeLaw.from_timing_controls(tc)
        self.assertEqual(tl.status, "BOUNDED_EXACT_MODEL")
        grid = TG.build_grid(tc, 16)
        if grid.status == "BOUNDED_EXACT_MODEL":
            res = C.check_grid_tables(grid, tl, range(grid.n_cells + 1))
            for fam in ("T_lo", "T_hi"):
                self.assertEqual(res[fam]["violated"] + res[fam]["undecided"], 0, fam)

    def test_zero_at_internal_knot_fails_closed(self):
        from curobo_certify import time_law_grid as TG
        tc = np.full(32, 2.0, np.float32)
        tc[14:17] = 0.0  # x(15/29)=0 with x'=0: nonintegrable
        self.assertEqual(TimeLaw.from_timing_controls(tc).status, "UNKNOWN_POSITIVITY")
        for L in (8, 12, 16):
            self.assertEqual(TG.build_grid(tc, L).status, "UNKNOWN_POSITIVITY")


class TestReviewedInverseBracket(unittest.TestCase):
    @unittest.skipUnless(REAL_NPZ.exists(), "real NPZ missing")
    def test_edges_real_path(self):
        from curobo_certify import time_law_grid as TG
        z = np.load(REAL_NPZ)
        tc, Tn = z["timing_controls"], float(z["native_timing_time_grid"][-1])
        grid = TG.build_grid(tc, 16)
        tl = TimeLaw.from_timing_controls(tc)
        bT = struct.unpack("<I", struct.pack("<f", Tn))[0]
        t_hi1 = float(tl.total_bounds()[1])
        bH = struct.unpack("<I", struct.pack("<f", t_hi1))[0]
        ts = [0.0, f32(1), f32(0x7FFFFF), f32(0x800000), 1e-30, Tn, f32(bT - 1), f32(bT + 1),
              f32(bH - 1), f32(bH), f32(bH + 1), 2 * Tn, 3.0e38]
        ts = [float(np.float32(t)) for t in ts]
        res = C.check_inverse_bracket(grid, tl, ts, TG.inverse_bracket)
        self.assertEqual(res["lower"]["violated"] + res["upper"]["violated"], 0, res)
        self.assertEqual(res["lower"]["undecided"] + res["upper"]["undecided"], 0, res)
        self.assertEqual(res["empty_brackets"], 0)
        # saturation: after the model duration s* = 1 and the bracket must be [1, 1]
        self.assertEqual(TG.inverse_bracket(grid, 2 * Tn), (1.0, 1.0))
        self.assertEqual(TG.inverse_bracket(grid, 0.0), (0.0, 0.0))
        lo, hi = TG.inverse_bracket(grid, f32(1))
        self.assertTrue(0.0 <= lo <= hi <= 1.0)


class TestStrictRestEndpointModule(unittest.TestCase):
    """time_law_enclosures.endpoint_root_integral (strict-rest x(0)=0; unused by real outputs)."""

    def test_simple_root_closed_forms(self):
        import mpmath
        from curobo_certify import time_law_enclosures as TE
        lo, hi, st = TE.endpoint_root_integral((Q(0), Q(4)), Q(1), Q(1, 4))  # int_0^{1/4} du/sqrt(4u) = 1/2
        self.assertEqual(st, "BOUNDED_EXACT_MODEL")
        self.assertTrue(lo <= Q(1, 2) <= hi)
        iv = mpmath.iv
        iv.prec = 200
        lo, hi, st = TE.endpoint_root_integral((Q(0), Q(1), Q(1)), Q(3), Q(1, 2))  # 3 * 2 asinh(sqrt(1/2))
        r = iv.sqrt(iv.mpf(1) / 2)
        ref = 6 * iv.log(r + iv.sqrt(r * r + 1))
        self.assertEqual(st, "BOUNDED_EXACT_MODEL")
        self.assertTrue(iv.mpf(lo.numerator) / lo.denominator <= ref.b)
        self.assertTrue(iv.mpf(hi.numerator) / hi.denominator >= ref.a)

    def test_classification(self):
        from curobo_certify import time_law_enclosures as TE
        self.assertEqual(TE.endpoint_root_integral((Q(0), Q(0), Q(1)), Q(1), Q(1, 2))[2], "NONINTEGRABLE_ENDPOINT")
        self.assertEqual(TE.endpoint_root_integral((Q(0), Q(-1)), Q(1), Q(1, 2))[2], "NEGATIVE_X_AT_ENDPOINT")
        self.assertEqual(TE.endpoint_root_integral((Q(0), Q(0)), Q(1), Q(1, 2))[2], "IDENTICALLY_ZERO_SPEED")


class TestFindings(unittest.TestCase):
    """Executable statements of the R0 findings (they PASS when the finding reproduces)."""

    def test_F1_legacy_epsilon_is_near_end_distance(self):
        # bracket [lo,hi] = [0.3, 0.4] proves s* in it; native s = 0.5
        lo, hi, s = 0.3, 0.4, 0.5
        legacy = max(s - hi, lo - s, 0.0)         # what time_inverse_audit.cuh reduces
        far = max(s - lo, hi - s)                 # sound upper bound of |s - s*|
        self.assertLess(legacy, far)
        self.assertAlmostEqual(far - legacy, hi - lo)
        # s inside the bracket: legacy says 0 although |s - s*| can be hi - lo
        self.assertEqual(max(0.35 - hi, lo - 0.35, 0.0), 0.0)

    @unittest.skipUnless(REAL_NPZ.exists() and AUDIT_JSON.exists(), "real data missing")
    def test_F1_reported_eps_violated_on_real_path(self):
        """Proven: true |s_native - s*| > audited eps at the audited argmax time (path 0p900).

        s_native at t is the native float32 progress returned by the pinned progress_at_time on
        the RTX 3090 (gpu_probe.py P7; bitwise equal to production on the audited chunk)."""
        row = [r for r in json.loads(AUDIT_JSON.read_text())["rows"]
               if r["path"] == "margin-20261117-0000-margin_0p900"][0]
        t, s_native = row["at_time"], 0.6203527450561523
        self.assertEqual(t, 0.17535433173179626)
        tl = TimeLaw.from_timing_controls(np.load(REAL_NPZ)["timing_controls"])
        lo, hi, st = tl.inverse(Q(t))
        self.assertEqual(st, "BRACKETED")
        true_error_lower = Q(s_native) - hi          # s* <= hi  =>  |s - s*| >= s - hi
        self.assertGreater(true_error_lower, Q(row["eps_s_certified_max"]))

    def test_F_small_grid_multi_knot_cells_ignored(self):
        """build_grid only considers the first and last timing span of a cell; for N < 29 a cell
        can contain two knots and the middle span polynomial is never evaluated."""
        N = 16
        k = np.arange(N)
        m0 = np.minimum((k * 29) // N, 28)
        m1 = np.minimum(((k + 1) * 29 - 1) // N, 28)
        self.assertTrue(np.any(m1 - m0 >= 2))      # cells with >= 3 spans exist at 2^4
        N = 1 << 5
        k = np.arange(N)
        m0 = np.minimum((k * 29) // N, 28)
        m1 = np.minimum(((k + 1) * 29 - 1) // N, 28)
        self.assertFalse(np.any(m1 - m0 >= 2))     # not reachable for log2_cells >= 5


if __name__ == "__main__":
    unittest.main()

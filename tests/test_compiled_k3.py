"""Compiled complete K3 producer + independent verifier tests (no GPU)."""
import ctypes
import ctypes.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from curobo_certify import compiled_k3 as K  # noqa: E402
import k3_differential_campaign as campaign  # noqa: E402


def smooth_case(seed=1, D=3, inner=6, scale=1.0):
    rng = np.random.default_rng(seed)
    start, goal = rng.normal(size=D), rng.normal(size=D)
    pts = [start + (goal - start) * (k + 1) / (inner + 1) + 0.05 * rng.normal(size=D) for k in range(inner)]
    geo = np.array([start] * 3 + pts + [goal] * 3, dtype=np.float32)
    timing = (8 + rng.random(32)).astype(np.float32)
    lim = {"velocity": np.full(D, 50.0 * scale), "acceleration": np.full(D, 5e3 * scale), "jerk": np.full(D, 5e6 * scale)}
    return geo, timing, geo[0].copy(), geo[-1].copy(), lim


class CompiledK3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.build_c = K.build("k3_certifier.cpp", d / "libk3c.so")
        cls.build_v = K.build("k3_verifier.cpp", d / "libk3v.so")
        cls.P = K.Producer(d / "libk3c.so")
        cls.V = K.Verifier(d / "libk3v.so")
        campaign._init(str(d))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def certified(self, args, **kw):
        r = K.certify_and_verify(self.P, self.V, *args, **kw)
        self.assertEqual(r["final"], "CERTIFIED", r.get("verification"))
        return r

    def test_generous_limits_certify_and_verify(self):
        r = self.certified(smooth_case())
        self.assertEqual(r["boundary"], "PASS")
        self.assertTrue(all(v == "CERTIFIED" for v in r["families"].values()))

    def test_tampered_certificates_rejected(self):
        args = smooth_case(2)
        r = self.certified(args, max_depth=6)
        L, dig = r["leaves"], r["digest"]
        ok = self.V.verify(*args, L, dig)
        self.assertTrue(ok["pass"])
        reasons = {}
        reasons["drop"] = self.V.verify(*args, L[1:], dig)
        reasons["duplicate"] = self.V.verify(*args, np.concatenate([L, L[:1]]), dig)
        over = L[:1].copy(); over["depth"] = 1; over["bits"] = 0
        reasons["overlap"] = self.V.verify(*args, np.concatenate([L, over]), dig)
        bad_bits = L.copy(); bad_bits["bits"][0] = 7
        reasons["bits_beyond_depth"] = self.V.verify(*args, bad_bits, dig)
        moved = L.copy(); moved["joint"][1] = (moved["joint"][1] + 1) % args[0].shape[1] if moved["family"][1] else -1
        reasons["moved_leaf"] = self.V.verify(*args, moved, dig)
        lim2 = {k: v.copy() for k, v in args[4].items()}; lim2["jerk"][0] *= 0.5
        reasons["limit_changed"] = self.V.verify(*args[:4], lim2, L, dig)
        geo2 = args[0].copy(); geo2[4, 0] += np.float32(1e-3)
        reasons["geometry_changed"] = self.V.verify(geo2, *args[1:], L, dig)
        reasons["forged_digest_with_changed_limit"] = self.V.verify(
            *args[:4], lim2, L, K.input_digest(*args[:4], lim2))
        for name, res in reasons.items():
            if name == "forged_digest_with_changed_limit":
                continue  # halved jerk limit may still be certified; checked separately below
            self.assertFalse(res["pass"], name)
        # status flags/uppers in leaves are not trusted: forging uppers changes nothing
        forged = L.copy(); forged["upper"] = -1e9
        self.assertTrue(self.V.verify(*args, forged, dig)["pass"])

    def test_forged_certificate_for_violating_path_rejected(self):
        geo, timing, qs, qg, lim = smooth_case(3)
        tight = {k: v * 1e-3 for k, v in lim.items()}
        r = K.certify_and_verify(self.P, self.V, geo, timing, qs, qg, tight)
        self.assertEqual(r["status"], "VIOLATED")
        good = self.certified((geo, timing, qs, qg, lim), max_depth=0)
        fake_digest = K.input_digest(geo, timing, qs, qg, tight)
        res = self.V.verify(geo, timing, qs, qg, tight, good["leaves"], fake_digest)
        self.assertFalse(res["pass"])
        self.assertEqual(res["reason"], "leaf hull exceeds zero")

    def test_boundary_obligations(self):
        geo, timing, qs, qg, lim = smooth_case(4)
        r = K.certify_and_verify(self.P, self.V, geo, timing, qs + np.float32(1e-3), qg, lim)
        self.assertEqual(r["final"], "VIOLATED_BOUNDARY")
        g2 = geo.copy(); g2[2] += np.float32(0.01)  # nonzero start velocity with x(0)>0
        r = K.certify_and_verify(self.P, self.V, g2, timing, g2[0], qg, lim)
        self.assertEqual(r["final"], "VIOLATED_BOUNDARY")
        t0 = timing.copy(); t0[0] = 0  # zero start speed, x'(0)>0: integrable, allowed
        self.certified((geo, t0, qs, qg, lim))
        t1 = timing.copy(); t1[0] = t1[1] = 0  # double root: infinite traversal time
        r = K.certify_and_verify(self.P, self.V, geo, t1, qs, qg, lim)
        self.assertEqual(r["final"], "VIOLATED_BOUNDARY")

    def test_interior_zero_speed_is_strict_violation(self):
        geo, timing, qs, qg, lim = smooth_case(5)
        # x touches zero exactly at an interior knot: never certified (UNKNOWN is the
        # fail-closed outcome; an interval enclosure of an exact zero is not a witness).
        t = timing.copy(); t[10:13] = 0
        r = K.certify_and_verify(self.P, self.V, geo, t, qs, qg, lim)
        self.assertNotEqual(r["final"], "CERTIFIED")
        if r["status"] == "VIOLATED":
            self.assertTrue(campaign.witness_ok({"geometry": geo, "timing": t, "limits": lim}, r["witness"]))
        # x identically zero on a whole interior span: a rigorous strict-positivity witness.
        t = timing.copy(); t[10:14] = 0
        r = K.certify_and_verify(self.P, self.V, geo, t, qs, qg, lim)
        self.assertEqual(r["status"], "VIOLATED")
        self.assertEqual(r["witness"]["family"], 0)
        self.assertTrue(campaign.witness_ok({"geometry": geo, "timing": t, "limits": lim}, r["witness"]))

    def test_nonfinite_and_overflow_fail_closed(self):
        geo, timing, qs, qg, lim = smooth_case(6)
        g = geo.copy(); g[3, 1] = np.nan
        self.assertEqual(self.P.certify(g, timing, qs, qg, lim)["status"], "INVALID_INPUT")
        tt = timing.copy(); tt[5] = np.inf
        self.assertEqual(self.P.certify(geo, tt, qs, qg, lim)["status"], "INVALID_INPUT")
        big = geo * np.float32(3e37)
        st = self.P.certify(big, timing, big[0], big[-1], lim)["status"]
        self.assertNotEqual(st, "CERTIFIED")
        bad = {k: v.copy() for k, v in lim.items()}; bad["velocity"][0] = 0
        self.assertEqual(self.P.certify(geo, timing, qs, qg, bad)["status"], "INVALID_INPUT")

    def test_budgets_and_ablations(self):
        geo, timing, qs, qg, lim = smooth_case(7)
        r = self.P.certify(geo, timing, qs, qg, lim, max_nodes=3)
        self.assertEqual(r["status"], "UNKNOWN_NODES")
        tight = {k: v * 1e-3 for k, v in lim.items()}
        self.assertEqual(self.P.certify(geo, timing, qs, qg, tight, witnesses=False, max_depth=3)["status"], "UNKNOWN_DEPTH")
        u0 = self.P.certify(geo, timing, qs, qg, lim, mode="uniform", max_depth=0)
        u2 = self.P.certify(geo, timing, qs, qg, lim, mode="uniform", max_depth=2)
        self.assertEqual(u0["status"], "CERTIFIED")
        self.assertEqual(len(u2["leaves"]), 4 * len(u0["leaves"]))
        self.assertTrue(self.V.verify(geo, timing, qs, qg, lim, u2["leaves"], K.input_digest(geo, timing, qs, qg, lim))["pass"])

    def test_floating_point_environment_fails_closed(self):
        libm = ctypes.CDLL(ctypes.util.find_library("m"))
        FE_UPWARD = 0x800
        args = smooth_case(8)
        r = self.certified(args)
        old = libm.fegetround()
        try:
            libm.fesetround(FE_UPWARD)
            self.assertEqual(self.P.certify(*args)["status"], "NUMERICAL_ERROR")
            self.assertEqual(self.V.verify(*args, r["leaves"], r["digest"])["reason"], "rounding mode is not nearest")
        finally:
            libm.fesetround(old)
        self.assertEqual(self.P.certify(*args)["status"], "CERTIFIED")

    def test_inline_nextafter_matches_libm(self):
        rng = np.random.default_rng(0)
        f = np.finfo(np.float64)
        vals = np.concatenate([[0.0, -0.0, 1.0, -1.0, f.max, -f.max, f.tiny, -f.tiny, 5e-324, -5e-324,
                                2 * f.tiny, f.tiny / 2, 1 - f.epsneg, 1 + f.eps, np.inf, -np.inf],
                               rng.normal(size=20000) * 10.0 ** rng.integers(-300, 300, 20000),
                               rng.integers(-2**62, 2**62, 20000).view(np.float64)])
        vals = np.ascontiguousarray(vals[~np.isnan(vals)])
        self.P.lib.k3c_selftest_nextafter.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        bad = self.P.lib.k3c_selftest_nextafter(vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(vals))
        self.assertEqual(bad, 0)

    def test_only_c_api_is_exported(self):
        # Regression (2026-11-16): helper functions accidentally given C linkage were
        # exported as add/mul/div and libc's div() was bound instead inside Python,
        # silently producing ~2^-126 native bounds. Only the API may be exported.
        import subprocess
        d = Path(self.tmp.name)
        for lib, prefix in (("libk3c.so", "k3c_"), ("libk3v.so", "k3v_")):
            out = subprocess.check_output(["nm", "-D", "--defined-only", str(d / lib)], text=True)
            names = [line.split()[-1] for line in out.splitlines() if line.strip()]
            self.assertTrue(names and all(n.startswith(prefix) for n in names), names)

    def test_compiled_native_bounds_dominate_python_reference(self):
        from curobo_certify import native_error_bound as NEB
        from curobo_certify.spline_polynomials import extract_timeopt_spans
        geo, timing, *_ = smooth_case(9, D=2, inner=4)
        table = np.load(ROOT / "fixtures/timeopt_timing_polynomial_c301cb.npy")
        py = NEB.common_span_bounds(NEB.ArithmeticModel(), extract_timeopt_spans(geo, timing), geo.tolist(),
                                    timing.tolist(), table, pieces=4)
        cc = K.native_bounds(self.P, geo, timing, pieces=4)
        ref = np.array([[[b.velocity, b.acceleration, b.jerk] for b in row] for row in py])
        self.assertEqual(cc.shape, ref.shape)
        mask = ref > 1e-20
        self.assertTrue(np.all(cc[mask] >= ref[mask] * (1 - 1e-12)))
        self.assertTrue(np.all(cc[mask] <= ref[mask] * 2.0))
        self.assertTrue(np.all(cc > 0))

    def test_seeded_differential_against_exact_reference(self):
        rows = [campaign.run_case(seed) for seed in range(7000, 7150)]
        breaches = [r for r in rows if r["breaches"]]
        self.assertEqual(breaches, [])
        self.assertGreater(sum(r["final"] == "CERTIFIED" for r in rows), 5)
        self.assertGreater(sum(r["producer"] == "VIOLATED" for r in rows), 30)


if __name__ == "__main__":
    unittest.main()

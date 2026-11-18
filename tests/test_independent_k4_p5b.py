"""P5b independent review: compiled K4 pipeline (C++ producer k4_certify_path, C++ verifier k4v_verify).

Run:  PYTHONPATH=src:. python -m unittest tests.test_independent_k4_p5b -v
Truth: independent_checks/k4p2 Lagrangian oracle on the EXACT spans of spline_polynomials (K3-reviewed),
plus exact rational arithmetic for the Taylor remainders. Libraries are built in a temp dir; skipped
without g++/mpmath. Dense sampling is falsification evidence only, not a proof.
"""
from __future__ import annotations

import copy
import json
import random
import shutil
import sys
import tempfile
import types
import unittest
from fractions import Fraction as Q
from math import factorial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import mpmath  # noqa: F401
    HAVE_MPMATH = True
except ImportError:  # pragma: no cover
    HAVE_MPMATH = False
HAVE_GXX = shutil.which("g++") is not None

from curobo_certify.k4_panda_model import PandaModel, load_contract  # noqa: E402

if HAVE_MPMATH:
    from independent_checks.k4p2 import campaign as C  # noqa: E402
    from independent_checks.k4p2 import oracle as O  # noqa: E402


class _CP:
    """P5b helpers (inline): drive the compiled pipeline and check its claims with the oracle on the exact
    spans of spline_polynomials (a third, K3-reviewed path construction). Reviewed code is only called."""


    @staticmethod
    def as_result(r):
        """dict from CompiledPath.certify -> object accepted by campaign.check_path_result."""
        w = r.get("witness")
        if w is not None:
            w = {"span": w["span"], "joint": w["joint"], "u": w["u"]}
        return types.SimpleNamespace(status=r["status"], leaves=r["leaves"], witness=w)


    @staticmethod
    def check(orc, geometry, timing, r, lower, upper, rng, samples):
        from curobo_certify.spline_polynomials import extract_timeopt_spans
        spans = extract_timeopt_spans(geometry, timing)
        st, bad = C.check_path_result(orc, spans, _CP.as_result(r), lower, upper, rng, samples)
        if r["status"] == "VIOLATED_X":
            w = r["witness"]
            x = O.peval(O.path_raw(spans[w["span"]])[3], Q(w["u"]))
            st["X_WITNESS_" + ("CONFIRMED" if x < 0 else "REFUTED")] = 1
            if x >= 0:
                bad.append({"what": "false VIOLATED_X", "u": w["u"]})
        return st, bad, spans


    @staticmethod
    def random_case(rng: random.Random, C_rows=None, neg_timing=False):
        """Random smooth float32 control points (Panda-range) and positive float32 timing controls."""
        nr = C_rows or rng.choice([8, 12, 20, 33])
        base = np.array([rng.uniform(-2.0, 2.0) for _ in range(7)])
        steps = np.cumsum(np.array([[rng.uniform(-0.06, 0.06) for _ in range(7)] for _ in range(nr)]), 0)
        g = (base + steps).astype(np.float32)
        g[:4] = g[0]                                      # clamped start like TimeOpt outputs
        tc = np.array([rng.uniform(0.05, 1.5) for _ in range(32)], dtype=np.float32)
        if neg_timing:
            tc[rng.randrange(4, 28)] = np.float32(-rng.uniform(2.0, 6.0))
        return g, tc


CP = _CP

C2 = ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json"
LO = np.array([-87.0] * 4 + [-12.0] * 3)
HI = -LO
F_CERT = 0.3350799560546875        # P2 bisection: CERTIFIED side (true joint-4 peak 29.151767 < 29.151956)
F_VIOL = 0.33507656921997064       # VIOLATED side (limit 29.151662 < true peak)


class RemainderAndPaths(unittest.TestCase):
    def test_small_order_remainder_is_valid(self):
        # ORDER_SMALL = 14 for |mid| <= 4: sine poly degree 29 (= Taylor degree 30), cosine degree 28 (= degree 29)
        # -> valid Lagrange bound 4^30/30! = 4.35e-15 for both, < REM_SMALL = 2e-14.
        r = Q(4) ** 30 / factorial(30)
        self.assertLess(r, Q(2, 10 ** 14))
        self.assertLess(r, Q(44, 10 ** 16))
        # DOC FINDING (N-P5b-1): the source comment's "4^29/29! < 1.1e-14" is false (= 3.26e-14 > REM_SMALL);
        # the bound is valid only through the (correct) degree+2 argument above.
        self.assertGreater(Q(4) ** 29 / factorial(29), Q(2, 10 ** 14))
        src = (ROOT / "src/curobo_certify/native/k4_rnea.cpp").read_text()
        self.assertIn("constexpr int ORDER_SMALL = 14;", src)
        self.assertIn("constexpr double REM_SMALL = 2e-14;", src)
        self.assertIn("const bool small = std::fabs(m) <= 4.0;", src)
        # the verifier keeps order 30 / 1e-27 everywhere (independent choice)
        vsrc = (ROOT / "src/curobo_certify/native/k4_verify.cpp").read_text()
        self.assertIn("for (int k = 29; k >= 0; --k) acc = acc * x2 + C[k];", vsrc)

    def test_bits_to_path_and_witness_u(self):
        from curobo_certify.k4_compiled import bits_to_path
        rng = random.Random(1)
        for _ in range(500):
            d = rng.randint(0, 30)
            bits = rng.randrange(1 << d) if d else 0
            p = bits_to_path(d, bits)
            self.assertEqual(len(p), d)
            a, b = C.leaf_interval(p)
            self.assertEqual((a, b), (Q(bits, 2 ** d), Q(bits + 1, 2 ** d)))          # producer's a, b
            self.assertEqual(int(p.replace("L", "0").replace("R", "1") or "0", 2), bits)  # verifier's inverse
            for where, pt in enumerate((a, (a + b) / 2, b)):                          # witness points a, mid, b
                self.assertEqual(Q(2 * bits + where, 2 ** (d + 1)), pt)
        # producer children: L = bits<<1, R = bits<<1|1  ==> MSB is the first decision
        self.assertEqual(bits_to_path(3, 0b011), "LRR")


@unittest.skipUnless(HAVE_GXX and HAVE_MPMATH, "g++ and mpmath required")
class CompiledPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from curobo_certify import k4_compiled as KC
        cls.KC = KC
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        KC.build(d / "k4.so")
        KC.build_verifier(d / "k4v.so")
        cls.c = load_contract(C2)
        cls.M = PandaModel(cls.c)
        cls.P = KC.CompiledPath(cls.M, d / "k4.so")
        cls.V = KC.CompiledVerifier(cls.c, d / "k4v.so")
        fx = json.loads((ROOT / "fixtures/k4_real_path_margin0000.json").read_text())
        cls.g = np.array([[float.fromhex(h) for h in r] for r in fx["effective_geometry_hex"]], dtype=np.float32)
        cls.tc = np.array([float.fromhex(h) for h in fx["timing_controls_hex"]], dtype=np.float32)
        cls.orc = C.oracle("iv", cls.c)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cert(self, f, g=None, tc=None, **kw):
        kw.setdefault("max_depth", 14)
        kw.setdefault("max_nodes", 200000)
        return self.P.certify(self.g if g is None else g, self.tc if tc is None else tc, LO * f, HI * f, **kw)

    def ver(self, r, f, g=None, tc=None, **kw):
        return self.V.verify(self.g if g is None else g, self.tc if tc is None else tc, r, LO * f, HI * f, **kw)

    # ------------------------------------------------------------------ 1. false CERTIFIED / witnesses
    def test_fixture_claims_against_oracle_including_boundary(self):
        for f, expect in ((1.0, "CERTIFIED"), (F_CERT, "CERTIFIED"), (F_VIOL, "VIOLATED_TORQUE"), (0.3, "VIOLATED_TORQUE")):
            r = self.cert(f)
            self.assertEqual(r["status"], expect, f)
            st, bad, _ = CP.check(self.orc, self.g, self.tc, r, LO * f, HI * f, random.Random(2), 4)
            self.assertEqual(bad, [], f)
            if expect == "CERTIFIED":
                self.assertTrue(self.ver(r, f)["pass"], f)
            else:
                self.assertEqual(st.get("WITNESS_CONFIRMED"), 1)

    def test_true_peak_is_inside_certified_limit(self):
        # oracle maximum of joint-4 torque near the P2 witness (span 7, u ~ 0.9174) lies between the limits
        from curobo_certify.spline_polynomials import extract_timeopt_spans
        sp = extract_timeopt_spans(self.g, self.tc)[7]
        orm = C.oracle("mp", self.c)
        vals = [O.iv_bounds(O.path_torque(orm, sp, Q(3669 + k, 4000))[3])[0] for k in range(0, 8)]
        best = max(vals)
        self.assertGreater(best, Q(F_VIOL) * 87)
        self.assertLess(best, Q(F_CERT) * 87)

    def test_random_synthetic_paths_against_oracle(self):
        rng = random.Random(3)
        seen = {}
        for k in range(6):
            g, tc = CP.random_case(rng, neg_timing=(k == 5))
            for f in (1.0, 0.4):
                r = self.cert(f, g, tc, max_depth=10, max_nodes=50000)
                seen[r["status"]] = seen.get(r["status"], 0) + 1
                st, bad, _ = CP.check(self.orc, g, tc, r, LO * f, HI * f, random.Random(k), 2)
                self.assertEqual(bad, [], (k, f, r["status"]))
                if r["status"] == "CERTIFIED":
                    self.assertTrue(self.ver(r, f, g, tc)["pass"])
        self.assertIn("CERTIFIED", seen)

    # ------------------------------------------------------------------ 2. verifier soundness / independence
    def test_verifier_rejects_forged_certificate_for_truly_violated_path(self):
        # depth-0 leaves for every span, recorded boxes = the limits themselves (contain the midpoint torque
        # and lie inside the limits), digest recomputed correctly: only the verifier's own enclosures stand
        # between this forgery and acceptance. The true peak exceeds the limit by ~1e-4 N m.
        rc = self.cert(F_CERT)
        ns = max(s for s, *_ in rc["leaves"]) + 1
        lo, hi = LO * F_VIOL, HI * F_VIOL
        forged = {"status": "CERTIFIED", "leaves": [(s, "", list(lo), list(hi)) for s in range(ns)],
                  "input_digest": self.KC.bytes_digest(self.g, self.tc, self.c["contract_sha256"], lo, hi)}
        for refine in (8, 20):
            v = self.V.verify(self.g, self.tc, forged, lo, hi, refine=refine)
            self.assertFalse(v["pass"], refine)
        # the producer's own certificate at the certified limit, replayed under the violated limit
        rc2 = dict(rc, input_digest=forged["input_digest"])
        self.assertFalse(self.V.verify(self.g, self.tc, rc2, lo, hi, refine=20)["pass"])

    def test_partition_and_digest_attacks(self):
        r = self.cert(0.34)
        L = r["leaves"]
        deep = [i for i, x in enumerate(L) if len(x[1]) >= 1]
        self.assertTrue(deep)
        i = deep[0]
        s, p, a, b = L[i]
        sib = p[:-1] + ("R" if p[-1] == "L" else "L")
        cases = {
            "missing": L[:i] + L[i + 1:],
            "duplicate": L + [L[i]],
            "parent_overlap": L + [(s, p[:-1], a, b)],
            "foreign_span": L + [(10 ** 4, "", a, b)],
            "negative_span": L + [(-1, "", a, b)],
            "bad_char": L[:i] + [(s, p[:-1] + "X", a, b)] + L[i + 1:],
            "too_deep": L[:i] + [(s, p + "L" * 31, a, b)] + L[i + 1:],
            "zero_box": L[:i] + [(s, p, [0.0] * 7, [0.0] * 7)] + L[i + 1:],
            "box_beyond_limit": L[:i] + [(s, p, a, [x + 50 for x in b])] + L[i + 1:],
            "shifted_leaf": L[:i] + [(s, sib, a, b)] + L[i + 1:] if (s, sib) not in {(x[0], x[1]) for x in L} else L[:i] + L[i + 1:],
        }
        for name, leaves in cases.items():
            self.assertFalse(self.ver(dict(r, leaves=leaves), 0.34)["pass"], name)
        self.assertFalse(self.ver(dict(r, input_digest=""), 0.34)["pass"])
        self.assertFalse(self.ver(dict(r, input_digest="0" * 64), 0.34)["pass"])
        g2 = self.g.copy()
        g2[10, 3] = np.nextafter(g2[10, 3], np.float32(9))
        self.assertFalse(self.ver(r, 0.34, g=g2)["pass"])                          # one float32 ulp of geometry
        tc2 = self.tc.copy()
        tc2[5] = np.nextafter(tc2[5], np.float32(9))
        self.assertFalse(self.ver(r, 0.34, tc=tc2)["pass"])
        self.assertFalse(self.ver(r, 0.35)["pass"])                                 # other limits
        self.assertFalse(self.ver(dict(r, status="UNKNOWN_DEPTH"), 0.34)["pass"])
        badc = copy.deepcopy(self.c)
        badc["model"]["gravity"] = ["0", "0", "0"]
        with self.assertRaises(ValueError):
            self.KC.CompiledVerifier(badc, Path(self.tmp.name) / "k4v.so")
        # the honest result still passes
        self.assertTrue(self.ver(r, 0.34)["pass"])

    def test_verifier_has_no_physical_limit_gate(self):
        # FINDING N-P5b-2 (low): unlike verify_k4(expect_physical=True), CompiledVerifier accepts ANY finite
        # limits bound in the digest; "physical" must be enforced by the caller.
        r = self.cert(0.34)
        self.assertTrue(self.ver(r, 0.34)["pass"])

    # ------------------------------------------------------------------ 5. determinism
    def test_thread_count_determinism(self):
        for f in (F_CERT, F_VIOL, 0.3):
            rs = [self.cert(f, threads=t) for t in (1, 2, 3, 8)]
            for r in rs[1:]:
                self.assertEqual(r, rs[0], f)
            if rs[0]["status"] == "CERTIFIED":
                vs = [self.ver(rs[0], f, threads=t) for t in (1, 4, 8)]
                self.assertTrue(all(v == vs[0] for v in vs))
        r = self.cert(F_CERT, max_nodes=150)
        self.assertEqual(r["status"], "UNKNOWN_NODES")
        self.assertEqual([self.cert(F_CERT, max_nodes=150, threads=t) for t in (2, 8)], [r, r])

    def test_x_negative_and_input_validation(self):
        rng = random.Random(7)
        found = False
        for _ in range(6):
            g, tc = CP.random_case(rng, neg_timing=True)
            r = self.cert(1.0, g, tc, max_depth=10)
            st, bad, _ = CP.check(self.orc, g, tc, r, LO, HI, rng, 1)
            self.assertEqual(bad, [])
            self.assertNotEqual(r["status"], "CERTIFIED")
            found |= r["status"] == "VIOLATED_X"
        self.assertTrue(found)
        self.assertEqual(self.P.certify(self.g, self.tc, LO, LO)["status"], "INVALID_INPUT")
        self.assertEqual(self.cert(1.0, max_depth=31)["status"], "INVALID_INPUT")


if __name__ == "__main__":
    unittest.main()

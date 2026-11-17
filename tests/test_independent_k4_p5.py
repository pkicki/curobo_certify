"""P5 independent review: compiled K4 backend (native/k4_rnea.cpp via k4_compiled) and P6 acceptance.

Run:  PYTHONPATH=src:. python -m unittest tests.test_independent_k4_p5 -v
Truth: independent_checks/k4p2 Lagrangian oracle (mpmath.iv rigorous / mpmath.mp 160-bit) and the
P1 trig-range reference. The compiled library is built into a temp dir (CPU only); skipped without g++.
`expectedFailure` marks desired behaviour not yet present (open findings).
"""
from __future__ import annotations

import ast
import random
import shutil
import sys
import tempfile
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

from curobo_certify.k4_interval import IA  # noqa: E402
from curobo_certify.k4_panda_model import PandaModel, load_contract  # noqa: E402

if HAVE_MPMATH:
    from independent_checks.k4p2 import campaign as C  # noqa: E402
    from independent_checks.k4p2 import oracle as O  # noqa: E402
    from independent_checks.k4 import oracle as O1  # noqa: E402

C2 = ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json"
PHYS_LO = np.array([-87.0] * 4 + [-12.0] * 3)
PHYS_HI = -PHYS_LO


def fq(x):
    return Q(float(x))


@unittest.skipUnless(HAVE_GXX and HAVE_MPMATH, "g++ and mpmath required")
class CompiledBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from curobo_certify import k4_compiled as KC
        cls.KC = KC
        cls.tmp = tempfile.TemporaryDirectory()
        cls.lib = Path(cls.tmp.name) / "k4.so"
        cls.build = KC.build(cls.lib)
        cls.c = load_contract(C2)
        cls.M = PandaModel(cls.c)
        cls.K = KC.CompiledCoefficients(cls.M, cls.lib)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---------------------------------------------------------------- arithmetic constants
    def test_remainder_constant_covers_lagrange_bound(self):
        self.assertLess(Q(8) ** 62 / factorial(62), Q(1, 10 ** 27))     # sine, degree 61
        self.assertLess(Q(8) ** 61 / factorial(61), Q(1, 10 ** 27))     # cosine, degree 60
        src = (ROOT / "src/curobo_certify/native/k4_rnea.cpp").read_text()
        self.assertIn("constexpr int ORDER = 30;", src)
        self.assertIn("constexpr double REM = 1e-27;", src)
        self.assertIn("if (!(std::fabs(m) <= 8.0)) return I(-1.0, 1.0);", src)

    def test_build_flags_disable_contraction_and_fast_math(self):
        self.assertIn("-ffp-contract=off", self.build["flags"])
        self.assertIn("-fno-fast-math", self.build["flags"])
        with self.assertRaises(Exception):
            self.KC.build(Path(self.tmp.name) / "bad.so", extra=["-ffast-math"])

    # F-P5-1 fixed (allowlist in k4_compiled.build): converted from expectedFailure by the implementer.
    def test_FINDING_build_should_refuse_individual_unsafe_flags(self):
        # F-P5-1 (low, hygiene): only flags that define __FAST_MATH__ are refused; -ffinite-math-only
        # (removes the isnan guard in operator*), -fassociative-math or a later -ffp-contract=fast pass.
        for flags in (["-ffinite-math-only"], ["-fassociative-math", "-fno-signed-zeros", "-fno-trapping-math"],
                      ["-ffp-contract=fast"]):
            with self.assertRaises(Exception):
                self.KC.build(Path(self.tmp.name) / "unsafe.so", extra=flags)

    # ---------------------------------------------------------------- isolated compiled trig
    def _trig_model(self, com):
        """base + one Z-revolute link, unit mass at `com`, gravity (0,-1,0): G = cos q (com x) / -sin q (com y)."""
        class Mini:
            pass
        m = Mini()
        z, o = IA(0.0), IA(1.0)
        m.n, m.dof, m.parent, m.joint = 2, 1, [0, 0], [-1, 0]
        m.Rf = [[[o, z, z], [z, o, z], [z, z, o]]] * 2
        m.pf = [[z, z, z]] * 2
        m.mass = [o, o]
        m.com = [[z, z, z], [IA(float(com[0])), IA(float(com[1])), z]]
        m.Ic = [[[z] * 3] * 3] * 2
        m.gravity = [z, IA(-1.0), z]
        return self.KC.CompiledCoefficients(m, self.lib)

    def test_compiled_sin_cos_enclose_exact_range(self):
        Kc, Ks = self._trig_model((1, 0)), self._trig_model((0, 1))
        rng = np.random.default_rng(1)
        mids = np.concatenate([rng.uniform(-9, 9, 300), [8.0, -8.0, np.nextafter(8.0, 0), np.nextafter(8.0, 9),
                                                        np.pi / 2, np.pi, 5e-324, 0.0]])
        rads = rng.choice([0.0, 1e-15, 1e-9, 1e-3, 0.4, 3.0], mids.size)
        q = [IA(mids - rads, mids + rads)]
        zero = [IA(np.zeros(mids.size))]
        Gc = Kc.coefficients(q, zero, zero)[2][0]
        Gs = Ks.coefficients(q, zero, zero)[2][0]
        for i in range(mids.size):
            lo, hi = fq(q[0].lo[i]), fq(q[0].hi[i])
            for enc_lo, enc_hi, cosine, sign in ((Gc.lo[i], Gc.hi[i], True, 1), (Gs.lo[i], Gs.hi[i], False, -1)):
                elo, ehi = (fq(enc_lo), fq(enc_hi)) if sign > 0 else (-fq(enc_hi), -fq(enc_lo))
                v = "AMBIGUOUS"
                for bits in (256, 640):
                    O1.set_prec(bits)
                    v = O1.true_trig_range_inside(lo, hi, elo, ehi, cosine)
                    if v != "AMBIGUOUS":
                        break
                O1.set_prec(256)
                self.assertEqual(v, "CONTAINED", (mids[i], rads[i], cosine))
            if rads[i] == 0 and abs(mids[i]) <= 8:
                self.assertLess(float(Gc.hi[i] - Gc.lo[i]), 5e-12)

    # ---------------------------------------------------------------- dynamics vs oracle
    def test_exact_states_rigorous_oracle(self):
        orc = C.oracle("iv", self.c)
        K = orc.ctx.m.mpf
        rng = np.random.default_rng(2)
        for _ in range(4):
            q, v, a = rng.uniform(-2.9, 2.9, 7), rng.uniform(-3, 3, 7), rng.uniform(-10, 10, 7)
            A, B, G = orc.abg([K(x) for x in q], [K(x) for x in v], [K(x) for x in a])
            CA, CB, CG = self.K.coefficients([IA(x) for x in q], [IA(x) for x in v], [IA(x) for x in a])
            for P_, T_ in ((CA, A), (CB, B), (CG, G)):
                for j in range(7):
                    self.assertEqual(O.verdict(P_[j].lo, P_[j].hi, T_[j]), "CONTAINED")

    def test_sub_boxes_dense_mp_samples(self):
        orc = C.oracle("mp", self.c)
        K = orc.ctx.m.mpf
        rng = np.random.default_rng(3)
        for r in (1e-6, 1e-2, 5e-2):
            cq, cv, ca = rng.uniform(-2.8, 2.8, 7), rng.uniform(-2, 2, 7), rng.uniform(-6, 6, 7)
            box = lambda c_: [IA(x - r, x + r) for x in c_]  # noqa: E731
            CA, CB, CG = self.K.coefficients(box(cq), box(cv), box(ca))
            for k in range(3):
                q, v, a = (c_ + rng.uniform(-r, r, 7) for c_ in (cq, cv, ca))
                if k == 0:
                    q, v, a = cq + r, cv - r, ca + r
                A, B, G = orc.abg([K(x) for x in q], [K(x) for x in v], [K(x) for x in a])
                for P_, T_ in ((CA, A), (CB, B), (CG, G)):
                    for j in range(7):
                        self.assertEqual(O.verdict(P_[j].lo, P_[j].hi, T_[j], C.TOL_MP), "CONTAINED")

    def test_tm1_slopes_contain_true_derivative(self):
        """d/ds of A, B, G along a cubic path vs 160-bit central differences of the oracle."""
        orc = C.oracle("mp", self.c)
        mp = orc.ctx.m
        rng = random.Random(4)
        P = [[Q(rng.randint(-1500, 1500), 1000)] + [Q(rng.randint(-700, 700), 1000) for _ in range(3)] for _ in range(7)]

        def state(s):
            q = [sum(c * s ** n for n, c in enumerate(p)) for p in P]
            qp = [sum(n * c * s ** (n - 1) for n, c in enumerate(p) if n >= 1) for p in P]
            qpp = [sum(n * (n - 1) * c * s ** (n - 2) for n, c in enumerate(p) if n >= 2) for p in P]
            qppp = [6 * p[3] for p in P]
            return q, qp, qpp, qppp
        hstep = mp.mpf(10) ** -20
        for s0 in (Q(1, 7), Q(1, 2), Q(9, 10)):
            q, qp, qpp, qppp = state(s0)
            pt = lambda xs: [IA.const(Q(x)) for x in xs]  # noqa: E731
            SA, SB, SG = self.K.slopes(pt(q), pt(qp), pt(qpp), pt(qppp))
            sm = mp.mpf(s0.numerator) / s0.denominator

            def abg_at(s):
                qq = [sum(mp.mpf(c.numerator) / c.denominator * s ** n for n, c in enumerate(p)) for p in P]
                q1 = [sum(n * mp.mpf(c.numerator) / c.denominator * s ** (n - 1) for n, c in enumerate(p) if n >= 1) for p in P]
                q2 = [sum(n * (n - 1) * mp.mpf(c.numerator) / c.denominator * s ** (n - 2) for n, c in enumerate(p) if n >= 2) for p in P]
                return orc.abg(qq, q1, q2)
            plus, minus = abg_at(sm + hstep), abg_at(sm - hstep)
            for k, S in enumerate((SA, SB, SG)):
                for j in range(7):
                    d = (plus[k][j] - minus[k][j]) / (2 * hstep)
                    self.assertEqual(O.verdict(S[j].lo, S[j].hi, d, Q(1, 10 ** 24)), "CONTAINED", (s0, k, j))  # FD error ~1e-28

    def test_nonfinite_inputs_fail_closed(self):
        n = 3
        z = [IA(np.zeros(n))] * 7
        # nan / inf ANGLES fall back to sin, cos in [-1, 1] (same as the Python reference; sound for any real
        # angle - path boxes from exact rationals never contain nan). Check identical behaviour to Python:
        q = [IA(np.array([np.nan, 0.1, np.inf]))] + [IA(np.zeros(n))] * 6
        Gc = self.K.coefficients(q, z, z)[2]
        Gp = self.M.coefficients(q, z, z)[2]
        for xc, xp in zip(Gc, Gp):
            self.assertTrue(((xc.lo <= xp.hi) & (xp.lo <= xc.hi)).all())
        # nan VELOCITY / ACCELERATION must propagate to a non-finite (rejected) enclosure
        bad = [IA(np.array([np.nan, 0.0, 0.0]))] + [IA(np.zeros(n))] * 6
        A, _, _ = self.K.coefficients(z, bad, z)
        self.assertTrue(any(not (np.isfinite(X.lo[0]) and np.isfinite(X.hi[0])) for X in A))
        A, _, _ = self.K.coefficients(z, z, bad)
        self.assertTrue(any(not (np.isfinite(X.lo[0]) and np.isfinite(X.hi[0])) for X in A))
        big = [IA(np.full(n, 1e308))] * 7
        A, B, G = self.K.coefficients(z, big, big)
        self.assertFalse(all(np.isfinite(X.lo).all() and np.isfinite(X.hi).all() for X in A))

    # ---------------------------------------------------------------- compiled tm1 cache end to end
    def test_compiled_tm1_cache_leaves_contain_oracle_and_verify(self):
        from curobo_certify.k4_reuse import GeometryCache
        from curobo_certify.k4_verify import verify_k4, verify_k4_witness
        spans = C.fixture_spans(ROOT)
        orc = C.oracle("iv", self.c)
        for mode in ("tm1", "box"):
            G = GeometryCache(self.M, spans, mode, backend=self.K)
            for f in (1.0, 0.34, 0.3):
                lo, hi = PHYS_LO * f, PHYS_HI * f
                r = G.certify(spans, lo, hi, max_depth=10)
                st, bad = C.check_path_result(orc, spans, r, lo, hi, random.Random(5), 3)
                self.assertEqual(bad, [], (mode, f))
                if r.status == "CERTIFIED":
                    self.assertTrue(verify_k4(spans, self.c, r, lo, hi, expect_physical=(f == 1.0))["pass"])
                else:
                    self.assertEqual(r.status, "VIOLATED_TORQUE")
                    self.assertTrue(verify_k4_witness(spans, self.c, r, lo, hi)["pass"])


# ==================================================================== P6 acceptance logic (static audit)
class RetimeAcceptance(unittest.TestCase):
    def setUp(self):
        self.src = (ROOT / "scripts/k4_retime_run.py").read_text()
        self.tree = ast.parse(self.src)

    def test_accepted_only_after_both_full_proofs(self):
        # the only assignment of `accepted` to a candidate is guarded by st == ("CERTIFIED", "CERTIFIED")
        assigns = [n for n in ast.walk(self.tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Tuple) and any(getattr(e, "id", "") == "accepted" for e in t.elts)
                           or getattr(t, "id", "") == "accepted" for t in n.targets)]
        guarded = [n for n in ast.walk(self.tree) if isinstance(n, ast.If)
                   and "CERTIFIED" in ast.unparse(n.test) and any(a in ast.walk(n) for a in assigns)]
        self.assertEqual(len(guarded), 1)
        self.assertIn('st == ("CERTIFIED", "CERTIFIED")', ast.unparse(guarded[0].test).replace("'", '"'))
        # K3 CERTIFIED is downgraded unless the independent verifier passes; K4 likewise with verify_k4
        self.assertIn('"REJECTED_BY_VERIFIER"', self.src)
        self.assertIn("V.verify(", self.src)
        self.assertIn("verify_k4(spans, c, r4, lo, hi", self.src)

    def test_durations_are_rigorous_brackets_of_the_certified_float32_controls(self):
        self.assertIn('g.status == "BOUNDED_EXACT_MODEL"', self.src)
        self.assertIn("Ta = duration(accepted)", self.src)
        self.assertIn("(Tn[0] - Ta[1]) / Tn[1]", self.src)     # native lower minus retimed upper: conservative
        self.assertIn(".astype(np.float32)", self.src)

    def test_acceptance_simulation(self):
        """Drive the bisection loop logic with scripted proof outcomes: never accepts a non-certified cand."""
        import textwrap
        lines = self.src.splitlines()
        i0 = next(i for i, l in enumerate(lines) if "accepted, lam_acc, tries = None, 0.0, []" in l)
        i1 = next(i for i, l in enumerate(lines) if "Tn = duration(tc0)" in l)
        code = textwrap.dedent("\n".join(lines[i0:i1]))
        tc0 = np.ones(32, dtype=np.float32)
        r_opt = np.full(32, 2.0)
        for outcomes in ([("CERTIFIED", "CERTIFIED")], [("CERTIFIED", "UNKNOWN_DEPTH")] * 10,
                         [("REJECTED_BY_VERIFIER", "CERTIFIED")] + [("CERTIFIED", "CERTIFIED")] + [("X", "Y")] * 8,
                         [("CERTIFIED", "REJECTED_BY_VERIFIER")] * 3 + [("CERTIFIED", "CERTIFIED")] * 7):
            seq = iter(outcomes + [("X", "Y")] * 10)
            log = []

            def full_certify(*args, _seq=seq, _log=log):
                st = next(_seq)
                _log.append((args[6].copy(), st))
                return st
            env = {"np": np, "tc0": tc0, "r_opt": r_opt, "full_certify": full_certify,
                   "P": None, "V": None, "M": None, "c": None, "arr": None, "geo": None, "lo": None, "hi": None,
                   "cache": None, "lim": None, "physical": True}  # implementer: new full_certify args (limit-scale)
            exec(code, env)
            acc = env["accepted"]
            if acc is not None:
                self.assertTrue(any(np.array_equal(acc, cand) and st == ("CERTIFIED", "CERTIFIED") for cand, st in log))
            else:
                self.assertFalse(any(st == ("CERTIFIED", "CERTIFIED") for _, st in log))

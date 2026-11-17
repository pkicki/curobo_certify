"""P2 independent review of the Panda K4 certifier (reviewer-written).

Run:  PYTHONPATH=src:. python -m unittest tests.test_independent_k4_p2 -v
(inserts src/ and the repository root into sys.path itself).

Truth: independent_checks/k4p2/oracle.py - Euler-Lagrange dynamics from a kinetic-energy function,
M(q)w by polarization, Coriolis/centrifugal and gravity by nested forward-mode AD, evaluated in
mpmath.iv (rigorous) or mpmath.mp (160-bit points), on a model parsed by the reviewer from the
pinned URDF (falls back to the reviewer's own reading of contract v2 when the URDF is absent).

`unittest.expectedFailure` marks DESIRED behaviour that the reviewed code does not yet have
(review findings). When a finding is fixed the test reports "unexpected success": update it.
Dense sampling is falsification evidence only, not a proof.
"""
from __future__ import annotations

import copy
import json
import random
import sys
import unittest
from dataclasses import replace
from fractions import Fraction as Q
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

from curobo_certify.k4_interval import IA, icos, isin  # noqa: E402
from curobo_certify import k4_interval as KI  # noqa: E402
from curobo_certify.k4_panda_model import PandaModel, load_contract, _rpy  # noqa: E402
from curobo_certify.k4_path import certify_path  # noqa: E402
from curobo_certify.k4_verify import WorldFrameDynamics, verify_k4, verify_k4_witness  # noqa: E402

if HAVE_MPMATH:
    from independent_checks.k4p2 import campaign as C  # noqa: E402
    from independent_checks.k4p2 import oracle as O  # noqa: E402
    from independent_checks.k4 import oracle as O1  # noqa: E402  (P1 trig range reference)

C2 = ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json"
C1 = ROOT / "docs/reports/k4/p0/panda_model_contract.json"
PHYS_LO = np.array([-87.0] * 4 + [-12.0] * 3)
PHYS_HI = -PHYS_LO


def fq(x):
    return Q(float(x))


# =============================================================================== interval layer
class IntervalLayer(unittest.TestCase):
    def _sample(self, rng, n):
        mags = [1e-310, 1e-300, 1e-12, 1.0, 1e12, 1e150]
        a = rng.uniform(-1, 1, n) * rng.choice(mags, n)
        b = a + np.abs(rng.uniform(0, 1, n) * rng.choice(mags, n))
        return a, b

    def test_basic_ops_enclose_exact_results(self):
        rng = np.random.default_rng(1)
        a, b = self._sample(rng, 3000)
        c, d = self._sample(rng, 3000)
        X, Y = IA(a, b), IA(c, d)
        S, D, P, R2, H = X + Y, X - Y, X * Y, X.sqr(), X.scale(0.5)
        for i in range(0, 3000, 7):
            for x in (a[i], b[i]):
                for y in (c[i], d[i]):
                    ex, ey = fq(x), fq(y)
                    self.assertTrue(fq(S.lo[i]) <= ex + ey <= fq(S.hi[i]))
                    self.assertTrue(fq(D.lo[i]) <= ex - ey <= fq(D.hi[i]))
                    self.assertTrue(fq(P.lo[i]) <= ex * ey <= fq(P.hi[i]))
                self.assertTrue(fq(R2.lo[i]) <= fq(x) ** 2 <= fq(R2.hi[i]))
                self.assertTrue(fq(H.lo[i]) <= fq(x) / 2 <= fq(H.hi[i]))
            if a[i] < 0 < b[i]:
                self.assertEqual(R2.lo[i], 0.0)

    def test_nonfinite_fails_closed(self):
        with np.errstate(all="ignore"):
            self.assertFalse((IA(1e308) + IA(1e308)).finite())
            self.assertFalse((IA(1e200) * IA(1e200)).finite())
            self.assertFalse((IA(0.0) * IA(np.inf)).finite())
            self.assertFalse((IA(1.0, 2.0) * IA(-np.inf, np.inf)).finite())
            self.assertFalse((IA(np.nan) * IA(1.0)).finite())
            self.assertFalse((IA(np.nan) + IA(1.0)).finite())
            self.assertFalse(IA(np.inf).scale(0.0).finite())
            self.assertFalse(IA(2.0, 1.0).finite())
            s = isin(IA(np.nan))
            self.assertEqual((float(s.lo), float(s.hi)), (-1.0, 1.0))

    def test_const_fraction_is_outward(self):
        rng = random.Random(2)
        for _ in range(2000):
            fr = Q(rng.randint(-10 ** 30, 10 ** 30), rng.randint(1, 10 ** 25)) * Q(10) ** rng.randint(-300, 250)
            c = IA.const(fr)
            self.assertTrue(fq(c.lo) <= fr <= fq(c.hi))
        self.assertEqual(IA.const(Q(1, 2)).lo, IA.const(Q(1, 2)).hi)   # exact when representable

    def test_const_of_nonfraction_hazard_is_documented(self):
        # FINDING N-IA1 (API hazard, no current call site affected): IA.const / IA.of of a Python int
        # (or a Fraction passed to scale()) is rounded to nearest, NOT enclosed.
        self.assertEqual(float(IA.const(2 ** 53 + 1).hi), 2.0 ** 53)
        self.assertLess(fq(IA.const(2 ** 53 + 1).hi), 2 ** 53 + 1)

    @unittest.skipUnless(HAVE_MPMATH, "mpmath required")
    def test_trig_encloses_exact_range(self):
        rng = np.random.default_rng(3)
        mids = np.concatenate([rng.uniform(-9, 9, 300), [8.0, -8.0, np.nextafter(8.0, 0), np.nextafter(8.0, 9),
                                                        np.pi / 2, np.pi, 3 * np.pi / 2, 5e-324]])
        rads = rng.choice([0.0, 1e-15, 1e-9, 1e-3, 0.4, 3.0], mids.size)
        X = IA(mids - rads, mids + rads)           # vectorised: shared remainder bound across the array
        for f, cosine in ((isin, False), (icos, True)):
            E = f(X)
            for i in range(mids.size):
                lo, hi = fq(X.lo[i]), fq(X.hi[i])
                v = "AMBIGUOUS"
                for bits in (256, 640):
                    O1.set_prec(bits)
                    v = O1.true_trig_range_inside(lo, hi, fq(E.lo[i]), fq(E.hi[i]), cosine)
                    if v != "AMBIGUOUS":
                        break
                O1.set_prec(256)
                self.assertEqual(v, "CONTAINED", (mids[i], rads[i], cosine))

    def test_taylor_remainder_and_cancellation_at_8(self):
        from math import factorial
        for cosine in (False, True):
            d = 61 if not cosine else 60
            r = KI._remainder_bound(8.0, cosine)
            self.assertGreaterEqual(fq(r), Q(8) ** (d + 1) / factorial(d + 1))
            self.assertLess(r, 1e-24)
            w = icos(IA(8.0)) if cosine else isin(IA(8.0))
            self.assertLess(float(w.hi - w.lo), 2e-12)        # interval-Horner cancellation width at |x|=8
        self.assertEqual(float(isin(IA(np.nextafter(8.0, 9.0))).lo), -1.0)   # fallback strictly above 8

    def test_subnormal_midpoint_lipschitz(self):
        s = isin(IA(5e-324))
        self.assertTrue(fq(s.lo) <= Q(5e-324) <= fq(s.hi))      # sin(x) <= x, > 0 here


# =============================================================================== contract / model
@unittest.skipUnless(HAVE_MPMATH, "mpmath required")
class ContractModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_contract(C2)
        cls.M = PandaModel(cls.c)

    @unittest.skipUnless(HAVE_MPMATH and O.URDF.exists(), "pinned URDF not available")
    def test_reviewer_urdf_parse_equals_contract_v2(self):
        links, g = O.parse_urdf_model()
        mine, gm = C.links_from_contract(self.c)
        self.assertEqual(tuple(g), tuple(gm))
        self.assertEqual([L.name for L in links], [L.name for L in mine])
        for a, b in zip(links, mine):
            self.assertEqual((a.parent if a.parent >= 0 else -1, a.joint, a.xyz, a.rpy, a.mass, a.com, a.inertia),
                             (b.parent, b.joint, b.xyz, b.rpy, b.mass, b.com, b.inertia), a.name)

    def test_contract_hash_and_v1_shows_right_gripper_not_lumped(self):
        raw = json.loads(C2.read_text())
        body = json.dumps(raw["model"], sort_keys=True, separators=(",", ":")).encode()
        import hashlib
        self.assertEqual(hashlib.sha256(body).hexdigest(), raw["contract_sha256"])
        v1 = json.loads(C1.read_text())["model"]
        m7 = float.fromhex(v1["masses_com_hex"][7 * 4 + 3])
        self.assertEqual(np.float32(m7), np.float32(0.35973))        # not 0.36973 (right_gripper 0.01 kg)
        self.assertEqual(int(v1["num_links"]), 13)

    def test_locked_fingers_and_attachment(self):
        L = {e["name"]: e for e in self.c["model"]["links"]}
        self.assertEqual([Q(v) for v in L["panda_leftfinger"]["origin_xyz"]], [0, Q("0.04"), Q("0.0584")])
        self.assertEqual([Q(v) for v in L["panda_rightfinger"]["origin_xyz"]], [0, -Q("0.04"), Q("0.0584")])
        names = [e["name"] for e in self.c["model"]["links"]]
        self.assertEqual(L["attached_object"]["parent"], names.index("panda_hand"))

    def test_rpy_convention_and_sensitivity(self):
        rng = np.random.default_rng(4)
        orc = C.oracle("iv", self.c)
        for _ in range(10):
            r, p, y = (Q(int(rng.integers(-3000, 3000)), 1000) for _ in range(3))
            Rp = _rpy(str(r), str(p), str(y))
            L = O.OLink("t", -1, -1, (0, 0, 0), (r, p, y), Q(1), (0, 0, 0), (0,) * 6)
            Ro = O.LagrangeOracle([L], (0, 0, 0), orc.ctx).Rfix[0]
            for i in range(3):
                for k in range(3):
                    self.assertEqual(O.verdict(Rp[i][k].lo, Rp[i][k].hi, Ro[i][k]), "CONTAINED")
        # the wrong order Rx Ry Rz is detected (sensitivity of the check)
        r, p, y = Q(1, 2), Q(-7, 10), Q(13, 10)
        Rp = _rpy(str(r), str(p), str(y))
        K = orc.ctx.const
        cr, sr = orc.ctx.cos(K(r)), orc.ctx.sin(K(r))
        cp, sp = orc.ctx.cos(K(p)), orc.ctx.sin(K(p))
        cy, sy = orc.ctx.cos(K(y)), orc.ctx.sin(K(y))
        z, o = K(0), K(1)
        Rx = [[o, z, z], [z, cr, -sr], [z, sr, cr]]
        Ry = [[cp, z, sp], [z, o, z], [-sp, z, cp]]
        Rz = [[cy, -sy, z], [sy, cy, z], [z, z, o]]
        wrong = O._mm(Rx, O._mm(Ry, Rz))
        self.assertIn("DISJOINT", [O.verdict(Rp[i][k].lo, Rp[i][k].hi, wrong[i][k]) for i in range(3) for k in range(3)])

    def _mutant(self, fn):
        bad = copy.deepcopy(self.c)
        fn(bad["model"])
        return PandaModel(bad)

    def test_gravity_sign_and_inertia_about_com_sensitivity(self):
        orc = C.oracle("iv", self.c)
        K = orc.ctx.m.mpf
        q = np.array([0.3, -0.7, 0.5, -2.0, 0.4, 1.6, -0.6])
        v = np.array([0.8, -0.5, 0.9, 0.7, -1.2, 1.1, 0.6])
        a = np.array([1.0, -2.0, 0.5, 1.5, -1.0, 2.0, 0.3])
        A, B, G = orc.abg([K(x) for x in q], [K(x) for x in v], [K(x) for x in a])
        bx = lambda arr: [IA(x) for x in arr]  # noqa: E731
        PA, PB, PG = self.M.coefficients(bx(q), bx(v), bx(a))
        for j in range(7):
            for P_, T_ in ((PA, A), (PB, B), (PG, G)):
                self.assertEqual(O.verdict(P_[j].lo, P_[j].hi, T_[j]), "CONTAINED")
        flipped = self._mutant(lambda m: m.__setitem__("gravity", ["0", "0", "9.81"]))
        _, _, FG = flipped.coefficients(bx(q), bx(v), bx(a))
        self.assertIn("DISJOINT", [O.verdict(FG[j].lo, FG[j].hi, G[j]) for j in range(7)])

        def shift(m):   # replace CoM inertia by the inertia about the link origin (parallel axis)
            for e in m["links"]:
                mm, c = Q(e["mass"]), [Q(x) for x in e["com"]]
                I = [Q(x) for x in e["inertia"]]
                cc = sum(x * x for x in c)
                e["inertia"] = [str(I[0] + mm * (cc - c[0] ** 2)), str(I[1] + mm * (cc - c[1] ** 2)),
                                str(I[2] + mm * (cc - c[2] ** 2)), str(I[3] - mm * c[0] * c[1]),
                                str(I[4] - mm * c[0] * c[2]), str(I[5] - mm * c[1] * c[2])]
        SA, _, _ = self._mutant(shift).coefficients(bx(q), bx(v), bx(a))
        self.assertIn("DISJOINT", [O.verdict(SA[j].lo, SA[j].hi, A[j]) for j in range(7)])


# =============================================================================== dynamics vs oracle
@unittest.skipUnless(HAVE_MPMATH, "mpmath required")
class DynamicsAgainstOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_contract(C2)
        cls.M, cls.W = PandaModel(cls.c), WorldFrameDynamics(cls.c)

    def test_random_exact_states_rigorous(self):
        st, bad = C.check_states(C.oracle("iv", self.c), self.M, self.W, np.random.default_rng(7), 4)
        self.assertEqual(bad, [])
        self.assertGreater(st.get("CONTAINED", 0), 100)

    def test_random_sub_boxes_dense_samples(self):
        st, bad = C.check_boxes(C.oracle("mp", self.c), self.M, self.W, np.random.default_rng(8), 2, 4)
        self.assertEqual(bad, [])


# =============================================================================== path certifier + verifier
@unittest.skipUnless(HAVE_MPMATH, "mpmath required")
class PathAndVerifierAttacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_contract(C2)
        cls.M = PandaModel(cls.c)
        cls.spans = C.fixture_spans(ROOT)
        cls.res = certify_path(cls.spans, cls.M, PHYS_LO, PHYS_HI)
        cls.orc = C.oracle("iv", cls.c)

    def V(self, res, lo=PHYS_LO, hi=PHYS_HI, spans=None, contract=None, **kw):
        return verify_k4(spans if spans is not None else self.spans, contract or self.c, res, lo, hi, **kw)

    def test_fixture_certificate_leaf_boxes_contain_oracle(self):
        self.assertEqual(self.res.status, "CERTIFIED")
        self.assertTrue(self.V(self.res)["pass"])
        st, bad = C.check_path_result(self.orc, self.spans, self.res, PHYS_LO, PHYS_HI, random.Random(9), 6)
        self.assertEqual(bad, [])

    def test_scaled_limits_witness_confirmed_by_oracle(self):
        for f in (0.3, 0.25):
            lo, hi = PHYS_LO * f, PHYS_HI * f
            r = certify_path(self.spans, self.M, lo, hi)
            self.assertEqual(r.status, "VIOLATED_TORQUE")
            st, bad = C.check_path_result(self.orc, self.spans, r, lo, hi, random.Random(1), 0)
            self.assertEqual((bad, st.get("WITNESS_CONFIRMED")), ([], 1))
            self.assertTrue(verify_k4_witness(self.spans, self.c, r, lo, hi)["pass"])

    def test_x_cases(self):
        rng = random.Random(10)
        for kind, allowed in (("neg_pocket", {"UNKNOWN_DEPTH", "UNKNOWN_NODES", "VIOLATED_X"}),
                              ("zero_start", {"CERTIFIED", "VIOLATED_TORQUE", "UNKNOWN_DEPTH"}),
                              ("zero", {"CERTIFIED", "VIOLATED_TORQUE"})):
            sp = C.synthetic_spans(rng, 2, kind)
            r = certify_path(sp, self.M, PHYS_LO, PHYS_HI, max_depth=8)
            self.assertIn(r.status, allowed, kind)
            st, bad = C.check_path_result(self.orc, sp, r, PHYS_LO, PHYS_HI, rng, 2)
            self.assertEqual(bad, [], kind)
        # SCOPE NOTE: x == 0 on the whole path (never traversed) is CERTIFIED as a static-gravity claim;
        # traversability must come from K3.

    def test_verifier_partition_attacks(self):
        L = self.res.leaves
        self.assertFalse(self.V(replace(self.res, leaves=L[1:]))["pass"])                      # missing span
        self.assertFalse(self.V(replace(self.res, leaves=L + [L[0]]))["pass"])                 # duplicate
        self.assertFalse(self.V(replace(self.res, leaves=L + [(0, "L", [0] * 7, [0] * 7)]))["pass"])   # overlap
        self.assertFalse(self.V(replace(self.res, leaves=L + [(len(self.spans), "", [0] * 7, [0] * 7)]))["pass"])
        self.assertFalse(self.V(replace(self.res, status="UNKNOWN_DEPTH"))["pass"])
        # a DIFFERENT but complete partition is re-proven by the verifier itself: accepted, and sound
        split = [x for x in L if x[0] != 0] + [(0, "L", [0] * 7, [0] * 7), (0, "R", [0] * 7, [0] * 7)]
        self.assertTrue(self.V(replace(self.res, leaves=split))["pass"])

    def test_verifier_bad_path_symbol_raises_instead_of_rejecting(self):
        # FINDING N-V1 (cosmetic, fail-closed): a malformed leaf path raises ValueError out of verify_k4.
        L = [x for x in self.res.leaves if x[0] != 0] + [(0, "L", [0] * 7, [0] * 7), (0, "X", [0] * 7, [0] * 7)]
        with self.assertRaises(ValueError):
            self.V(replace(self.res, leaves=L))

    def test_verifier_ignores_stored_leaf_boxes(self):
        # FINDING F-P2-4: stored leaf enclosures (and worst_utilization) are never checked.
        nar = replace(self.res, leaves=[(a, b, [0.0] * 7, [0.0] * 7) for (a, b, _, _) in self.res.leaves],
                      worst_utilization=[0.0] * 7)
        self.assertTrue(self.V(nar)["pass"])

    def test_limits_physical_gate(self):
        self.assertFalse(self.V(self.res, PHYS_LO * 0.95, PHYS_HI * 0.95)["pass"])
        self.assertTrue(self.V(self.res, PHYS_LO * 0.95, PHYS_HI * 0.95, expect_physical=False)["pass"])
        self.assertFalse(self.V(self.res, PHYS_LO * 2, PHYS_HI * 2)["pass"])
        self.assertEqual(certify_path(self.spans, self.M, PHYS_LO * np.nan, PHYS_HI).status, "INVALID_INPUT")
        self.assertEqual(certify_path(self.spans, self.M, PHYS_LO, PHYS_LO).status, "INVALID_INPUT")

    @unittest.expectedFailure
    def test_FINDING_nonfinite_limits_should_be_rejected(self):
        # F-P2-5: +-inf limits give a vacuous CERTIFIED from the producer.
        self.assertEqual(certify_path(self.spans, self.M, PHYS_LO * np.inf, PHYS_HI * np.inf).status, "INVALID_INPUT")

    @unittest.expectedFailure
    def test_FINDING_verifier_should_check_contract_hash(self):
        # F-P2-1: verify_k4 docstring promises a contract-hash check; a tampered model (hash unchanged) passes.
        bad = copy.deepcopy(self.c)
        bad["model"]["links"][7]["mass"] = "0.0001"
        self.assertFalse(self.V(self.res, contract=bad)["pass"])

    @unittest.expectedFailure
    def test_FINDING_span_coverage_of_unit_interval(self):
        # F-P2-2: neither producer nor verifier checks that the spans tile [0,1]; a prefix of the spans
        # (covering only [0, 8/19]) is CERTIFIED and verified although the claim is "every s in [0,1]".
        sub = self.spans[:20]
        r = certify_path(sub, self.M, PHYS_LO, PHYS_HI)
        self.assertFalse(r.status == "CERTIFIED" and self.V(r, spans=sub)["pass"])

    @unittest.expectedFailure
    def test_FINDING_path_result_binds_inputs(self):
        # F-P2-3: the G3 input digest was added to k4_reference only; K4PathResult carries no digest of
        # spans / contract / limits / source.
        self.assertTrue(any("digest" in f or "sha" in f for f in self.res.__dataclass_fields__))

    def test_witness_forgeries_rejected(self):
        lo, hi = PHYS_LO * 0.3, PHYS_HI * 0.3
        r = certify_path(self.spans, self.M, lo, hi)
        self.assertTrue(verify_k4_witness(self.spans, self.c, r, lo, hi)["pass"])
        self.assertFalse(verify_k4_witness(self.spans, self.c, r, PHYS_LO, PHYS_HI)["pass"])
        for k, v in (("u", "1/2"), ("u", "2"), ("u", "-1/3"), ("span", 999), ("span", -1), ("joint", 7),
                     ("joint", (r.witness["joint"] + 3) % 7), ("u", "abc")):
            w = dict(r.witness)
            w[k] = v
            self.assertFalse(verify_k4_witness(self.spans, self.c, replace(r, witness=w), lo, hi)["pass"], (k, v))
        self.assertFalse(verify_k4_witness(self.spans, self.c, self.res, lo, hi)["pass"])

    def test_budget_fail_closed(self):
        self.assertEqual(certify_path(self.spans, self.M, PHYS_LO, PHYS_HI, max_nodes=0).status, "UNKNOWN_NODES")
        self.assertEqual(certify_path(self.spans, self.M, PHYS_LO, PHYS_HI, max_nodes=46).status, "UNKNOWN_NODES")
        r = certify_path(self.spans, self.M, PHYS_LO * 0.62, PHYS_HI * 0.62, max_depth=-1)
        self.assertIn(r.status, ("UNKNOWN_DEPTH", "VIOLATED_TORQUE"))
        self.assertEqual(certify_path(self.spans[:1], self.M, PHYS_LO[:6], PHYS_HI[:6]).status, "INVALID_INPUT")


# =============================================================================== G1 / G3 fixes (k4_reference)
class G1G3Fixes(unittest.TestCase):
    def setUp(self):
        from curobo_certify.k4_reference import Pendulum, PathSpan, certify_torque
        self.P = Pendulum(1, 1, 1, 1)
        self.sp = [PathSpan.from_values(0, 1, [[0, 1]], [1])]
        self.lim = [(-Q(1, 2), Q(1, 2))]
        self.r = certify_torque(self.sp, self.P, self.lim)
        self.assertEqual(self.r.status, "VIOLATED_TORQUE")

    def test_replay_witness_accepts_true_and_rejects_forged(self):
        from curobo_certify.k4_reference import Pendulum, replay_witness
        self.assertTrue(replay_witness(self.sp, self.P, self.lim, self.r))
        for k, v in (("u", Q(1, 4)), ("u", Q(2)), ("joint", 1), ("span", 1), ("span", -1), ("u", "x")):
            w = dict(self.r.witness)
            w[k] = v
            self.assertFalse(replay_witness(self.sp, self.P, self.lim, replace(self.r, witness=w)), (k, v))
        self.assertFalse(replay_witness(self.sp, self.P, [(-1, 1)], self.r))                 # digest mismatch
        self.assertFalse(replay_witness(self.sp, Pendulum(1, 1, 1, 2), self.lim, self.r))   # model mismatch

    def test_input_digest_sensitivity(self):
        from curobo_certify.k4_reference import Pendulum, PathSpan, input_digest
        d = input_digest(self.sp, self.P, self.lim)
        self.assertNotEqual(d, input_digest(self.sp, Pendulum(1, 1, 1, 2), self.lim))
        self.assertNotEqual(d, input_digest(self.sp, self.P, [(-Q(1, 2), Q(1, 3))]))
        self.assertNotEqual(d, input_digest([PathSpan.from_values(0, 1, [[0, 1]], [2])], self.P, self.lim))
        self.assertEqual(self.r.input_digest, d)

    @unittest.expectedFailure
    def test_FINDING_stripped_digest_should_not_bypass_binding(self):
        # F-G3: replay checks the digest only `if result.input_digest`; stripping it disables the binding.
        from curobo_certify.k4_reference import replay_witness
        stripped = replace(self.r, input_digest="")
        self.assertFalse(replay_witness(self.sp, self.P, [(-Q(1, 3), Q(1, 3))], stripped))


if __name__ == "__main__":
    unittest.main()

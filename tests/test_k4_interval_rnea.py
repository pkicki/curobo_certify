"""K4 interval arithmetic, rigorous trig and Panda interval RNEA regression tests."""
import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from curobo_certify.k4_interval import IA, icos, isin  # noqa: E402
from curobo_certify.k4_panda_model import PandaModel, load_contract  # noqa: E402

CONTRACT = ROOT / "docs/reports/k4/p0/panda_model_contract.json"
FIXTURE = ROOT / "fixtures/k4_pinocchio_parity_states.json"

try:
    import mpmath
except ImportError:  # pragma: no cover
    mpmath = None


class IntervalArithmetic(unittest.TestCase):
    def test_mul_all_sign_cases_contain_exact(self):
        rng = np.random.default_rng(1)
        for _ in range(2000):
            a, b, c, d = sorted(rng.uniform(-5, 5, 2)) + sorted(rng.uniform(-5, 5, 2))
            r = IA(a, b) * IA(c, d)
            for x in (a, b, (a + b) / 2):
                for y in (c, d, (c + d) / 2):
                    p = Fraction(x) * Fraction(y)
                    self.assertTrue(Fraction(float(r.lo)) <= p <= Fraction(float(r.hi)))

    def test_add_sub_outward(self):
        x = IA(0.1) + IA(0.2)
        self.assertTrue(Fraction(float(x.lo)) <= Fraction(0.1) + Fraction(0.2) <= Fraction(float(x.hi)))
        y = IA(1.0, 2.0) - IA(0.5, 3.0)
        self.assertLessEqual(y.lo, -2.0)
        self.assertGreaterEqual(y.hi, 1.5)

    def test_nonfinite_fails_closed(self):
        self.assertFalse((IA(1e308) * IA(1e308)).finite())
        self.assertFalse((IA(0.0) * IA(np.inf)).finite())

    @unittest.skipIf(mpmath is None, "mpmath required")
    def test_sin_cos_enclosures_contain_true_range(self):
        mpmath.mp.prec = 200
        rng = np.random.default_rng(3)
        lo = rng.uniform(-9, 9, 400)
        w = rng.choice([0.0, 1e-12, 1e-6, 1e-3, 0.3, 2.0], 400)
        a = IA(lo, lo + w)
        S, C = isin(a), icos(a)
        for i in range(400):
            pts = [lo[i], lo[i] + w[i]] + list(np.linspace(lo[i], lo[i] + w[i], 9))
            # analytic extrema inside the box
            k0 = int(np.floor((lo[i] - np.pi) / (np.pi / 2))) - 1
            for k in range(k0, k0 + 12):
                t = k * np.pi / 2
                if lo[i] <= t <= lo[i] + w[i]:
                    pts.append(t)
            for t in pts:
                s, c = mpmath.sin(mpmath.mpf(float(t))), mpmath.cos(mpmath.mpf(float(t)))
                self.assertTrue(S.lo[i] <= s <= S.hi[i], (lo[i], w[i], t))
                self.assertTrue(C.lo[i] <= c <= C.hi[i], (lo[i], w[i], t))
        # tight for point boxes inside the Taylor range
        self.assertLess(float((S.hi - S.lo)[(w == 0) & (np.abs(lo) < 8)].max()), 2e-12)  # Taylor cancellation near 8
        self.assertLess(float((S.hi - S.lo)[(w == 0) & (np.abs(lo) < 3.8)].max()), 5e-14)  # Panda joint range

    def test_large_angle_fallback(self):
        s = isin(IA(100.0, 100.5))
        self.assertEqual((float(s.lo), float(s.hi)), (-1.0, 1.0))
        s = isin(IA(np.nan, np.nan))
        self.assertEqual((float(s.lo), float(s.hi)), (-1.0, 1.0))


class PandaRnea(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_contract(CONTRACT)
        cls.M = PandaModel(cls.c)

    def test_contract_tamper_rejected(self):
        bad = json.loads(CONTRACT.read_text())
        bad["model"]["masses_com_hex"][7] = float(2.0).hex()
        p = ROOT / "runs" / "_tmp_bad_contract.json"
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            load_contract(p)
        p.unlink()

    def test_payload_override_blocked(self):
        with self.assertRaises(ValueError):
            PandaModel(self.c, payload_override={"mass": 3.0})

    def test_algorithm_against_pinocchio_fixture(self):
        """Pinocchio (independent URDF parse + RNEA) torques must lie inside the interval RNEA computed on
        Pinocchio's own parameterisation (validates frames, axes, signs, recursion)."""
        fx = json.loads(FIXTURE.read_text())
        P = fx["pinocchio_parameterisation"]
        hx = lambda a: [float(x).hex() for x in np.asarray(a, dtype=np.float64).reshape(-1)]  # noqa: E731
        T = [np.hstack([np.array(p["R"]), np.array(p["p"])[:, None]]) for p in P]
        model = {"num_links": len(P), "num_dof": len(P) - 1, "parent": list(range(-1, len(P) - 1)),
                 "joint_index": [-1] + list(range(len(P) - 1)), "joint_type": [-1] + [5] * (len(P) - 1),
                 "fixed_transforms_hex": hx(T), "masses_com_hex": hx([p["com"] + [p["mass"]] for p in P]),
                 "inertias_hex": hx([[p["I"][0][0], p["I"][1][1], p["I"][2][2], p["I"][0][1], p["I"][0][2], p["I"][1][2]] for p in P]),
                 "gravity_hex": hx(fx["gravity"])}
        model["parent"][0] = 0
        M = PandaModel({"model": model, "contract_sha256": "fixture"})
        for st in fx["states"]:
            box = lambda k: [IA(float.fromhex(h)) for h in st[k]]  # noqa: E731
            tau = M.torque(box("q"), box("qd"), box("qdd"))
            for j, t in enumerate(tau):
                ref = st["tau_pinocchio"][j]
                tol = 1e-12 * max(1.0, abs(ref))
                self.assertTrue(t.lo - tol <= ref <= t.hi + tol, (j, ref, t.lo, t.hi))

    def test_coefficient_decomposition_contains_direct_torque(self):
        """tau(q, q' sqrt(x), q'' x + q' x'/2) must lie in A x + B x' + G for any x > 0, x'."""
        rng = np.random.default_rng(5)
        for _ in range(10):
            q, qp, qpp = rng.uniform(-2, 2, 7), rng.uniform(-1, 1, 7), rng.uniform(-3, 3, 7)
            x, xp = 0.75, -1.25
            A, B, G = self.M.coefficients([IA(v) for v in q], [IA(v) for v in qp], [IA(v) for v in qpp])
            qd = [IA(v) * IA(np.sqrt(x)) for v in qp]  # sqrt(0.75) rounded: enclose instead
            sx = IA(np.nextafter(np.sqrt(x), -np.inf), np.nextafter(np.sqrt(x), np.inf))
            qd = [IA(v) * sx for v in qp]
            qdd = [IA(a) * IA(x) + IA(b) * IA(xp * 0.5) for a, b in zip(qpp, qp)]
            tau = self.M.torque([IA(v) for v in q], qd, qdd)
            for j in range(7):
                dec = A[j] * IA(x) + B[j] * IA(xp) + G[j]
                self.assertTrue(dec.lo <= tau[j].hi and tau[j].lo <= dec.hi, j)

    def test_box_enclosure_contains_point_samples(self):
        rng = np.random.default_rng(9)
        c = rng.uniform(-2, 2, 7)
        r = 0.02
        boxes = [IA(v - r, v + r) for v in c]
        G = self.M.rnea(boxes, None, None, gravity=True)
        for _ in range(50):
            p = c + rng.uniform(-r, r, 7)
            g = self.M.rnea([IA(v) for v in p], None, None, gravity=True)
            for j in range(7):
                self.assertTrue(G[j].lo <= g[j].lo and g[j].hi <= G[j].hi)


if __name__ == "__main__":
    unittest.main()

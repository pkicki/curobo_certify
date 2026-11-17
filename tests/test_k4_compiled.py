"""Differential tests: compiled K4 RNEA backend vs the Python interval reference."""
import json
import sys
import tempfile
import unittest
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify import k4_compiled as KC  # noqa: E402
from curobo_certify.k4_interval import DIA, IA  # noqa: E402
from curobo_certify.k4_panda_model import PandaModel, load_contract  # noqa: E402
from curobo_certify.k4_reuse import GeometryCache  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402


class CompiledK4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.build = KC.build(Path(cls.tmp.name) / "k4.so")
        cls.c = load_contract(ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json")
        cls.M = PandaModel(cls.c)
        cls.K = KC.CompiledCoefficients(cls.M, Path(cls.tmp.name) / "k4.so")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _boxes(self, rng, n, r):
        c = [rng.uniform(-2.5, 2.5, n) for _ in range(7)], [rng.uniform(-1, 1, n) for _ in range(7)], \
            [rng.uniform(-3, 3, n) for _ in range(7)], [rng.uniform(-9, 9, n) for _ in range(7)]
        return [[IA(x - r * (k + 1), x + r * (k + 1)) for x in blk] for k, blk in enumerate(c)]

    def test_box_coefficients_contain_and_match_python(self):
        rng = np.random.default_rng(4)
        for r in (0.0, 1e-4, 1e-2):
            q, qp, qpp, _ = self._boxes(rng, 200, r)
            Ac, Bc, Gc = self.K.coefficients(q, qp, qpp)
            Ap, Bp, Gp = self.M.coefficients(q, qp, qpp)
            for xc, xp in zip(Ac + Bc + Gc, Ap + Bp + Gp):
                # both are rigorous enclosures of the same quantity: they must intersect, and widths agree
                self.assertTrue(((xc.lo <= xp.hi) & (xp.lo <= xc.hi)).all())
                wc, wp = xc.hi - xc.lo, xp.hi - xp.lo
                self.assertTrue((wc <= 1.01 * wp + 1e-9).all() and (wp <= 1.01 * wc + 1e-9).all())

    def test_point_values_inside_compiled_box(self):
        rng = np.random.default_rng(5)
        q, qp, qpp, _ = self._boxes(rng, 50, 1e-2)
        Ac, Bc, Gc = self.K.coefficients(q, qp, qpp)
        for t in range(10):
            pts = [[IA(b.lo + (b.hi - b.lo) * rng.random(50)) for b in blk] for blk in (q, qp, qpp)]
            Ap, Bp, Gp = self.M.coefficients(*pts)
            for xc, xp in zip(Ac + Bc + Gc, Ap + Bp + Gp):
                self.assertTrue(((xc.lo <= xp.lo) & (xp.hi <= xc.hi)).all())

    def test_slopes_match_python_dual(self):
        rng = np.random.default_rng(6)
        q, qp, qpp, qppp = self._boxes(rng, 100, 1e-3)
        S = self.K.slopes(q, qp, qpp, qppp)
        dq = [DIA(q[j], qp[j]) for j in range(7)]
        dqp = [DIA(qp[j], qpp[j]) for j in range(7)]
        dqpp = [DIA(qpp[j], qppp[j]) for j in range(7)]
        A, B, G = self.M.coefficients(dq, dqp, dqpp)
        for xc, xp in zip(S[0] + S[1] + S[2], [x.d for x in A + B + G]):
            self.assertTrue(((xc.lo <= xp.hi) & (xp.lo <= xc.hi)).all())

    def test_compiled_tm1_cache_equals_python_status(self):
        fx = json.loads((ROOT / "fixtures/k4_real_path_margin0000.json").read_text())
        g = np.array([[float.fromhex(h) for h in r] for r in fx["effective_geometry_hex"]], dtype=np.float32)
        tc = np.array([float.fromhex(h) for h in fx["timing_controls_hex"]], dtype=np.float32)
        lo = np.array([float(Q(v)) for v in self.c["model"]["effort_lower"]])
        sp = extract_timeopt_spans(g, tc)
        rc = GeometryCache(self.M, sp, "tm1", backend=self.K).certify(sp, lo, -lo)
        rp = GeometryCache(self.M, sp, "tm1").certify(sp, lo, -lo)
        self.assertEqual(rc.status, rp.status)
        self.assertAlmostEqual(max(rc.worst_utilization), max(rp.worst_utilization), places=6)

    def test_fast_math_refused(self):
        with self.assertRaises(Exception):
            KC.build(Path(self.tmp.name) / "bad.so", extra=["-ffast-math"])


if __name__ == "__main__":
    unittest.main()

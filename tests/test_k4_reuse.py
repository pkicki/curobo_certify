"""K4 P3 geometry-once / retime-many caches (box, tm1) and dual intervals."""
import json
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.k4_interval import DIA, IA, icos, isin  # noqa: E402
from curobo_certify.k4_panda_model import PandaModel, load_contract  # noqa: E402
from curobo_certify.k4_path import _point_boxes, torque_enclosure  # noqa: E402
from curobo_certify.k4_reference import dpoly  # noqa: E402
from curobo_certify.k4_reuse import GeometryCache, _leaf_bounds  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402


class Reuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_contract(ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json")
        cls.M = PandaModel(cls.c)
        fx = json.loads((ROOT / "fixtures/k4_real_path_margin0000.json").read_text())
        cls.g = np.array([[float.fromhex(h) for h in r] for r in fx["effective_geometry_hex"]], dtype=np.float32)
        cls.tc = np.array([float.fromhex(h) for h in fx["timing_controls_hex"]], dtype=np.float32)
        cls.lo = np.array([float(Q(v)) for v in cls.c["model"]["effort_lower"]])
        cls.hi = np.array([float(Q(v)) for v in cls.c["model"]["effort_upper"]])

    def _check_contains(self, spans, res):
        raws = [[tuple(p) for p in s.q] + [dpoly(p, 1, s.right - s.left) for p in s.q] + [dpoly(p, 2, s.right - s.left) for p in s.q]
                + [tuple(s.x), dpoly(s.x, 1, s.right - s.left)] for s in spans]
        pts, rl, own = [], [], []
        for (si, path, tlo, thi) in res.leaves:
            a, b = _leaf_bounds(path)
            for t in range(9):
                pts.append(a + (b - a) * Q(t, 8)); rl.append(raws[si]); own.append((tlo, thi))
        pq, pqp, pqpp, px, pxp, _ = _point_boxes(rl, pts, 7)
        tau = torque_enclosure(self.M, pq, pqp, pqpp, px, pxp)
        L = np.stack([t.lo for t in tau], 1); H = np.stack([t.hi for t in tau], 1)
        for i, (tlo, thi) in enumerate(own):
            self.assertTrue((L[i] >= np.array(tlo)).all() and (H[i] <= np.array(thi)).all(), i)

    def test_tm1_and_box_sound_across_retimings(self):
        base = extract_timeopt_spans(self.g, self.tc)
        for mode in ("box", "tm1"):
            C = GeometryCache(self.M, base, mode)
            for k in (1.0, 1.3):
                sp = extract_timeopt_spans(self.g, (self.tc * np.float32(k * k)).astype(np.float32))
                r = C.certify(sp, self.lo, self.hi)
                self.assertEqual(r.status, "CERTIFIED")
                self._check_contains(sp, r)

    def test_tm1_tighter_than_box(self):
        base = extract_timeopt_spans(self.g, self.tc)
        rb = GeometryCache(self.M, base, "box").certify(base, self.lo, self.hi)
        rt = GeometryCache(self.M, base, "tm1").certify(base, self.lo, self.hi)
        self.assertLessEqual(max(rt.worst_utilization), max(rb.worst_utilization))

    def test_geometry_mismatch_rejected(self):
        base = extract_timeopt_spans(self.g, self.tc)
        C = GeometryCache(self.M, base, "box")
        g2 = self.g.copy(); g2[5, 2] += np.float32(1e-3)
        with self.assertRaises(ValueError):
            C.certify(extract_timeopt_spans(g2, self.tc), self.lo, self.hi)

    def test_dual_derivative_encloses_finite_difference(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            x0, dx = rng.uniform(-3, 3), rng.uniform(-2, 2)
            h = 1e-6
            d = DIA(IA(x0 - h, x0 + h), IA(dx))
            s, c = isin(d), icos(d)
            # d/dt sin(x0 + dx t) = cos(x0) dx
            self.assertTrue(s.d.lo <= np.cos(x0) * dx <= s.d.hi)
            self.assertTrue(c.d.lo <= -np.sin(x0) * dx <= c.d.hi)


if __name__ == "__main__":
    unittest.main()

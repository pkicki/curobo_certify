"""K4 Panda path certificate: producer, independent world-frame verifier, witnesses, falsification."""
import json
import sys
import unittest
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.k4_interval import IA  # noqa: E402
from curobo_certify.k4_panda_model import PandaModel, load_contract  # noqa: E402
from curobo_certify.k4_path import certify_path, torque_enclosure  # noqa: E402
from curobo_certify.k4_reference import dpoly, eval_poly  # noqa: E402
from curobo_certify.k4_verify import WorldFrameDynamics, verify_k4, verify_k4_witness  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402

C2 = ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json"
FX = ROOT / "fixtures/k4_real_path_margin0000.json"


class K4PathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_contract(C2)
        cls.M = PandaModel(cls.c)
        fx = json.loads(FX.read_text())
        geo = np.array([[float.fromhex(h) for h in r] for r in fx["effective_geometry_hex"]], dtype=np.float32)
        tc = np.array([float.fromhex(h) for h in fx["timing_controls_hex"]], dtype=np.float32)
        cls.spans = extract_timeopt_spans(geo, tc)
        cls.lo = np.array([float(Q(v)) for v in cls.c["model"]["effort_lower"]])
        cls.hi = np.array([float(Q(v)) for v in cls.c["model"]["effort_upper"]])
        cls.res = certify_path(cls.spans, cls.M, cls.lo, cls.hi)

    def test_certified_and_independently_verified(self):
        self.assertEqual(self.res.status, "CERTIFIED")
        v = verify_k4(self.spans, self.c, self.res, self.lo, self.hi)
        self.assertTrue(v["pass"], v)

    def test_verifier_rejects_tampering_and_foreign_limits(self):
        r = replace(self.res, leaves=self.res.leaves[1:])
        self.assertFalse(verify_k4(self.spans, self.c, r, self.lo, self.hi)["pass"])
        r = replace(self.res, leaves=self.res.leaves + [self.res.leaves[0]])
        self.assertFalse(verify_k4(self.spans, self.c, r, self.lo, self.hi)["pass"])
        self.assertFalse(verify_k4(self.spans, self.c, self.res, self.lo * 2, self.hi * 2)["pass"])
        self.assertFalse(verify_k4(self.spans, self.c, replace(self.res, status="UNKNOWN_DEPTH"), self.lo, self.hi)["pass"])

    def test_world_frame_and_link_frame_enclosures_intersect(self):
        W = WorldFrameDynamics(self.c)
        rng = np.random.default_rng(11)
        for _ in range(40):
            q, v, a = ([IA(x) for x in rng.uniform(-2.5, 2.5, 7)] for _ in range(3))
            for t1, t2 in zip(self.M.torque(q, v, a), W.torque(q, v, a, True)):
                self.assertTrue(t1.lo <= t2.hi and t2.lo <= t1.hi)

    def test_dense_points_inside_certified_leaf_boxes(self):
        """Falsification: rigorous point torque at many s must lie in the box of the leaf covering s."""
        by = {}
        for (si, path, lo, hi) in self.res.leaves:
            a, b = Q(0), Q(1)
            for ch in path:
                m = (a + b) / 2
                a, b = (a, m) if ch == "L" else (m, b)
            by.setdefault(si, []).append((a, b, lo, hi))
        rng = np.random.default_rng(3)
        for _ in range(150):
            si = int(rng.integers(len(self.spans)))
            u = Q(int(rng.integers(0, 10 ** 6)), 10 ** 6)
            sp = self.spans[si]
            h = sp.right - sp.left
            raw = ([p for p in sp.q] + [dpoly(p, 1, h) for p in sp.q] + [dpoly(p, 2, h) for p in sp.q] + [sp.x, dpoly(sp.x, 1, h)])
            box = [IA.const(eval_poly(p, u)) for p in raw]
            tau = torque_enclosure(self.M, box[:7], box[7:14], box[14:21], box[21], box[22])
            for a, b, lo, hi in by[si]:
                if a <= u <= b:
                    for j in range(7):
                        self.assertTrue(lo[j] <= float(tau[j].lo) and float(tau[j].hi) <= hi[j])
                    break

    def test_tightened_limits_give_true_witness_and_forgery_rejected(self):
        f = 0.5 * max(self.res.worst_utilization)  # well below the certified peak utilization
        lo, hi = self.lo * f, self.hi * f
        r = certify_path(self.spans, self.M, lo, hi)
        self.assertEqual(r.status, "VIOLATED_TORQUE")
        self.assertTrue(verify_k4_witness(self.spans, self.c, r, lo, hi)["pass"])
        self.assertFalse(verify_k4_witness(self.spans, self.c, r, self.lo, self.hi)["pass"])
        w = dict(r.witness)
        w["joint"] = (w["joint"] + 6) % 7
        self.assertFalse(verify_k4_witness(self.spans, self.c, replace(r, witness=w), lo, hi)["pass"])

    def test_budget_fails_closed(self):
        r = certify_path(self.spans, self.M, self.lo * 0.96, self.hi * 0.96, max_depth=0)
        self.assertIn(r.status, ("UNKNOWN_DEPTH", "VIOLATED_TORQUE", "CERTIFIED"))
        r = certify_path(self.spans, self.M, self.lo, self.hi, max_nodes=10)
        self.assertEqual(r.status, "UNKNOWN_NODES")


if __name__ == "__main__":
    unittest.main()

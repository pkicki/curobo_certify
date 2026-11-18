"""Compiled K4 pipeline (C++ producer k4_certify_path + independent C++ verifier k4v_verify)."""
import copy
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
from curobo_certify.k4_panda_model import PandaModel, load_contract  # noqa: E402
from curobo_certify.k4_path import K4PathResult, k4_input_digest  # noqa: E402
from curobo_certify.k4_reuse import GeometryCache  # noqa: E402
from curobo_certify.k4_verify import verify_k4, verify_k4_witness  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402


class CompiledPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        KC.build(d / "k4.so")
        KC.build_verifier(d / "k4v.so")
        cls.c = load_contract(ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json")
        cls.M = PandaModel(cls.c)
        cls.P = KC.CompiledPath(cls.M, d / "k4.so")
        cls.V = KC.CompiledVerifier(cls.c, d / "k4v.so")
        fx = json.loads((ROOT / "fixtures/k4_real_path_margin0000.json").read_text())
        cls.g = np.array([[float.fromhex(h) for h in r] for r in fx["effective_geometry_hex"]], dtype=np.float32)
        cls.tc = np.array([float.fromhex(h) for h in fx["timing_controls_hex"]], dtype=np.float32)
        cls.lo = np.array([float(Q(v)) for v in cls.c["model"]["effort_lower"]])
        cls.hi = -cls.lo

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_matches_python_tm1_and_python_verifier_accepts(self):
        r = self.P.certify(self.g, self.tc, self.lo, self.hi)
        sp = extract_timeopt_spans(self.g, self.tc)
        rp = GeometryCache(self.M, sp, "tm1").certify(sp, self.lo, self.hi)
        self.assertEqual(r["status"], rp.status)
        self.assertEqual([(s, p) for s, p, *_ in r["leaves"]], [(s, p) for s, p, *_ in rp.leaves])
        self.assertAlmostEqual(max(r["worst_utilization"]), max(rp.worst_utilization), places=6)
        res = K4PathResult(r["status"], "x", r["nodes"], r["max_depth"], r["leaves"], None, r["worst_utilization"])
        res.input_digest = k4_input_digest(sp, self.c["contract_sha256"], self.lo, self.hi)
        self.assertTrue(verify_k4(sp, self.c, res, self.lo, self.hi, refine=8)["pass"])

    def test_thread_count_determinism(self):
        r1 = self.P.certify(self.g, self.tc, self.lo, self.hi, threads=1)
        r8 = self.P.certify(self.g, self.tc, self.lo, self.hi, threads=8)
        self.assertEqual(r1, r8)
        v1 = self.V.verify(self.g, self.tc, r1, self.lo, self.hi, threads=1)
        v8 = self.V.verify(self.g, self.tc, r1, self.lo, self.hi, threads=8)
        self.assertEqual(v1, v8)
        self.assertTrue(v1["pass"])

    def test_verifier_rejects_tampering(self):
        r = self.P.certify(self.g, self.tc, self.lo, self.hi)
        V = lambda x, **kw: self.V.verify(kw.get("g", self.g), kw.get("tc", self.tc), x, kw.get("lo", self.lo), kw.get("hi", self.hi))  # noqa: E731
        bad = copy.deepcopy(r); bad["leaves"] = bad["leaves"][1:]
        self.assertFalse(V(bad)["pass"])
        bad = copy.deepcopy(r); bad["leaves"] = bad["leaves"] + [bad["leaves"][0]]
        self.assertFalse(V(bad)["pass"])
        bad = copy.deepcopy(r); s, p, a, b = bad["leaves"][3]; bad["leaves"][3] = (s, p, [0.0] * 7, [0.0] * 7)
        self.assertFalse(V(bad)["pass"])
        bad = copy.deepcopy(r); bad["leaves"][0] = (999, "", [0.0] * 7, [0.0] * 7)
        self.assertFalse(V(bad)["pass"])
        bad = copy.deepcopy(r); bad["leaves"][0] = (0, "LX", [0.0] * 7, [0.0] * 7)
        self.assertFalse(V(bad)["pass"])
        bad = copy.deepcopy(r); bad["input_digest"] = ""
        self.assertFalse(V(bad)["pass"])
        self.assertFalse(V(r, tc=self.tc * np.float32(1.05))["pass"])
        self.assertFalse(V(r, lo=self.lo * 2, hi=self.hi * 2)["pass"])
        self.assertFalse(V(dict(r, status="UNKNOWN_DEPTH"))["pass"])
        badc = copy.deepcopy(self.c); badc["model"]["links"][7]["mass"] = "0.0001"
        with self.assertRaises(ValueError):
            KC.CompiledVerifier(badc, Path(self.tmp.name) / "k4v.so")

    def test_witness_replays_in_python_world_frame(self):
        r0 = self.P.certify(self.g, self.tc, self.lo, self.hi)
        f = 0.5 * max(r0["worst_utilization"])
        lo, hi = self.lo * f, self.hi * f
        r = self.P.certify(self.g, self.tc, lo, hi)
        self.assertEqual(r["status"], "VIOLATED_TORQUE")
        sp = extract_timeopt_spans(self.g, self.tc)
        res = K4PathResult("VIOLATED_TORQUE", "x", r["nodes"], r["max_depth"], [], r["witness"], None)
        res.input_digest = k4_input_digest(sp, self.c["contract_sha256"], lo, hi)
        self.assertTrue(verify_k4_witness(sp, self.c, res, lo, hi)["pass"])

    def test_budgets_and_invalid_input_fail_closed(self):
        self.assertEqual(self.P.certify(self.g, self.tc, self.lo, self.hi, max_nodes=10)["status"], "UNKNOWN_NODES")
        self.assertEqual(self.P.certify(self.g, self.tc, self.lo * np.inf, self.hi * np.inf)["status"], "INVALID_INPUT")
        g2 = self.g.copy(); g2[3, 2] = np.nan
        self.assertEqual(self.P.certify(g2, self.tc, self.lo, self.hi)["status"], "INVALID_INPUT")

    def test_p5b_hardening_shapes_physical_gate_digest(self):
        r = self.P.certify(self.g, self.tc, self.lo, self.hi)
        # N-P5b-4: wrong timing length / geometry width fail closed in producer and verifier
        self.assertEqual(self.P.certify(self.g, self.tc[:31], self.lo, self.hi)["status"], "INVALID_INPUT")
        self.assertEqual(self.P.certify(self.g[:, :6], self.tc, self.lo, self.hi)["status"], "INVALID_INPUT")
        self.assertFalse(self.V.verify(self.g, self.tc[:31], r, self.lo, self.hi)["pass"])
        # the digest binds shapes: same bytes reshaped give a different digest
        self.assertNotEqual(KC.bytes_digest(self.g, self.tc, "x", self.lo, self.hi),
                            KC.bytes_digest(self.g.reshape(7, -1), self.tc, "x", self.lo, self.hi))
        # N-P5b-2: expect_physical accepts exactly the contract limits, rejects anything else
        self.assertTrue(self.V.verify(self.g, self.tc, r, self.lo, self.hi, expect_physical=True)["pass"])
        r2 = self.P.certify(self.g, self.tc, self.lo * 2, self.hi * 2)
        self.assertTrue(self.V.verify(self.g, self.tc, r2, self.lo * 2, self.hi * 2)["pass"])
        self.assertFalse(self.V.verify(self.g, self.tc, r2, self.lo * 2, self.hi * 2, expect_physical=True)["pass"])


if __name__ == "__main__":
    unittest.main()

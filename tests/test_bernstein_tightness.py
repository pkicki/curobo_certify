import json
import random
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from curobo_certify.bernstein_tightness import bracket_polynomial_peak, diagnostic_dict
from explore_k3_models import analyze


class TightnessTests(unittest.TestCase):
    def test_constant_exact(self):
        p = bracket_polynomial_peak([F(-5)])
        self.assertEqual(p.status, 'BOUNDED')
        self.assertEqual((p.lower, p.upper), (F(-5), F(-5)))
        self.assertEqual(p.overestimation_upper, 0)

    def test_no_split_hull_can_overestimate(self):
        # g=-1+3u-3u², exact max=-1/4 at u=1/2.
        p = bracket_polynomial_peak([F(-1), F(3), F(-3)], tolerance=0)
        self.assertEqual(p.status, 'BOUNDED')
        self.assertEqual((p.lower, p.upper), (F(-1, 4), F(-1, 4)))
        self.assertGreater(p.initial_hull_upper, 0)
        self.assertEqual(p.margin_lower, F(1, 4))
        self.assertEqual(p.overestimation_lower, p.overestimation_upper)

    def test_budget_unknown_still_has_valid_upper_bound(self):
        p = bracket_polynomial_peak([F(-1), F(3), F(-3)], tolerance=0, max_depth=0)
        self.assertEqual(p.status, 'UNKNOWN_DEPTH')
        self.assertLessEqual(p.lower, F(-1, 4))
        self.assertGreaterEqual(p.upper, F(-1, 4))
        p2 = bracket_polynomial_peak([F(-1), F(3), F(-3)], tolerance=0, max_nodes=1)
        self.assertEqual(p2.status, 'UNKNOWN_NODES')

    def test_random_exact_dyadic_probes_in_bracket(self):
        rng = random.Random(301111)
        for _ in range(128):
            poly = [F(rng.randrange(-7, 8), 8) for _ in range(5)]
            p = bracket_polynomial_peak(poly, tolerance=F(1, 256), max_depth=15,
                                        max_nodes=1000)
            self.assertLessEqual(p.lower, p.upper)
            for k in range(33):
                u = F(k, 32)
                y = sum((c * u**i for i, c in enumerate(poly)), F(0))
                self.assertLessEqual(y, p.upper)
            self.assertLessEqual(p.overestimation_lower, p.overestimation_upper)

    def test_analyze_fixture_is_model_only(self):
        content = (ROOT / 'fixtures/boundary_multispan_analytic.json').read_bytes()
        r = analyze(content, tolerance=F(1, 10000))
        self.assertFalse(r['campaign_eligible'])
        self.assertEqual(r['runtime_latency'], 'NOT_MEASURED')
        self.assertEqual(r['residual_count'], 2 * (1 + 4))
        self.assertEqual(r['boundary_status'], 'PASS')
        self.assertTrue(all(z['sampled'] is False for z in r['residual_records']))
        self.assertEqual(len(r['source_sha256']), 64)

    def test_invalid_basis_rejected(self):
        payload = {'schema_version': 1, 'basis': 'quantized_samples'}
        with self.assertRaises(ValueError):
            analyze(json.dumps(payload).encode())

    def test_bad_budget_rejected(self):
        for kwargs in ({'max_depth': -1}, {'max_nodes': 0}, {'tolerance': F(-1)}):
            with self.assertRaises(ValueError):
                bracket_polynomial_peak([F(1)], **kwargs)

    def test_serialization_preserves_exact_fraction(self):
        p = bracket_polynomial_peak([F(-1), F(3), F(-3)], tolerance=F(1, 1024))
        d = diagnostic_dict(p)
        self.assertEqual(F(d['peak_lower']), F(-1, 4))
        self.assertEqual(d['rigorous_for'], 'EXACT_SERIALIZED_POLYNOMIAL')


if __name__ == '__main__':
    unittest.main()

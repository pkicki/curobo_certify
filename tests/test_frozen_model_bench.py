import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from bench_frozen_model_cpu import benchmark_model, exact_reference_hook, get_hook


class FrozenBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.blob = (ROOT / 'fixtures/boundary_multispan_analytic.json').read_bytes()

    def test_reference_hook_replay(self):
        result = exact_reference_hook(self.blob)
        self.assertEqual(result['status'], 'CERTIFIED_K3_REFERENCE')
        self.assertTrue(result['proof_replay'])

    def test_repeats_are_fresh_and_labelled(self):
        out = benchmark_model(self.blob, {'A': exact_reference_hook, 'B': exact_reference_hook},
                              warmups=1, repeats=3, seed=11)
        self.assertEqual(len(out['raw_records']), 6)
        self.assertFalse(out['campaign_eligible'])
        self.assertFalse(out['native_linked'])
        self.assertEqual(out['benchmark_scope'], 'EXPLORATORY_FROZEN_MODEL_CPU_ONLY')
        self.assertEqual([e['hook'] for e in out['raw_records']], ['A','B','B','A','A','B'])
        self.assertTrue(all(e['elapsed_ns'] >= 0 for e in out['raw_records']))

    def test_bad_result_refused(self):
        with self.assertRaises(TypeError):
            benchmark_model(self.blob, {'bad': lambda _: None}, 0, 1, 1)
        with self.assertRaises(ValueError):
            benchmark_model(self.blob, {}, 0, 1, 1)

    def test_bad_model_refused(self):
        with self.assertRaises(Exception):
            exact_reference_hook(b'[]')

    def test_hook_spec_errors(self):
        with self.assertRaises(ValueError):
            get_hook('no_such_hook')


if __name__ == '__main__':
    unittest.main()

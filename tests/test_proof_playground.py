"""No-GPU checks for offline proof playground and analytic fixture export."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from export_playground_residuals import export

class ProofPlaygroundTests(unittest.TestCase):
    def test_builder_reproducible_without_external_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'lab.html'
            subprocess.run([sys.executable,str(ROOT/'scripts/build_proof_playground.py'),
                            '--out',str(out)],check=True,capture_output=True)
            built=out.read_text(encoding='utf-8')
            committed=(ROOT/'reports/bernstein_proof_lab.html').read_text(encoding='utf-8')
            self.assertEqual(built,committed)
            self.assertEqual(built.count('<script>'),2)
            self.assertNotIn('/* __BERNSTEIN_CORE__ */',built)
            self.assertNotIn('/* __BERNSTEIN_APP__ */',built)
    def test_analytic_fixture_exports_only_mathematical_residuals(self):
        model=json.loads((ROOT/'fixtures/boundary_multispan_analytic.json').read_text())
        data=export(model,span_index=0,joint=0)
        self.assertEqual(data['native_parity'],'NOT_EVALUATED')
        self.assertTrue(data['scope'].startswith('ANALYTIC_EXACT_'))
        self.assertGreaterEqual(len(data['examples']),4)
        for row in data['examples']:
            self.assertEqual(row['source_span'],0)
            self.assertTrue(row['coefficients'])
    def test_strict_positivity_label(self):
        model=json.loads((ROOT/'fixtures/boundary_multispan_analytic.json').read_text())
        data=export(model,span_index=0,family='positive_x')
        self.assertEqual(len(data['examples']),1)
        self.assertTrue(data['examples'][0]['strict'])
    def test_oversized_unfiltered_export_fails_closed(self):
        model=json.loads((ROOT/'fixtures/boundary_multispan_analytic.json').read_text())
        with self.assertRaises(ValueError):
            export(model,max_examples=1)
    def test_no_matching_selection_fails_closed(self):
        model=json.loads((ROOT/'fixtures/boundary_multispan_analytic.json').read_text())
        with self.assertRaises(ValueError):
            export(model,span_index=10000)

if __name__=='__main__':unittest.main()

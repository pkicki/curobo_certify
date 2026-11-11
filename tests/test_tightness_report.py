import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from explore_k3_models import analyze
from render_tightness_html import build_report


class ReportTests(unittest.TestCase):
    def test_offline_embeds_exact_model_only(self):
        raw = (ROOT / 'fixtures/boundary_multispan_analytic.json').read_bytes()
        with tempfile.TemporaryDirectory() as p:
            source = Path(p) / 'analysis.json'
            source.write_text(json.dumps(analyze(raw)))
            out = Path(p) / 'report.html'
            result = build_report([source], out)
            self.assertEqual(result, out.read_text())
            self.assertIn('margin_lower', result)
            self.assertIn('Exploratory mathematical-model analysis only', result)
            self.assertIn('model_data', result)
            self.assertNotIn('https://', result)
            self.assertNotIn('__DATA__', result)

    def test_rejects_wrong_scope(self):
        with tempfile.TemporaryDirectory() as p:
            source = Path(p) / 'bad.json'
            source.write_text('{"scope":"SAMPLED_ONLY", "campaign_eligible":false,"residual_records":[]}')
            with self.assertRaises(ValueError):
                build_report([source], Path(p) / 'out.html')


if __name__ == '__main__':
    unittest.main()

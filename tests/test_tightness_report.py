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

    def test_embeds_only_labelled_exploratory_cpu_timing(self):
        raw = (ROOT / 'fixtures/boundary_multispan_analytic.json').read_bytes()
        with tempfile.TemporaryDirectory() as p:
            source = Path(p) / 'analysis.json'
            source.write_text(json.dumps(analyze(raw)))
            bench = Path(p) / 'bench.json'
            bench.write_text(json.dumps({'latency_scope':'CPU_EXPLORATORY_FROZEN_SERIALIZED_MODEL',
                'campaign_grade':False,'results':[{'source_path':'pilot.npz','warmups_per_hook':50,
                'repetitions_per_hook':10,'hook_latency_summary_ns':{'outward':{'p50_ns':1,'p95_ns':2,'p99_ns':2,'min_ns':1,'max_ns':3}},
                'raw_records':[{'status':'CERTIFIED_K3','proof_replay':'PASS'}]}]}))
            result = build_report([source], Path(p) / 'out.html', bench)
            self.assertIn('EXPLORATORY_FROZEN_MODEL_CPU_ONLY', result)
            self.assertIn('proof_replay', result)
            self.assertIn('cpu_plot', result)
            self.assertIn('renderCPU', result)
            bench.write_text('{"latency_scope":"COMPLETE_REQUEST", "campaign_grade":false}')
            with self.assertRaises(ValueError):
                build_report([source], Path(p) / 'bad.html', bench)

    def test_embeds_intrusive_device_trace_without_promoting_scope(self):
        raw = (ROOT / 'fixtures/boundary_multispan_analytic.json').read_bytes()
        with tempfile.TemporaryDirectory() as p:
            source = Path(p) / 'analysis.json'
            source.write_text(json.dumps(analyze(raw)))
            trace = Path(p) / 'device.json'
            trace.write_text(json.dumps({'scope':'INTRUSIVE_DEVICE_TRACE_DIAGNOSTIC_NOT_CERTIFICATION',
                'records':[{'path_id':'analytic','physical_time_s':0.1,'internal_progress_device':0.2,
                            'joint':0,'jerk_native_pristine':1.0}]}))
            result = build_report([source], Path(p) / 'out.html', device_analysis=trace)
            self.assertIn('Native CUDA jerk operand trace', result)
            self.assertIn('INTRUSIVE_DEVICE_TRACE_DIAGNOSTIC_NOT_CERTIFICATION', result)
            self.assertIn('device_operand_fma_replay', result)


if __name__ == '__main__':
    unittest.main()

"""Offline margin dashboard output must be reproducible and injection safe."""
import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_margin_report import build  # noqa: E402


class MarginDashboardTests(unittest.TestCase):
    def test_empty_dashboard_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"report.html"
            build(out)
            expected=(ROOT/"reports/margin_certifiability_dashboard.html").read_text()
            self.assertEqual(out.read_text(),expected)
            self.assertIn("window.MARGIN_ANALYSIS = null;",expected)
            self.assertEqual(expected.count("<script>"),2)
            self.assertNotIn("/*__EMBED_DATA__*/",expected)

    def test_rejects_incompatible_scopes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"input.json"
            source.write_text(json.dumps({"schema_version":1,"scope":"FAKE_GPU_RESULT"}))
            with self.assertRaises(ValueError):
                build(Path(tmp)/"output.html",source)

    def test_embedded_html_escapes_script_breakouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"input.json"
            source.write_text(json.dumps({
                "schema_version":1,"scope":"EXPLORATORY_MATCHED_MARGIN_TEST",
                "rows":[],"summary_by_margin":{},"paired_to_0p95":{},
                "denominators":{"source_queries":0},"malicious":"</script><img src=x onerror=alert(1)>"}))
            result=Path(tmp)/"safe.html"
            build(result,source)
            rendered=result.read_text()
            self.assertEqual(rendered.count("</script>"),2)
            self.assertNotIn("<img src=x",rendered)
            self.assertIn("\\u003c",rendered)


if __name__=="__main__":
    unittest.main()

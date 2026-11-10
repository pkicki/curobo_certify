import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkGateTests(unittest.TestCase):
    def test_frozen_generic_manifest_still_cannot_start_campaign_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"frozen": True, "records": [{"path_id": "placeholder"}]}))
            output = root / "measurements"
            result = subprocess.run(
                [sys.executable, str(ROOT / "benchmarks/run_certification.py"),
                 "--manifest", str(manifest), "--out", str(output)],
                text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no compatible frozen Panda/UR5 TimeOpt cohort", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

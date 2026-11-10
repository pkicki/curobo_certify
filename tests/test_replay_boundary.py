import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from replay_boundary import replay


class ReplayTests(unittest.TestCase):
    def test_proof_scope_and_immutable_digest(self):
        payload = {"schema_version": 1, "basis": "local_power_ascending_u",
                   "spans": [{"left": "0", "right": "1", "q": [[0, 1]], "x": [1]}],
                   "targets": {"q_start": [0], "q_end": [1], "x_start": 1, "x_end": 1}}
        raw = json.dumps(payload).encode()
        old = raw[:]
        result = replay(payload, hashlib.sha256(raw).hexdigest())
        self.assertEqual(result["status"], "BOUNDARY_PASS")
        self.assertEqual(result["full_k3_status"], "NOT_EVALUATED")
        self.assertEqual(raw, old)

    def test_cli_json_array_fails_closed(self):
        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / "bad.json"
            source.write_text("[]")
            cmd = [sys.executable, str(ROOT / "scripts/replay_boundary.py"), str(source)]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "BOUNDARY_INVALID_INPUT")

    def test_cli_invalid_basis_fails_closed(self):
        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / "bad.json"
            source.write_text('{"schema_version":1,"basis":"unknown","spans":[],"targets":{}}')
            cmd = [sys.executable, str(ROOT / "scripts/replay_boundary.py"), str(source)]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["status"], "BOUNDARY_INVALID_INPUT")


if __name__ == "__main__":
    unittest.main()

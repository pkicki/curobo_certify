import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from replay_k3_reference import replay


class K3ReplayTests(unittest.TestCase):
    def test_analytic_certificate_replays(self):
        payload = {'schema_version': 1, 'basis': 'local_power_ascending_u',
                   'spans': [{'left': 0, 'right': 1, 'q': [[0, 1]], 'x': [1]}],
                   'targets': {'q_start': [0], 'q_end': [1], 'x_start': 1, 'x_end': 1},
                   'limits': {'velocity': [3], 'acceleration': [5], 'jerk': [5]}}
        r = replay(payload, json.dumps(payload).encode())
        self.assertEqual(r['status'], 'CERTIFIED_K3_REFERENCE')
        self.assertTrue(r['proof_replay_pass'])
        self.assertEqual(r['native_parity'], 'NOT_EVALUATED')
        self.assertEqual(r['full_contract_status'], 'UNSUPPORTED_DYNAMICS')

    def test_invalid_input_exits_two(self):
        with tempfile.TemporaryDirectory() as p:
            path = Path(p) / 'bad.json'
            path.write_text('[]')
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/replay_k3_reference.py'),
                                     str(path)], text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)['status'], 'INVALID_INPUT_OR_REPLAY')


if __name__ == '__main__':
    unittest.main()

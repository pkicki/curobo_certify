import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from curobo_certify.frozen_npz_cpu_hook import certify_and_replay_npz


class FrozenNpzHookTests(unittest.TestCase):
    def test_rejects_missing_native_fields(self):
        stream = io.BytesIO()
        np.savez(stream, x=np.array([1], dtype=np.float32))
        with self.assertRaises(KeyError):
            certify_and_replay_npz(stream.getvalue())


if __name__ == "__main__":
    unittest.main()

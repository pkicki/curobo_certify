"""GPU fault-injection tests of the v2 exhaustive time-law audit kernel.

Run in the pinned environment, e.g.
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src conda run -n curobo-timeopt-c301cb \
      python -m unittest tests.test_t1_audit_v2_gpu -v
Skipped automatically without torch/cuRobo/CUDA or the frozen real NPZ.
"""
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
NPZ = ROOT / "runs/margins/native/margin-20261117-0000-margin_0p950.npz"

try:
    import torch
    import curobo  # noqa: F401
    HAVE = torch.cuda.is_available() and NPZ.exists()
except Exception:  # pragma: no cover - CPU-only environment
    HAVE = False

INFKEY = 0x7F800000


@unittest.skipUnless(HAVE, "needs CUDA, pinned cuRobo and the frozen real NPZ")
class T1AuditV2GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from curobo_certify import time_law_grid as TG
        from native_same_s_falsify import native_inputs
        import native_time_audit_v2 as V2
        cls.V2 = V2
        cls.dev = torch.device("cuda:0")
        cls.A = V2.AuditV2(torch, cls.dev)
        with np.load(NPZ) as z:
            cls.arr = {k: z[k].copy() for k in z.files}
        cls.grid = TG.build_grid(cls.arr["timing_controls"], 16)
        cls.inp = native_inputs(cls.arr, torch, cls.dev)

    def run_kernel(self, start, count, s_ref=None, tables=None):
        g = self.grid
        tab = tables or {"T_lo": g.T_lo, "T_hi": g.T_hi, "fmin": g.fmin, "fmax": g.fmax}
        tab = {k: torch.as_tensor(np.ascontiguousarray(v), device=self.dev) for k, v in tab.items()}
        keys = [torch.zeros(1, dtype=torch.int64, device=self.dev) for _ in range(3)]
        cnt = torch.zeros(5, dtype=torch.int64, device=self.dev)
        inp = self.inp
        self.A.launch("inverse_audit_v2", count, inp["polynomial"], inp["timing"], inp["nodes"], inp["weights"],
                      inp["time"], np.int32(inp["intervals"]), np.uint32(start), np.uint32(count),
                      s_ref if s_ref is not None else np.uint64(0), tab["T_lo"], tab["T_hi"], tab["fmin"], tab["fmax"],
                      np.int32(16), *keys, cnt)
        torch.cuda.synchronize(self.dev)
        return [int(k.item()) & 0xFFFFFFFFFFFFFFFF for k in keys], [int(c) for c in cnt.cpu().tolist()]

    def test_processed_count_for_ragged_chunk(self):
        start = struct.unpack("<I", struct.pack("<f", 0.1))[0]
        keys, cnt = self.run_kernel(start, 1_000_003)
        self.assertEqual(cnt, [1_000_003, 0, 0, 0, 0])
        self.assertLess(keys[0] >> 32, INFKEY)

    def test_subnormal_and_zero_times(self):
        keys, cnt = self.run_kernel(0, 0x00800000 + 5)  # +0, all positive subnormals, first normals
        self.assertEqual(cnt, [0x00800005, 0, 0, 0, 0])
        e = struct.unpack("<f", struct.pack("<I", keys[0] >> 32))[0]
        self.assertTrue(np.isfinite(e) and e < 1e-6)

    def test_reference_progress_mismatch_fails_closed(self):
        n = 4096
        start = struct.unpack("<I", struct.pack("<f", 0.2))[0]
        bad = torch.full((n,), float("nan"), dtype=torch.float32, device=self.dev)
        keys, cnt = self.run_kernel(start, n, s_ref=bad)
        self.assertEqual(cnt[0], n)
        self.assertEqual(cnt[3], n)
        self.assertEqual(keys[0] >> 32, INFKEY)

    def test_empty_bracket_fails_closed(self):
        g = self.grid
        tables = {"T_lo": g.T_lo, "T_hi": g.T_hi, "fmin": g.fmin, "fmax": g.fmax * 0 + 1e-300}  # absurd slope
        tables["T_lo"] = g.T_lo.copy()
        tables["T_lo"][1:] += 1.0   # T_lo far above truth -> hi bracket collapses below lo
        start = struct.unpack("<I", struct.pack("<f", 0.2))[0]
        keys, cnt = self.run_kernel(start, 4096, tables=tables)
        self.assertGreater(cnt[2], 0)
        self.assertEqual(keys[0] >> 32, INFKEY)

    def test_bit_exact_time_conversion_used(self):
        # smallest subnormal time: exact binary64 value 2^-149 > 0 so a bracket is computed
        keys, cnt = self.run_kernel(1, 1)
        self.assertEqual(cnt, [1, 0, 0, 0, 0])

    def test_far_end_bound_regression_U1(self):
        """R0 finding U1: the bound must be max(s-lo, hi-s) (far end), never the distance to the bracket."""
        from curobo_certify import time_law_grid as TG
        import native_time_audit_v2 as V2
        for tval in (0.05, 0.1234, 0.2):
            b = struct.unpack("<I", struct.pack("<f", tval))[0]
            keys, cnt = self.run_kernel(b, 1)
            self.assertEqual(cnt, [1, 0, 0, 0, 0])
            e = struct.unpack("<f", struct.pack("<I", keys[0] >> 32))[0]
            s = float(V2._harness_progress(self.A, self.inp, torch.tensor([tval], dtype=torch.float32, device=self.dev))[0])
            lo, hi = TG.inverse_bracket(self.grid, float(np.float32(tval)))
            far = max(s - lo, hi - s)
            self.assertGreater(far, 0.0)
            self.assertGreaterEqual(e, far * (1 - 1e-12))  # GPU directed rounding vs CPU nextafter


if __name__ == "__main__":
    unittest.main()

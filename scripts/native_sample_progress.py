#!/usr/bin/env python3
"""Recover the true native internal progress s for saved dense native samples.

The saved ``diagnostic_*_progress`` arrays are the kernel's *sample_progress*
output ((coordinate+start_offset-1)/(knots+2), clamped), which is not the
internal s for planner outputs with a reference start span. This runs the
bitwise-reproducing harness (``progress_eval``) at the saved requested times,
checks bitwise equality of the reproduced states with the saved production
states, and stores s. Used for plotting native samples (SAMPLED_DIAGNOSTIC).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from native_same_s_falsify import Harness, native_inputs, run_progress  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    import torch
    h = Harness(torch.device("cuda:0"))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    bad = 0
    for p in a.npz:
        with np.load(p, allow_pickle=False) as z:
            arr = {k: z[k].copy() for k in z.files}
        inp = native_inputs(arr, torch, torch.device("cuda:0"))
        out, s = run_progress(h, inp, arr["diagnostic_32768_times"])
        mism = int(np.sum(out.view(np.uint32) != arr["diagnostic_32768_states"].astype(np.float32).view(np.uint32)))
        bad += mism
        np.savez(a.out_dir / f"{p.stem}.npz", s=s, bit_mismatches=np.int64(mism),
                 source_npz_sha256=np.bytes_(hashlib.sha256(p.read_bytes()).hexdigest()))
        print(p.stem, "mismatches", mism, flush=True)
    print("total mismatches", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

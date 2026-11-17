#!/usr/bin/env python3
"""Re-enumerate audited paths with the reviewer's corrected reduction (far-end distance).

    CUDA_VISIBLE_DEVICES=1 conda run -n curobo-timeopt-c301cb python \
        independent_checks/t1/corrected_eps_run.py --first 0 --count 56 --out <json>

Uses the reviewed CPU tables (time_law_grid.build_grid, 2^22 cells, whose containment is
checked independently by run_review.py) and the same compiled upstream progress_at_time
(same NVRTC path/options as scripts/native_time_inversion_audit.py). Reports per path:
corrected eps (upper bound of |s_native - s*|), the legacy reviewed quantity (must equal the
audited eps), the maximum bracket width, and processed/empty/non-finite counters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
KERNEL = Path(__file__).resolve().parent / "corrected_inverse_audit.cuh"


def unpack(v):
    v = int(v) & 0xFFFFFFFFFFFFFFFF
    return struct.unpack("<f", struct.pack("<I", v >> 32))[0], struct.unpack("<f", struct.pack("<I", v & 0xFFFFFFFF))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-json", type=Path, default=ROOT / "runs/margins/time_inversion_audit_224paths.json")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=224)
    ap.add_argument("--chunk", type=int, default=1 << 26)
    ap.add_argument("--log2-cells", type=int, default=22)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    import native_same_s_falsify as NS
    from native_same_s_falsify import Harness, native_inputs
    from curobo_certify import time_law_grid as TG
    from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import TrajectoryKernelCfg
    dev = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(dev)
    assert "3090" in gpu, gpu
    h = Harness(dev)
    cfg = TrajectoryKernelCfg()
    sources = [cfg.kernel_dir / "timeopt" / "timeopt_common.cuh", cfg.kernel_dir / "timeopt" / "timeopt_path_kernel.cuh",
               NS.HARNESS, KERNEL]
    name = "curobo_certify_review::corrected_inverse_audit"
    h.kernels["corr"] = h.cache.get_or_compile_kernel(source_files=sources, kernel_name=name,
                                                      include_dirs=cfg.get_include_dirs(), compile_flags=h.flags)
    h.qualified["corr"] = name
    audit = json.loads(a.audit_json.read_text())
    npzmap = {}
    for f in ROOT.glob("runs/**/*.npz"):
        npzmap.setdefault(f.stem, []).append(f)
    rows = []
    for r in audit["rows"][a.first:a.first + a.count]:
        t0 = time.time()
        f = [p for p in npzmap[r["path"]] if hashlib.sha256(p.read_bytes()).hexdigest() == r["source_npz_sha256"]][0]
        with np.load(f, allow_pickle=False) as z:
            arr = {k: z[k].copy() for k in z.files}
        grid = TG.build_grid(arr["timing_controls"], a.log2_cells)
        inp = native_inputs(arr, torch, dev)
        tab = {k: torch.as_tensor(np.ascontiguousarray(v), device=dev) for k, v in
               (("T_lo", grid.T_lo), ("T_hi", grid.T_hi), ("fmin", grid.fmin), ("fmax", grid.fmax))}
        last = struct.unpack("<I", struct.pack("<f", float(arr["native_timing_time_grid"][-1])))[0]
        total = last + 1
        res = {k: torch.zeros(1, dtype=torch.int64, device=dev) for k in ("far", "legacy", "width")}
        cnt_t = torch.zeros(3, dtype=torch.int64, device=dev)
        start = 0
        while start < total:
            cnt = min(a.chunk, total - start)
            h.launch("corr", cnt, inp["polynomial"], inp["timing"], inp["nodes"], inp["weights"], inp["time"],
                     np.int32(inp["intervals"]), np.uint32(start), np.uint32(cnt), tab["T_lo"], tab["T_hi"],
                     tab["fmin"], tab["fmax"], np.int32(a.log2_cells), res["far"], res["legacy"], res["width"], cnt_t)
            start += cnt
        far, far_t = unpack(res["far"].item())
        leg, leg_t = unpack(res["legacy"].item())
        wid, wid_t = unpack(res["width"].item())
        c = [int(v) for v in cnt_t.cpu().numpy()]
        row = {"path": r["path"], "times": total, "processed": c[0], "processed_equals_expected": c[0] == total,
               "empty_brackets": c[1], "nonfinite": c[2],
               "corrected_eps": far, "corrected_eps_at": far_t, "legacy_eps": leg, "legacy_at": leg_t,
               "audited_eps": r["eps_s_certified_max"], "legacy_equals_audited": leg == r["eps_s_certified_max"],
               "max_bracket_width": wid, "corrected_minus_audited": far - r["eps_s_certified_max"],
               "elapsed_s": time.time() - t0}
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = {"scope": "R0_REVIEW_CORRECTED_REDUCTION_NOT_A_REGISTERED_RERUN", "gpu": gpu, "kernel_sha256":
           hashlib.sha256(KERNEL.read_bytes()).hexdigest(), "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

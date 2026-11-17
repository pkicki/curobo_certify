#!/usr/bin/env python3
"""R0 round-2 GPU probes of T1 v2 (RTX 3090 only; short jobs).

    CUDA_VISIBLE_DEVICES=1 conda run -n curobo-timeopt-c301cb python independent_checks/t1/round2_probe.py --out <json>

Q1 exact_f64 (time_inverse_audit_v2.cuh, compiled with the cuRobo cache options incl. FTZ, from
   the same source list) vs the hardware float->double conversion compiled WITHOUT ftz, for EVERY
   bit pattern 0x00000000..0x7F7FFFFF (all nonnegative finite float32); mismatches counted.
Q2 key_ru on chosen doubles (0, 1e-45, 1e-40, 2^-126-, 2^-126, 1e-7, +Inf): resulting float32 bits.
Q3 production sample() at -0.0 vs +0.0 (all 4*dof words, bitwise) on two real paths.
(Q4, the beyond-duration bit range, is the separate CPU script round2_beyond_window_check.py.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
AUDIT = ROOT / "src/curobo_certify/native/time_inverse_audit_v2.cuh"

PROBE_FTZ = r'''
namespace r2probe {
__global__ void exact_dump(unsigned int start, unsigned int count, double *out) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = curobo_certify_harness_v2::exact_f64(start + i);
}
__global__ void key_dump(const double *e, unsigned long long *out, int n) {
  const int i = threadIdx.x;
  if (i < n) out[i] = curobo_certify_harness_v2::key_ru(e[i], 0u);
}
}
'''
PROBE_NOFTZ = r'''
extern "C" __global__ void compare(unsigned int start, unsigned int count, const double *got,
                                   unsigned long long *mism, unsigned long long *first) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const double ref = (double)__uint_as_float(start + i);   // no FTZ in this module
  if (__double_as_longlong(ref) != __double_as_longlong(got[i])) {
    atomicAdd(mism, 1ULL);
    atomicMin(first, (unsigned long long)(start + i));
  }
}
'''


def f32bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def bits2f(b):
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    from cuda.core import Program, ProgramOptions, LaunchConfig, launch
    from cuda import pathfinder
    from curobo._src.context import get_runtime
    from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import TrajectoryKernelCfg
    import native_same_s_falsify as NS
    dev = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(dev)
    assert "3090" in gpu, gpu
    cache = get_runtime().get_cuda_core_cache()
    cache.initialize()
    cfg = TrajectoryKernelCfg()
    sources = [cfg.kernel_dir / "timeopt" / "timeopt_common.cuh", cfg.kernel_dir / "timeopt" / "timeopt_path_kernel.cuh",
               NS.HARNESS, AUDIT]
    code = cache._read_sources(sources, cfg.get_include_dirs()) + PROBE_FTZ
    inc = [str(i) for i in cfg.get_include_dirs()]
    for lib in ("cudart", "nvrtc"):
        d = pathfinder.find_nvidia_header_directory(lib)
        if d:
            inc.append(d)
    kw = dict(std="c++17", arch=cache.arch, include_path=inc, ftz=True, fma=True, prec_div=False, prec_sqrt=False,
              lineinfo=True, device_code_optimize=True)
    names = ("r2probe::exact_dump", "r2probe::key_dump")
    mod = Program(code, code_type="c++", options=ProgramOptions(**kw)).compile("cubin", name_expressions=names)
    k_dump, k_key = mod.get_kernel(names[0]), mod.get_kernel(names[1])
    k_cmp = Program(PROBE_NOFTZ, code_type="c++", options=ProgramOptions(std="c++17", arch=cache.arch, ftz=False)
                    ).compile("cubin").get_kernel("compare")
    stream = cache.get_stream_wrapper(torch.cuda.current_stream(dev))
    rep = {"gpu": gpu, "audit_v2_sha256": hashlib.sha256(AUDIT.read_bytes()).hexdigest()}
    # Q1
    chunk = 1 << 24
    buf = torch.empty(chunk, dtype=torch.float64, device=dev)
    mism = torch.zeros(1, dtype=torch.int64, device=dev)
    first = torch.full((1,), -1, dtype=torch.int64, device=dev)
    start, end, total = 0, 0x7F7FFFFF, 0
    while start <= end:
        cnt = min(chunk, end - start + 1)
        cfgl = LaunchConfig(grid=(cnt + 255) // 256, block=256)
        launch(stream, cfgl, k_dump, np.uint32(start), np.uint32(cnt), buf.data_ptr())
        launch(stream, cfgl, k_cmp, np.uint32(start), np.uint32(cnt), buf.data_ptr(), mism.data_ptr(), first.data_ptr())
        start += cnt
        total += cnt
    torch.cuda.synchronize(dev)
    rep["Q1_exact_f64"] = {"patterns": total, "expected": 0x7F800000, "mismatches": int(mism.item()),
                           "first_mismatch_bits": int(first.item()) & 0xFFFFFFFFFFFFFFFF}
    # Q2
    vals = [0.0, 1e-45, 1e-40, float(np.nextafter(2.0 ** -126, 0)), 2.0 ** -126, 1e-7, float("inf")]
    e = torch.tensor(vals, dtype=torch.float64, device=dev)
    o = torch.zeros(len(vals), dtype=torch.int64, device=dev)
    launch(stream, LaunchConfig(grid=1, block=32), k_key, e.data_ptr(), o.data_ptr(), np.int32(len(vals)))
    torch.cuda.synchronize(dev)
    rep["Q2_key_ru"] = [{"e": v, "f32_bits": (int(x) & 0xFFFFFFFFFFFFFFFF) >> 32,
                         "f32": bits2f((int(x) & 0xFFFFFFFFFFFFFFFF) >> 32),
                         "upper_bound": bits2f((int(x) & 0xFFFFFFFFFFFFFFFF) >> 32) >= v}
                        for v, x in zip(vals, o.cpu().tolist())]
    # Q3 -0 vs +0 on production
    import native_time_audit_v2 as V2
    from native_same_s_falsify import native_inputs
    A = V2.AuditV2(torch, dev)
    q3 = []
    for name in ("runs/margins/native/margin-20261117-0000-margin_0p900.npz",
                 "runs/margins/native/margin-20261117-0017-margin_0p990.npz"):
        with np.load(ROOT / name) as z:
            arr = {k: z[k].copy() for k in z.files}
        inp = native_inputs(arr, torch, dev)
        ts = torch.tensor([0.0, -0.0, -1e-30], dtype=torch.float32, device=dev)
        out, prog = A.production(inp, ts)
        torch.cuda.synchronize(dev)
        w = out[:, 0].cpu().numpy().view(np.uint32)
        q3.append({"path": Path(name).stem, "minus0_eq_plus0_bitwise": bool(np.array_equal(w[:, 0], w[:, 1])),
                   "plus0_vaj_all_zero": bool(np.all(out[1:, 0, 0].cpu().numpy() == 0)),
                   "plus0_jerk": out[3, 0, 0].cpu().tolist(),
                   "neg_vaj_all_zero": bool(np.all(out[1:, 0, 2].cpu().numpy() == 0))})
    rep["Q3_signed_zero_production"] = q3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

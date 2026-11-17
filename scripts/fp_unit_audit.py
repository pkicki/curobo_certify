#!/usr/bin/env python3
"""Exhaustive accuracy audit of the approximate FP32 units used by the native kernels.

With cuRobo's NVRTC options (--ftz --fmad --prec-div=false --prec-sqrt=false) the
native '/', sqrtf and rsqrtf compile to MUFU.RCP-based division, MUFU.SQRT and
MUFU.RSQ (see scripts/sass_audit.py). This enumerates EVERY positive normal float32
input y and measures, against binary64 references (exact enough: the binary64
reference error is < 2^-52 relative), the maximum error in units of the float32
ulp of the exact result for:  1.0f / y,  sqrtf(y),  rsqrtf(y),  and (1.5f*y)/6.0f
style constant division (x / 6.0f).
Result: an exhaustive, hardware-observed replacement for the documented bounds
A3 (division) and A4 (sqrtf) for the reciprocal / constant-division / sqrt cases.
General x/y with both operands varying is not enumerable; it is RCP + MUL (+FMA)
and its bound follows from the measured RCP error plus one rounding.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = r'''
__device__ __forceinline__ double ulp32(double v) {  // ulp of float32 nearest to |v| (normal range)
  v = fabs(v); int e; frexp(v, &e); return ldexp(1.0, e - 24);
}
extern "C" __global__ void unary_audit(unsigned int start, unsigned int count, unsigned long long *out) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const unsigned int bits = start + i;
  const float y = __uint_as_float(bits);
  const double yd = (double)y;
  const float r = 1.0f / y, s = sqrtf(y), q = rsqrtf(y), d6 = y / 6.0f;
  const double er = 1.0 / yd, es = sqrt(yd), eq = 1.0 / sqrt(yd), e6 = yd / 6.0;
  const double vals[4] = {fabs((double)r - er) / ulp32(er), fabs((double)s - es) / ulp32(es),
                          fabs((double)q - eq) / ulp32(eq), fabs((double)d6 - e6) / ulp32(e6)};
  const double ex[4] = {er, es, eq, e6};
  for (int k = 0; k < 4; ++k) {
    // skip results that leave the float32 normal range (FTZ / overflow are separate, documented cases)
    if (!(fabs(ex[k]) >= 1.1754943508222875e-38 && fabs(ex[k]) <= 3.4028234663852886e38)) continue;
    const float f = __double2float_ru(vals[k]);
    atomicMax(&out[k], ((unsigned long long)__float_as_uint(f) << 32) | bits);
  }
}
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import struct
    import torch
    from cuda.core import Program, ProgramOptions, LaunchConfig, launch
    from curobo._src.context import get_runtime
    cache = get_runtime().get_cuda_core_cache()
    cache.initialize()
    opts = ProgramOptions(std="c++17", arch=cache.arch, ftz=True, fma=True, prec_div=False, prec_sqrt=False,
                          device_code_optimize=True)
    kern = Program(SRC, code_type="c++", options=opts).compile("cubin").get_kernel("unary_audit")
    dev = torch.device("cuda:0")
    out = torch.zeros(4, dtype=torch.int64, device=dev)
    stream = cache.get_stream_wrapper(torch.cuda.current_stream(dev))
    start, end = 0x00800000, 0x7F7FFFFF  # all positive normal floats
    chunk = 1 << 28
    while start <= end:
        cnt = min(chunk, end - start + 1)
        launch(stream, LaunchConfig(grid=(cnt + 255) // 256, block=256), kern, np.uint32(start), np.uint32(cnt), out.data_ptr())
        start += cnt
    torch.cuda.synchronize(dev)
    res = {}
    for k, name in enumerate(("rcp_1_div_y", "sqrtf", "rsqrtf", "div_by_6")):
        v = int(out[k].item()) & 0xFFFFFFFFFFFFFFFF
        err = struct.unpack("<f", struct.pack("<I", v >> 32))[0]
        y = struct.unpack("<f", struct.pack("<I", v & 0xFFFFFFFF))[0]
        res[name] = {"max_error_ulps": float(err), "worst_input": float(y)}
    report = {"scope": "EXHAUSTIVE_POSITIVE_NORMAL_FLOAT32_UNARY_AUDIT", "gpu": torch.cuda.get_device_name(dev),
              "arch": cache.arch, "inputs": 0x7F7FFFFF - 0x00800000 + 1, "results": res,
              "assumption_check": {"A3_division_2ulp_documented": res["rcp_1_div_y"]["max_error_ulps"] <= 2 and res["div_by_6"]["max_error_ulps"] <= 2,
                                   "A4_sqrtf_3ulp_assumed": res["sqrtf"]["max_error_ulps"] <= 3}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Binary identity + SASS audit of the T1 v2 audit kernels and the production sampler.

Compiles, with EXACTLY the ProgramOptions of the pinned cuRobo kernel cache
(kernel_cache._compile_kernel: std=c++17, arch=cache.arch, ftz, fma, prec_div=False,
prec_sqrt=False, lineinfo, device_code_optimize; the textual compile_flags list is NOT passed
to NVRTC) and the same source readers:
  * production: [timeopt_path_kernel.cuh] -> curobo::trajectory::timeopt::sample_with_progress_kernel
  * audit:      [timeopt_common.cuh, timeopt_path_kernel.cuh, same_s_harness.cuh,
                 time_inverse_audit_v2.cuh] -> inverse_audit_v2, progress_chunk_v2
Records cubin SHA-256, NVRTC/driver versions and an instruction inventory. Checks for the
inverse kernel that the directed-rounding binary64 operations appear as rounding-moded SASS
(DADD/DFMA with .RM/.RP) and that no FTZ float32->float64 conversion of inputs is used.
This is an audit of generated code (compiled twice to check determinism), not a silicon proof.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "src/curobo_certify/native/same_s_harness.cuh"
AUDIT = ROOT / "src/curobo_certify/native/time_inverse_audit_v2.cuh"


def compile_like_cache(cache, sources, include_dirs, names, arch):
    import os
    from cuda import pathfinder
    from cuda.core import Program, ProgramOptions
    from curobo._src.runtime import debug_cuda_compile
    if debug_cuda_compile:
        raise SystemExit("cuRobo debug compile is enabled; production options differ")
    code = cache._read_sources(sources, include_dirs)
    inc = [str(i) for i in include_dirs]
    for lib in ("cudart", "nvrtc"):
        d = pathfinder.find_nvidia_header_directory(lib)
        if d is not None and os.path.isdir(d):
            inc.append(d)
    kw = {"std": "c++17", "arch": arch, "include_path": inc, "ftz": True, "fma": True, "prec_div": False,
          "prec_sqrt": False, "lineinfo": True, "device_code_optimize": True}
    mod = Program(code, code_type="c++", options=ProgramOptions(**kw)).compile("cubin", name_expressions=names)
    return bytes(mod.code), {k: v for k, v in kw.items() if k != "include_path"}, hashlib.sha256(code.encode()).hexdigest()


def inventory(cubin: bytes):
    import triton
    tools = Path(triton.__file__).parent / "backends/nvidia/bin"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "k.cubin"
        p.write_bytes(cubin)
        sass = subprocess.check_output([str(tools / "nvdisasm"), "-c", str(p)], text=True)
        ver = subprocess.check_output([str(tools / "nvdisasm"), "--version"], text=True).strip().splitlines()[-1]
    funcs, cur = {}, None
    for line in sass.splitlines():
        m = re.match(r"\s*\.text\.(\S+):", line)
        if m:
            cur = m.group(1)
            funcs[cur] = collections.Counter()
            continue
        m = re.search(r"/\*[0-9a-f]+\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", line)
        if m and cur:
            funcs[cur][m.group(2)] += 1
    return funcs, ver, sass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    from cuda.bindings import driver, nvrtc
    from curobo._src.context import get_runtime
    from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import TrajectoryKernelCfg
    cache = get_runtime().get_cuda_core_cache()
    cache.initialize()
    cfg = TrajectoryKernelCfg()
    inc = cfg.get_include_dirs()
    kd = cfg.kernel_dir / "timeopt"
    builds = {
        "production_sampler": ([kd / "timeopt_path_kernel.cuh"],
                               ("curobo::trajectory::timeopt::sample_with_progress_kernel",)),
        "t1_v2_audit": ([kd / "timeopt_common.cuh", kd / "timeopt_path_kernel.cuh", HARNESS, AUDIT],
                        ("curobo_certify_harness_v2::inverse_audit_v2", "curobo_certify_harness_v2::progress_chunk_v2")),
    }
    rep = {"scope": "T1_V2_BINARY_AND_SASS_AUDIT", "gpu": torch.cuda.get_device_name(0), "arch": cache.arch,
           "nvrtc_version": list(nvrtc.nvrtcVersion()[1:]), "driver_version": driver.cuDriverGetVersion()[1],
           "torch_cuda": torch.version.cuda, "builds": {}}
    for name, (src, names) in builds.items():
        c1, opts, src_sha = compile_like_cache(cache, src, inc, names, cache.arch)
        c2, _, _ = compile_like_cache(cache, src, inc, names, cache.arch)
        funcs, ver, sass1 = inventory(c1)
        _, _, sass2 = inventory(c2)
        rep["nvdisasm"] = ver
        b = {"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src},
             "combined_source_sha256": src_sha, "nvrtc_options": opts, "kernels": list(names),
             # raw cubin bytes embed a random NVRTC temporary TU name in internal symbols, so the
             # identity used is the SHA-256 of the full nvdisasm SASS listing (checked deterministic).
             "cubin_sha256_nondeterministic": hashlib.sha256(c1).hexdigest(),
             "sass_sha256": hashlib.sha256(sass1.encode()).hexdigest(),
             "deterministic_recompile": sass1 == sass2, "functions": {}}
        for f, c in funcs.items():
            b["functions"][f] = {"total": sum(c.values()),
                                 "fp64": {k: v for k, v in sorted(c.items()) if k.split(".")[0] in ("DADD", "DFMA", "DMUL", "DSETP", "MUFU", "F2F", "I2F", "DMNMX")},
                                 "fp32_ftz": sum(v for k, v in c.items() if ".FTZ" in k)}
        rep["builds"][name] = b
    inv = next((v for k, v in rep["builds"]["t1_v2_audit"]["functions"].items() if "inverse_audit_v2" in k), None)
    fp64 = inv["fp64"] if inv else {}
    rep["checks"] = {
        "inverse_kernel_found": inv is not None,
        "directed_rounding_ops_present": any(".RM" in k or ".RP" in k for k in fp64),
        "rounding_moded_ops": {k: v for k, v in fp64.items() if ".RM" in k or ".RP" in k},
        "no_ftz_f32_to_f64_conversion": not any(k.startswith("F2F.F64.F32") and "FTZ" in k for k in fp64),
        "f2f_ops": {k: v for k, v in fp64.items() if k.startswith("F2F")},
        "deterministic": all(b["deterministic_recompile"] for b in rep["builds"].values()),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"gpu": rep["gpu"], "arch": rep["arch"], "nvrtc": rep["nvrtc_version"],
                      "sass": {k: v["sass_sha256"][:16] for k, v in rep["builds"].items()}, "checks": rep["checks"]},
                     indent=1))


if __name__ == "__main__":
    main()

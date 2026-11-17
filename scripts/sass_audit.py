#!/usr/bin/env python3
"""SASS audit of the native sample arithmetic compiled with cuRobo's NVRTC options.

Compiles the harness kernels (which inline the unmodified pinned upstream
sample_joint / timing_state / progress_at_time) to a cubin for the given GPU
architecture with exactly the options of cuRobo's kernel cache, disassembles it
with nvdisasm and inventories the floating-point instructions. Purpose: check the
arithmetic assumptions A1-A5 of docs/NATIVE_LINKAGE_20261116.md against the actual
instruction stream (FMA contraction, MUFU approximations behind '/' and sqrtf,
.FTZ modifiers). This is an audit of the generated code, not a proof about silicon.
"""
import argparse
import collections
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "src/curobo_certify/native/same_s_harness.cuh"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default=None, help="sm_86 (RTX 3090) / sm_80 (A100); default: current device")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    from cuda.core import Program, ProgramOptions
    from cuda import pathfinder
    from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import TrajectoryKernelCfg
    from curobo._src.context import get_runtime
    cache = get_runtime().get_cuda_core_cache()
    cache.initialize()
    arch = a.arch or cache.arch
    cfg = TrajectoryKernelCfg()
    sources = [cfg.kernel_dir / "timeopt" / "timeopt_common.cuh", cfg.kernel_dir / "timeopt" / "timeopt_path_kernel.cuh", HARNESS]
    code = cache._read_sources(sources, cfg.get_include_dirs())
    inc = [str(i) for i in cfg.get_include_dirs()]
    for lib in ("cudart", "nvrtc"):
        d = pathfinder.find_nvidia_header_directory(lib)
        if d:
            inc.append(d)
    opts = ProgramOptions(std="c++17", arch=arch, include_path=inc, ftz=True, fma=True, prec_div=False,
                          prec_sqrt=False, device_code_optimize=True)
    names = ("curobo_certify_harness::same_s_eval", "curobo_certify_harness::progress_eval",
             "curobo::trajectory::timeopt::sample_with_progress_kernel")
    mod = Program(code, code_type="c++", options=opts).compile("cubin", name_expressions=names)
    tools = Path(sys.modules["triton"].__file__).parent / "backends/nvidia/bin" if "triton" in sys.modules else None
    if tools is None:
        import triton
        tools = Path(triton.__file__).parent / "backends/nvidia/bin"
    with tempfile.TemporaryDirectory() as d:
        cub = Path(d) / "k.cubin"
        cub.write_bytes(bytes(mod.code))
        sass = subprocess.check_output([str(tools / "nvdisasm"), "-c", str(cub)], text=True)
    funcs = {}
    current = None
    for line in sass.splitlines():
        m = re.match(r"\s*\.text\.(\S+):", line)
        if m:
            current = m.group(1)
            funcs[current] = collections.Counter()
            continue
        m = re.search(r"/\*[0-9a-f]+\*/\s+(@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)", line)
        if m and current:
            funcs[current][m.group(2)] += 1
    report = {"scope": "SASS_INSTRUCTION_INVENTORY", "arch": arch, "gpu": torch.cuda.get_device_name(0),
              "nvrtc_options": "std=c++17 ftz fma prec_div=false prec_sqrt=false -O (cuRobo kernel cache options)",
              "functions": {}}
    for f, c in funcs.items():
        fp = {k: v for k, v in c.items() if k.split(".")[0] in ("FFMA", "FMUL", "FADD", "FMNMX", "MUFU", "FSETP", "FCHK",
                                                                  "DFMA", "DMUL", "DADD", "F2F", "I2F", "F2I", "FRND")}
        report["functions"][f] = {"fp_instructions": dict(sorted(fp.items())),
                                  "mufu": {k: v for k, v in c.items() if k.startswith("MUFU")},
                                  "ftz_modified": sum(v for k, v in c.items() if ".FTZ" in k),
                                  "ffma_total": sum(v for k, v in c.items() if k.startswith("FFMA")),
                                  "total_instructions": sum(c.values())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    for f, r in report["functions"].items():
        print(f[:60], "MUFU", r["mufu"], "FFMA", r["ffma_total"], "FTZ-modified", r["ftz_modified"], "FCHK", r["fp_instructions"].get("FCHK", 0))


if __name__ == "__main__":
    main()

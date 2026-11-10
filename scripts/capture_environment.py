#!/usr/bin/env python3
"""Capture available environment facts without implying benchmark evidence."""
from __future__ import annotations
import argparse, importlib.metadata, json, os, platform, shutil, subprocess, sys
from pathlib import Path

PIN = "c301cb00a87012c2808b1a60649c4065c9b3c95a"

def cmd(args):
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=10)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("runs/environment.json")); a = ap.parse_args()
    versions = {}
    for name in ("torch", "curobo", "numpy", "scipy", "mpmath", "gmpy2", "python-flint"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    data = {
        "evidence_kind": "environment_capture_not_benchmark",
        "upstream_required_sha": PIN,
        "git": cmd(["git", "rev-parse", "HEAD"]),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "dependencies": versions,
        "gpu": cmd(["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.gr,clocks.mem,power.limit", "--format=csv,noheader"]),
        "cuda_compiler": shutil.which("nvcc"),
        "blockers": ["PyTorch and cuRobo are not installed", "required upstream SHA not present in repository object database", "adapter parity gate not run", "no frozen campaign inputs"],
        "timing_capture_profile": "not run",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    print(a.out)
if __name__ == "__main__": main()

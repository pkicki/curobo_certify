#!/usr/bin/env python3
"""Capture exact runtime and source provenance; this is not benchmark data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PIN = "c301cb00a87012c2808b1a60649c4065c9b3c95a"


def command(argv: list[str], timeout: int = 30) -> dict:
    try:
        p = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        return {"argv": argv, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}


def imported_curobo() -> dict:
    try:
        import curobo
    except Exception as exc:
        return {"importable": False, "error": f"{type(exc).__name__}: {exc}"}
    package = Path(curobo.__file__).resolve().parent
    repo = next((p for p in package.parents if (p / ".git").exists()), None)
    if repo is None:
        return {"importable": True, "package_path": str(package), "git_revision": None,
                "error": "imported package is not inside a Git checkout"}
    return {"importable": True, "package_path": str(package), "repository": str(repo),
            "git_revision": command(["git", "-C", str(repo), "rev-parse", "HEAD"]),
            "git_status": command(["git", "-C", str(repo), "status", "--porcelain=v1"])}


def torch_state() -> dict:
    try:
        import torch
        devices = []
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            devices.append({"logical_index": index, "name": prop.name,
                            "total_memory_bytes": prop.total_memory,
                            "compute_capability": [prop.major, prop.minor],
                            "uuid": str(getattr(prop, "uuid", "unavailable"))})
        return {"version": torch.__version__, "cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(), "devices": devices}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/environment.json"))
    parser.add_argument("--parity-json", type=Path)
    args = parser.parse_args()
    names = ("numpy", "scipy", "nvidia-curobo", "cuda-core", "warp-lang", "triton", "pytest")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    pip_freeze = command([sys.executable, "-m", "pip", "freeze"], timeout=60)
    conda_packages = command(["conda", "list", "--json"], timeout=60) if shutil.which("conda") else None
    gpu = command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,pci.bus_id,clocks.gr,clocks.mem,power.limit,power.draw,temperature.gpu,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"])
    provenance = {
        "evidence_kind": "environment_provenance_not_benchmark",
        "upstream_required_sha": PIN,
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "cpu": {"processor": platform.processor(), "logical_cores": os.cpu_count(),
                "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
        "runtime": {"torch": torch_state(), "dependencies": versions,
                    "pip_freeze": pip_freeze, "conda_packages": conda_packages},
        "curobo_import": imported_curobo(),
        "gpu_snapshot": gpu,
        "cuda_compiler": command([shutil.which("nvcc"), "--version"]) if shutil.which("nvcc") else None,
        "selected_environment_settings": {key: os.environ.get(key) for key in (
            "CONDA_PREFIX", "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH",
            "TORCH_CUDA_ARCH_LIST", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_MODULE_LOADING",
        )},
        "compilation_and_graph_settings": {
            "timeopt_use_cuda_graph": True,
            "source_compile_flags": "upstream defaults; runtime compiler configuration is in pinned source",
        },
        "parity_artifact": None,
        "limitations": ["GPU clocks, power, utilization and memory are a shared-device snapshot, not a controlled benchmark."],
    }
    if args.parity_json:
        raw = args.parity_json.read_bytes()
        parity = json.loads(raw)
        provenance["parity_artifact"] = {
            "path": str(args.parity_json),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "evidence_kind": parity.get("evidence_kind"),
            "status": parity.get("status"),
            "random_interior_points": parity.get("random_interior_points"),
            "knot_side_points": parity.get("knot_side_points"),
        }
    blockers = []
    curobo = provenance["curobo_import"]
    revision_output = curobo.get("git_revision", {}).get("stdout") if isinstance(curobo.get("git_revision"), dict) else None
    if revision_output != PIN:
        blockers.append("imported cuRobo source does not verify to the required commit")
    if provenance["runtime"]["torch"].get("cuda_available") is not True:
        blockers.append("CUDA is not available")
    parity = provenance.get("parity_artifact")
    if parity is None or parity.get("status") != "PASS":
        blockers.append("native-adapter parity is not recorded as PASS")
    provenance["gate_blockers"] = blockers
    repo = Path(__file__).resolve().parents[1]
    source_files = ("pyproject.toml", "scripts/capture_environment.py",
                    "scripts/native_timeopt_parity.py", "src/curobo_certify/adapters/timeopt.py")
    provenance["campaign_repository"] = {
        "path": str(repo),
        "branch": command(["git", "-C", str(repo), "branch", "--show-current"]),
        "head": command(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "status": command(["git", "-C", str(repo), "status", "--porcelain=v1"]),
        "tracked_diff_sha256": hashlib.sha256(
            command(["git", "-C", str(repo), "diff", "--binary", "HEAD"], timeout=60)["stdout"].encode()
        ).hexdigest(),
        "source_file_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in source_files
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

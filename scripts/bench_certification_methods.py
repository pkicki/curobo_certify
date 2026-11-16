#!/usr/bin/env python3
"""Matched frozen-output certification benchmark (compiled producer + independent verifier).

Every method runs on the identical in-memory frozen float32 TimeOpt arrays of each
path. Scope: FROZEN_OUTPUT_FULL_CERTIFICATE_CPU (producer + independent verifier,
plus compiled native-bound stage for the native-reported contract). NPZ parsing is
timed separately and excluded from method latency. EXPLORATORY: these are not
formal Dev40/Holdout100/Confirm300 cohorts.

Contracts:
  physical   original physical limits (model K3)
  planning   planning-scaled limits (0.95 x physical) = the solver's own contract
  native     physical limits reduced by the compiled native FP32 error bound
Methods: H0, U1, U2, U4, U6 (uniform), A (adaptive, depth 12, 8192 nodes),
  A-nw (adaptive without early witness handling), and registered sweeps
  A-d{0,2,4,8,12,16} (8192 nodes) and A-n{2048,8192,32768} (depth 12).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify import compiled_k3 as K  # noqa: E402

METHODS = {"H0": ("uniform", 0, 1 << 22, True), "U1": ("uniform", 1, 1 << 22, True),
           "U2": ("uniform", 2, 1 << 22, True), "U4": ("uniform", 4, 1 << 22, True),
           "U6": ("uniform", 6, 1 << 22, True), "A": ("adaptive", 12, 8192, True),
           "A-nw": ("adaptive", 12, 8192, False)}
SWEEP = {**{f"A-d{d}": ("adaptive", d, 8192, True) for d in (0, 2, 4, 8, 12, 16)},
         **{f"A-n{n}": ("adaptive", 12, n, True) for n in (2048, 8192, 32768)}}


def limits_of(a, contract, planning_scale=0.95):
    lim = {n: np.minimum(np.abs(a[k][0]), np.abs(a[k][1])).astype(np.float64)
           for n, k in (("velocity", "velocity_limits"), ("acceleration", "acceleration_limits"), ("jerk", "jerk_limits"))}
    if contract == "planning":
        if not 0 < float(planning_scale) <= 1:
            raise ValueError("invalid planning scale")
        lim = {k: (v.astype(np.float32) * np.float32(planning_scale)).astype(np.float64)
               for k, v in lim.items()}
    return lim


def run_once(P, V, args, method, tighten, native_stage):
    mode, depth, nodes, wit = method
    t0 = time.perf_counter_ns()
    if native_stage is not None:
        tighten = K.native_bounds(P, args[0], args[1])
    t1 = time.perf_counter_ns()
    r = P.certify(*args, tighten=tighten, max_depth=depth, max_nodes=nodes, mode=mode, witnesses=wit)
    t2 = time.perf_counter_ns()
    ver = None
    if r["status"] == "CERTIFIED":
        dig = K.input_digest(*args, tighten)
        ver = V.verify(*args, r["leaves"], dig, tighten=tighten)
    t3 = time.perf_counter_ns()
    final = r["status"] if r["status"] != "CERTIFIED" else ("CERTIFIED" if ver["pass"] else "REJECTED_BY_VERIFIER")
    if native_stage is not None and final == "VIOLATED":
        final = "UNKNOWN_TIGHTENED_WITNESS"  # witness vs tightened limits is not a native violation
    L = r["leaves"]
    return {"final": final, "producer": r["status"], "families": r["families"], "nodes": int(r["nodes"]),
            "leaves": int(len(L)), "max_depth_used": int(L["depth"].max()) if len(L) else 0,
            "split_residuals": int(len({(int(a), int(b), int(c)) for a, b, c, d in zip(L["span"], L["family"], L["joint"], L["depth"]) if d > 0})),
            "ms_native_bound": (t1 - t0) / 1e6, "ms_producer": (t2 - t1) / 1e6, "ms_verifier": (t3 - t2) / 1e6,
            "ms_total": (t3 - t0) / 1e6}



def margin_index_from_records(query_path: Path, records_dir: Path):
    """Associate exact frozen NPZ SHA with the source query's true margin.

    No source-id substring heuristics, no benchmark-row guessing. A collision
    between different planning scales of identical NPZ bytes is a blocker.
    """
    from curobo_certify.margin_campaign import check_manifest, as_margin
    manifest = json.loads(query_path.read_text())
    check_manifest(manifest)
    entries = {}
    for query in manifest:
        rec_path = records_dir / (query["source_id"] + ".json")
        if not rec_path.exists():
            continue  # retained failed/never-run entry, cannot benchmark
        record = json.loads(rec_path.read_text())
        if record.get("status") == "FAILED_RETAINED":
            continue
        if record.get("source_query_sha256") != query["source_query_sha256"]:
            raise ValueError("source query hash mismatch for " + query["source_id"])
        if record.get("source_query") != query or record.get("planner_success") is not True:
            raise ValueError("missing matching frozen successful native source")
        sha = record.get("arrays_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("missing original NPZ hash")
        scale = float(as_margin(query["planning_margin"]))
        if sha in entries and entries[sha][0] != scale:
            raise ValueError("two different margin arms point to identical frozen NPZ bytes")
        entries[sha] = (scale, query["source_id"], query["solver_condition"])
    return entries

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--contracts", nargs="+", default=["physical", "planning", "native"])
    ap.add_argument("--warmups", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--sweeps", action="store_true")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--margin-queries", type=Path,
                    help="frozen multi-margin source query manifest for planning-scale replay")
    ap.add_argument("--margin-records-dir", type=Path,
                    help="generator output records; required with --margin-queries")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if bool(a.margin_queries) != bool(a.margin_records_dir):
        ap.error("--margin-queries and --margin-records-dir must be provided together")
    margin_index = (margin_index_from_records(a.margin_queries, a.margin_records_dir)
                    if a.margin_queries else None)
    methods = dict(METHODS, **(SWEEP if a.sweeps else {}))
    with tempfile.TemporaryDirectory() as d:
        builds = [K.build("k3_certifier.cpp", Path(d) / "c.so"), K.build("k3_verifier.cpp", Path(d) / "v.so")]
        P, V = K.Producer(Path(d) / "c.so"), K.Verifier(Path(d) / "v.so")
        rows = []
        for pi, path in enumerate(a.npz):
            t0 = time.perf_counter_ns()
            raw = path.read_bytes()
            with np.load(path, allow_pickle=False) as z:
                arr = {k: z[k].copy() for k in z.files}
            parse_ms = (time.perf_counter_ns() - t0) / 1e6
            before = hashlib.sha256(arr["effective_geometry"].tobytes() + arr["timing_controls"].tobytes()).hexdigest()
            source_sha = hashlib.sha256(raw).hexdigest()
            if margin_index is not None and source_sha not in margin_index:
                raise SystemExit("NPZ not in frozen margin source registry: " + path.stem)
            margin_entry = margin_index.get(source_sha) if margin_index is not None else None
            planning_scale = margin_entry[0] if margin_entry else 0.95
            for contract in a.contracts:
                args = (arr["effective_geometry"], arr["timing_controls"], arr["requested_start"][0],
                        arr["requested_goal"][0], limits_of(arr, contract, planning_scale))
                native = True if contract == "native" else None
                names = list(methods)
                for name in names:
                    for _ in range(a.warmups):
                        run_once(P, V, args, methods[name], None, native)
                reps = {n: [] for n in names}
                rng = np.random.default_rng(a.seed + 1000 * pi + 17 * a.contracts.index(contract))
                for r in range(a.repeats):
                    order = list(rng.permutation(names))
                    for name in order:
                        reps[name].append(run_once(P, V, args, methods[name], None, native))
                for name in names:
                    first = reps[name][0]
                    if any(x["final"] != first["final"] or x["leaves"] != first["leaves"] for x in reps[name]):
                        raise SystemExit(f"nondeterministic outcome {path.stem} {contract} {name}")
                    rows.append({"path": path.stem, "source_npz_sha256": hashlib.sha256(raw).hexdigest(),
                                 "contract": contract, "method": name, "parse_ms": parse_ms,
                                 "planning_scale": planning_scale, "source_query_id": margin_entry[1] if margin_entry else None,
                                 "solver_condition": margin_entry[2] if margin_entry else "default_margin_0p95",
                                 **{k: first[k] for k in ("final", "producer", "families", "nodes", "leaves",
                                                          "max_depth_used", "split_residuals")},
                                 "repeats_ms_total": [x["ms_total"] for x in reps[name]],
                                 "repeats_ms_producer": [x["ms_producer"] for x in reps[name]],
                                 "repeats_ms_verifier": [x["ms_verifier"] for x in reps[name]],
                                 "repeats_ms_native_bound": [x["ms_native_bound"] for x in reps[name]]})
            after = hashlib.sha256(arr["effective_geometry"].tobytes() + arr["timing_controls"].tobytes()).hexdigest()
            if before != after:
                raise SystemExit("input mutated")
            print(f"{path.stem}: done", flush=True)
    out = {"scope": "FROZEN_OUTPUT_FULL_CERTIFICATE_CPU_EXPLORATORY", "builds": builds,
           "cpu": platform.processor() or platform.machine(), "platform": platform.platform(),
           "affinity": sorted(os.sched_getaffinity(0)), "warmups": a.warmups, "repeats": a.repeats,
           "order": "per-repeat seeded random permutation of methods", "methods": {k: list(v) for k, v in methods.items()},
           "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()

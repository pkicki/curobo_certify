#!/usr/bin/env python3
"""R0 CPU review run over the audited real outputs (no GPU needed).

    PYTHONPATH=src:. python independent_checks/t1/run_review.py \
        --probe docs/reports/r0_independent_t1_review/gpu_probe_results.json \
        --out docs/reports/r0_independent_t1_review/cpu_review_results.json [--limit N] [--jobs 8]

Per audited path (matched to its NPZ by SHA-256):
  * independent de Boor model == repo exact model (exact coefficient comparison);
  * independent Taylor-model enclosure built (positivity proved);
  * reviewed time_law_grid.build_grid(2^L) tables: full structural checks (finite, monotone,
    T_lo<=T_hi, fmin<=fmax) and exact containment on a sample of indices (all first/last 64,
    every knot-straddling and knot-adjacent cell, 512 random);
  * reviewed inverse_bracket containment at t = 0, 2^-149, 2^-126, 1e-30, T_native and its
    float32 neighbours, the audited argmax times, 64 random float32 times, and float32 times
    just above the independent T(1) upper bound (saturation);
  * independent epsilon: rigorous |s_native - s*(t)| at the audited argmax times (s_native
    from the GPU probe P7) compared with the reported eps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def bits32(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def review_one(args):
    row, npz, probe_row, log2 = args
    from curobo_certify import time_law_grid as TG
    from independent_checks.t1 import crosscheck as C
    from independent_checks.t1.taylor_time_law import TimeLaw
    t0 = time.time()
    with np.load(npz, allow_pickle=False) as z:
        tc = z["timing_controls"].copy()
        Tn = float(z["native_timing_time_grid"][-1])
    out = {"path": row["path"], "npz": str(npz)}
    out["model_coefficient_mismatches"] = C.model_identity_vs_repo(tc)
    tl = TimeLaw.from_timing_controls(tc)
    out["independent_status"] = tl.status
    if tl.status != "BOUNDED_EXACT_MODEL":
        return out
    lo1, hi1 = tl.total_bounds()
    out["independent_T1"] = [float(lo1), float(hi1), float(hi1 - lo1)]
    out["independent_cells"] = len(tl.cells)
    out["independent_max_K"] = tl.max_K
    grid = TG.build_grid(tc, log2)
    out["grid_status"] = grid.status
    if grid.status != "BOUNDED_EXACT_MODEL":
        return out
    N = grid.n_cells
    rng = np.random.default_rng(int(row["source_npz_sha256"][:8], 16))
    ks = set(range(0, 64)) | set(range(N - 63, N + 1))
    for m in range(1, 29):
        kc = (m * N) // 29
        ks |= {kc - 1, kc, kc + 1, kc + 2}
    ks |= set(int(v) for v in rng.integers(0, N + 1, 512))
    ks = sorted(k for k in ks if 0 <= k <= N)
    out["grid_check"] = C.check_grid_tables(grid, tl, ks)
    out["grid_check"]["indices_checked"] = len(ks)
    out["grid_duration_bracket"] = [float(grid.T_lo[-1]), float(grid.T_hi[-1])]
    out["duration_contains_independent"] = bool(Q(float(grid.T_lo[-1])) <= lo1 and Q(float(grid.T_hi[-1])) >= hi1)
    # times
    ts = [0.0, f32(1), f32(0x7FFFFF), f32(0x800000), 1e-30, Tn, f32(bits32(Tn) - 1), f32(bits32(Tn) + 1),
          row["at_time"], row["ahead_at"], row["behind_at"]]
    ts += [float(np.float32(v)) for v in rng.random(64) * Tn]
    b_hi = bits32(float(hi1))
    ts += [f32(b_hi + d) for d in (-2, -1, 0, 1, 2, 64)]
    ts = [float(np.float32(t)) for t in ts]
    out["inverse_check"] = C.check_inverse_bracket(grid, tl, ts, TG.inverse_bracket)
    sat = []
    for t in ts[-6:]:
        lo, hi = TG.inverse_bracket(grid, t)
        a, b, st = tl.inverse(Q(t))
        sat.append({"t": t, "their": [lo, hi], "independent": [float(a), float(b)], "status": st})
    out["saturation_probe"] = sat
    # independent epsilon at audited argmax times
    if probe_row is not None:
        eps_rows = []
        for t, s in zip(probe_row["t"], probe_row["s_native"]):
            bound, st = C.certified_error_at(tl, t, s)
            lo, hi = TG.inverse_bracket(grid, t)
            their_e = max(s - hi, lo - s, 0.0)
            eps_rows.append({"t": t, "s_native": s, "independent_error_upper_bound": float(bound),
                             "independent_bound_le_reported_eps": bool(bound <= Q(row["eps_s_certified_max"])),
                             "their_cpu_e_at_t": their_e, "status": st})
        out["independent_eps_at_argmax"] = eps_rows
    out["elapsed_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-json", type=Path, default=ROOT / "runs/margins/time_inversion_audit_224paths.json")
    ap.add_argument("--probe", type=Path, default=ROOT / "docs/reports/r0_independent_t1_review/gpu_probe_results.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--log2-cells", type=int, default=22)
    a = ap.parse_args()
    audit = json.loads(a.audit_json.read_text())
    probe = {}
    if a.probe.exists():
        probe = {r["path"]: r for r in json.loads(a.probe.read_text())["probes"].get("P7_native_s_at_argmax", [])}
    npzmap = {}
    for f in ROOT.glob("runs/**/*.npz"):
        npzmap.setdefault(f.stem, []).append(f)
    jobs = []
    for r in audit["rows"]:
        hit = [f for f in npzmap.get(r["path"], []) if hashlib.sha256(f.read_bytes()).hexdigest() == r["source_npz_sha256"]]
        if not hit:
            raise SystemExit(f"NPZ for {r['path']} with recorded SHA not found")
        jobs.append((r, hit[0], probe.get(r["path"]), a.log2_cells))
    if a.limit:
        jobs = jobs[:a.limit]
    from multiprocessing import Pool
    with Pool(a.jobs) as pool:
        results = []
        for res in pool.imap(review_one, jobs):
            results.append(res)
            print(json.dumps({k: res.get(k) for k in ("path", "model_coefficient_mismatches", "independent_status",
                                                      "grid_status", "duration_contains_independent", "elapsed_s")}), flush=True)
    summ = {"paths": len(results),
            "model_mismatch_paths": sum(1 for r in results if r.get("model_coefficient_mismatches")),
            "independent_positive": sum(1 for r in results if r.get("independent_status") == "BOUNDED_EXACT_MODEL"),
            "grid_bounded": sum(1 for r in results if r.get("grid_status") == "BOUNDED_EXACT_MODEL")}
    for fam in ("T_lo", "T_hi", "fmin", "fmax"):
        for o in ("proved", "violated", "undecided"):
            summ[f"{fam}_{o}"] = sum(r["grid_check"][fam][o] for r in results if "grid_check" in r)
    summ["structure_all_ok"] = all(all(r["grid_check"]["structure"].values()) for r in results if "grid_check" in r)
    summ["duration_contains_independent_all"] = all(r.get("duration_contains_independent") for r in results if "grid_check" in r)
    for side in ("lower", "upper"):
        for o in ("proved", "violated", "undecided"):
            summ[f"inverse_{side}_{o}"] = sum(r["inverse_check"][side][o] for r in results if "inverse_check" in r)
    summ["inverse_empty_brackets"] = sum(r["inverse_check"]["empty_brackets"] for r in results if "inverse_check" in r)
    eps = [e for r in results for e in r.get("independent_eps_at_argmax", [])]
    summ["eps_points"] = len(eps)
    summ["eps_independent_le_reported"] = sum(1 for e in eps if e["independent_bound_le_reported_eps"])
    summ["max_independent_error_at_argmax"] = max((e["independent_error_upper_bound"] for e in eps), default=None)
    reviewed = ["src/curobo_certify/time_law_grid.py", "src/curobo_certify/native_error_bound.py"]
    out = {"scope": "R0_INDEPENDENT_T1_CPU_REVIEW", "log2_cells": a.log2_cells, "summary": summ,
           "reviewed_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in reviewed},
           "numpy": np.__version__, "rows": results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""K4 P4 population analysis (K4-Panda-Dev40, exploratory, PAYLOAD_DEFAULT_0P01KG).

Joins (by NPZ SHA): generation ledger, per-output certification (k4_certify_outputs), native sampled
torque diagnostics (k4_native_torque_samples), certified retiming (k4_retime_run) and fresh latency
(k4_bench_fresh). Contracts kept separate: MODEL_K3, MODEL_K4 (physical), K3_PLUS_K4, PLANNING_K4 (cuRobo's
own 0.95-scaled torque limits). Coverage with exact Clopper-Pearson bounds; paired durations with
source-query bootstrap (10,000). "Between-sample" violations: rigorous K4 VIOLATED while every native
sampled torque (512 grid, S8193, S32768) is within the same limits. Arms are never pooled.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_formal_cohort import pct, proportion  # noqa: E402


def boot(values, seed=20261117, n=10000):
    if not values:
        return None
    rng = random.Random(seed)
    k = len(values)
    meds = sorted(statistics.median(values[rng.randrange(k)] for _ in range(k)) for _ in range(n))
    return {"n": k, "median": statistics.median(values), "ci95": [meds[int(0.025 * n)], meds[int(0.975 * n) - 1]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--cert", type=Path, required=True)
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--retime", type=Path)
    ap.add_argument("--fresh", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    gen = json.loads(a.gen.read_text())
    cert = {r["npz_sha256"]: r for r in json.loads(a.cert.read_text())["rows"]}
    nat = {r["npz_sha256"]: r for r in json.loads(a.native.read_text())["rows"]}
    ret = {r["npz_sha256"]: r for r in json.loads(a.retime.read_text())["rows"]} if a.retime else {}
    arms = {}
    stops = []
    for arm, rep in (("primary", 0), ("secondary", 0), ("stress", 0), ("primary_repeat1", 1), ("primary_repeat2", 2)):
        base = arm.split("_")[0]
        L = [e for e in gen["ledger"] if e["arm"] == base and e["repeat"] == rep]
        elig = [e for e in L if (e["selected_eligible"] if arm == "primary" else e["planner_success"])]
        rows = []
        for e in elig:
            c = cert.get(e["npz_sha256"])
            n = nat.get(e["npz_sha256"])
            if c is None or n is None:
                stops.append("missing evidence " + e["source_id"])
                continue
            samp_phys = max(g["max_util_physical"] for g in n["grids"].values())
            samp_plan = max(g["max_util_planning"] for g in n["grids"].values())
            between_phys = c["MODEL_K4"] == "VIOLATED_TORQUE" and samp_phys <= 1.0
            between_plan = c["PLANNING_K4"] == "VIOLATED_TORQUE" and samp_plan <= 1.0
            if c["MODEL_K4"] == "CERTIFIED" and samp_phys > 1.0 + 1e-4:
                stops.append("FALSE K4 CERTIFIED? native sampled exceeds physical: " + e["source_id"])
            if "REJECTED_BY_VERIFIER" in (c["MODEL_K3"], c["MODEL_K4"], c["PLANNING_K4"]):
                stops.append("verifier disagreement " + e["source_id"])
            rows.append({"source_id": e["source_id"], "pair_id": e["pair_id"], "scene": e["scene"], "magnitude": e["magnitude"],
                         "npz_sha256": e["npz_sha256"], "duration": c["native_duration_s"], "MODEL_K3": c["MODEL_K3"],
                         "MODEL_K4": c["MODEL_K4"], "K3_PLUS_K4": c["K3_PLUS_K4"], "PLANNING_K4": c["PLANNING_K4"],
                         "k4_bound_util": max(c["K4_physical"]["bound_util"] or [0]),
                         "rigorous_point_max_util": c["dense_point_max_util_physical"],
                         "native_sampled_max_util_physical": samp_phys, "native_sampled_max_util_planning": samp_plan,
                         "native_sampled_by_grid": {k: g["max_util_physical"] for k, g in n["grids"].items()},
                         "between_samples_violation_physical": between_phys,
                         "between_samples_violation_planning": between_plan,
                         "k4_witness": c["K4_physical"]["witness"] or c["K4_planning"]["witness"],
                         "k4_seconds": c["K4_physical"]["seconds_producer_incl_prep"],
                         "k4_verifier_seconds": c["K4_physical"]["seconds_verifier"]})
        n_el = len(rows)
        cnt = lambda col, v: sum(r[col] == v for r in rows)  # noqa: E731
        arms[arm] = {
            "attempted": len(L), "native_success": sum(e["planner_success"] for e in L), "eligible": n_el,
            "native_failures": sum(not e["planner_success"] for e in L),
            **{col: {"CERTIFIED": cnt(col, "CERTIFIED"),
                     "VIOLATED": sum(str(r[col]).startswith("VIOLATED") for r in rows),
                     "UNKNOWN": sum(not (r[col] == "CERTIFIED" or str(r[col]).startswith("VIOLATED")) for r in rows),
                     "coverage": proportion(cnt(col, "CERTIFIED"), n_el)}
               for col in ("MODEL_K3", "MODEL_K4", "K3_PLUS_K4", "PLANNING_K4")},
            "between_samples_violations_physical": sum(r["between_samples_violation_physical"] for r in rows),
            "between_samples_violations_planning": sum(r["between_samples_violation_planning"] for r in rows),
            "rigorous_point_max_util": pct([r["rigorous_point_max_util"] for r in rows]),
            "k4_bound_minus_point": pct([r["k4_bound_util"] - r["rigorous_point_max_util"] for r in rows]),
            "native_sampled_minus_rigorous_point": pct([r["native_sampled_max_util_physical"] - r["rigorous_point_max_util"] for r in rows]),
            "k4_seconds": pct([r["k4_seconds"] for r in rows]), "k4_verifier_seconds": pct([r["k4_verifier_seconds"] for r in rows]),
            "rows": rows}
    # paired durations
    byp = {}
    for arm in ("primary", "secondary", "stress"):
        for r in arms[arm]["rows"]:
            byp.setdefault(r["pair_id"], {})[arm] = r
    paired = {}
    for arm in ("secondary", "stress"):
        g = [100 * (v["primary"]["duration"] - v[arm]["duration"]) / v["primary"]["duration"] for v in byp.values()
             if "primary" in v and arm in v]
        paired[f"{arm}_vs_primary_pct_faster"] = boot(g)
    # repeats nondeterminism
    prim = {r["pair_id"]: r for r in arms["primary"]["rows"]}
    nd = {}
    for k in ("primary_repeat1", "primary_repeat2"):
        same = [r["npz_sha256"] == prim[r["pair_id"]]["npz_sha256"] for r in arms[k]["rows"] if r["pair_id"] in prim]
        dur = [abs(r["duration"] - prim[r["pair_id"]]["duration"]) for r in arms[k]["rows"] if r["pair_id"] in prim]
        nd[k] = {"compared": len(same), "bit_identical_npz": sum(same), "max_abs_duration_diff_s": max(dur) if dur else None,
                 "statuses_equal_K3_PLUS_K4": sum(r["K3_PLUS_K4"] == prim[r["pair_id"]]["K3_PLUS_K4"] for r in arms[k]["rows"] if r["pair_id"] in prim)}
    retime = None
    if ret:
        acc = [r for sha, r in ret.items() if r.get("accepted_lambda") is not None]
        gains = [r["gain_vs_native_pct"] for r in acc]
        retime = {"attempted": len(ret), "accepted_certified": len(acc), "gain_vs_native_0p95_pct": boot(gains),
                  "gain_distribution": pct(gains) if gains else None,
                  "seconds": pct([r["seconds"] for r in ret.values()])}
        # paired comparison with native 0.99 on the same source query (same geometry? no: different solve)
        g99 = []
        for sha, r in ret.items():
            p = next((x for x in arms["primary"]["rows"] if x["npz_sha256"] == sha), None)
            if p and r.get("retimed_duration_bracket_s") and p["pair_id"] in byp and "secondary" in byp[p["pair_id"]]:
                g99.append(100 * (byp[p["pair_id"]]["secondary"]["duration"] - r["retimed_duration_bracket_s"][1])
                           / byp[p["pair_id"]]["secondary"]["duration"])
        retime["certified_retimed_vs_native_0p99_pct_faster"] = boot(g99)
    fresh = None
    if a.fresh:
        fr = json.loads(a.fresh.read_text())
        med = lambda v: statistics.median(v)  # noqa: E731
        fresh = {"queries": len(fr["rows"]), "warmups": fr["warmups"], "repeats": fr["repeats"],
                 "native_ms": pct([med(r["native_ms"]) for r in fr["rows"]]),
                 "added_ms_K4": pct([med(r["added_ms_B"]) for r in fr["rows"]]),
                 "added_ms_K3_plus_K4": pct([med(r["added_ms_C"]) for r in fr["rows"]]),
                 "ci95_median_added_K4": boot([med(r["added_ms_B"]) for r in fr["rows"]]),
                 "stages_ms_p50": {k: statistics.median([rep["B"]["stages"][k] for r in fr["rows"] for rep in r["repeats"]
                                                          if rep["B"]["stages"].get(k) is not None])
                                   for k in ("plan_ms", "d2h_ms", "extract_ms", "k4_producer_ms", "k4_verifier_ms")},
                 "queries_multiple_digests": sum(r["distinct_digests"] > 1 for r in fr["rows"])}
    out = {"scope": "K4_EXPLORATORY_POPULATION_ANALYSIS", "cohort": gen["cohort"], "first_k": gen["first_k"],
           "payload": "PAYLOAD_DEFAULT_0P01KG; 3 kg / 5 kg BLOCKED_PAYLOAD", "arms": arms, "paired_durations": paired,
           "repeat_nondeterminism": nd, "retiming": retime, "fresh_latency": fresh, "hard_stops": stops,
           "gate": "PASS_SOUNDNESS" if not stops else "HARD_STOP"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({"gate": out["gate"], "stops": stops[:5],
                      **{arm: {k: v for k, v in s.items() if k in ("attempted", "eligible") or k in ("MODEL_K3", "MODEL_K4", "K3_PLUS_K4", "PLANNING_K4")
                               and False} | {c: s[c]["CERTIFIED"] for c in ("MODEL_K3", "MODEL_K4", "K3_PLUS_K4", "PLANNING_K4")}
                         for arm, s in arms.items()}}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze one frozen Panda-only formal cohort (AMENDED_POPULATION_PANDA_ONLY).

Joins, by exact NPZ SHA-256: the generation ledger (every attempt), the frozen-output
certification benchmark (MODEL_K3 = physical contract, NATIVE_REPORTED_K3 = native contract;
compiled producer + independent verifier), the exact tightness analysis (exact witnesses),
the v2 exhaustive T1 audit and the fresh paired latency benchmark. Three contracts are kept
in separate columns and never pooled; arms (primary 0.95 / secondary 0.99) are never pooled.

Soundness cross-checks (any failure = HARD_STOP):
  MODEL_K3 CERTIFIED   but exact tightness finds a genuine physical violation witness
  MODEL_K3 VIOLATED    without an exact witness in the tightness analysis
  NATIVE_REPORTED CERTIFIED but some production float32 output exceeds a limit (T1 exhaustive)
  NATIVE_REPORTED CERTIFIED but a saved sampled diagnostic exceeds a limit
  any REJECTED_BY_VERIFIER, any T1 audit integrity failure, any NPZ SHA mismatch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify import formal_cohorts as FC  # noqa: E402
from curobo_certify.margin_analysis import KNOWN, paired_bootstrap, _percentile  # noqa: E402

UNKNOWN_PREFIX = ("UNKNOWN", "NUMERICAL_ERROR", "INVALID_INPUT")


def betainc_lower(k: int, n: int, alpha: float) -> float | None:
    """Exact Clopper-Pearson lower bound (one-sided level 1-alpha) by bisection on the binomial tail."""
    if n == 0:
        return None
    if k == 0:
        return 0.0

    def tail(p):  # P[X >= k | p]
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if tail(mid) < alpha:
            lo = mid
        else:
            hi = mid
    return lo


def betainc_upper(k: int, n: int, alpha: float) -> float | None:
    if n == 0:
        return None
    if k == n:
        return 1.0

    def tail(p):  # P[X <= k | p]
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if tail(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def proportion(k: int, n: int) -> dict:
    return {"k": k, "n": n, "fraction": (k / n) if n else None,
            "one_sided_95_lower_exact": betainc_lower(k, n, 0.05),
            "two_sided_95_exact": [betainc_lower(k, n, 0.025), betainc_upper(k, n, 0.025)] if n else None,
            "meets_99_descriptive": (k / n >= 0.99) if n else None,
            "meets_99_inferential_one_sided": (betainc_lower(k, n, 0.05) >= 0.99) if n else None}


def pct(values, qs=(0.5, 0.95, 0.99)):
    v = sorted(values)
    return {f"p{int(q * 100)}": _percentile(v, q) for q in qs} | {"max": v[-1] if v else None, "n": len(v)}


def cluster_bootstrap_median(per_cluster: list[float], seed=20261117, samples=10000):
    if not per_cluster:
        return None
    rng = random.Random(seed)
    n = len(per_cluster)
    meds = sorted(statistics.median(per_cluster[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return [_percentile(meds, 0.025), _percentile(meds, 0.975)]


def load(p: Path | None):
    return json.loads(p.read_text()) if p else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", required=True, choices=sorted(FC.COHORTS))
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--records-dir", type=Path, required=True)
    ap.add_argument("--cert-bench", type=Path, required=True)
    ap.add_argument("--tightness", type=Path, required=True)
    ap.add_argument("--t1", type=Path, nargs="+", required=True)
    ap.add_argument("--fresh", type=Path, nargs="*", default=[])
    ap.add_argument("--fresh-arm", default="B8", help="registered primary certification arm (8 threads)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    raw = a.queries.read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.expected_sha256:
        raise SystemExit("FROZEN_MANIFEST_SHA_MISMATCH")
    queries = {r["source_id"]: r for r in json.loads(raw)}
    gen = json.loads((a.records_dir / "generation_summary.json").read_text())
    if gen["query_manifest_sha256"] != a.expected_sha256 or gen["cohort"] != a.cohort:
        raise SystemExit("GENERATION_SUMMARY_MISMATCH")
    stops: list[str] = []
    # ---------------- evidence indices keyed by NPZ SHA ----------------
    bench = load(a.cert_bench)
    B = {}
    for r in bench["rows"]:
        if r["final"] not in KNOWN:
            raise SystemExit("unknown status " + r["final"])
        B[(r["source_npz_sha256"], r["contract"], r["method"])] = r
    tight = {r["source_npz_sha256"]: r for r in load(a.tightness)["rows"]}
    T1 = {}
    t1_meta = []
    for p in a.t1:
        d = load(p)
        if d["scope"] != "T1_AUDITED_FLOAT32_TIMES_V2":
            raise SystemExit("T1 evidence is not v2")
        t1_meta.append({k: d[k] for k in ("gpu", "capability", "compile_flags", "source_sha256", "grid_module_sha256",
                                          "driver_script_sha256")})
        for r in d["rows"]:
            T1[r["source_npz_sha256"]] = r
    fresh = {}
    fresh_meta = []
    for p in a.fresh:
        d = load(p)
        fresh_meta.append({k: d.get(k) for k in ("gpu", "contract", "threads_arms", "warmups", "repeats",
                                                   "query_manifest_sha256", "builds")})
        for r in d["rows"]:
            fresh[r["source_id"]] = r
    # ---------------- per-output join ----------------
    per_arm = {}
    for arm in ("primary", "secondary"):
        rows = []
        for e in gen["ledger"]:
            if e["arm"] != arm:
                continue
            q = queries[e["source_id"]]
            rec = json.loads((a.records_dir / f"{e['source_id']}.json").read_text())
            if rec.get("source_query_sha256") != q["source_query_sha256"]:
                stops.append("record/query hash mismatch " + e["source_id"])
            eligible = e["selected_eligible"] if arm == "primary" else e["planner_success"]
            row = {"source_id": e["source_id"], "pair_id": e["pair_id"], "pool_order": e["pool_order"],
                   "scene": e["scene"], "magnitude": e["magnitude"], "native_status": e["status"],
                   "planner_success": e["planner_success"], "eligible": bool(eligible), "failure_reason": e["reason"]}
            if e["planner_success"]:
                npz = a.records_dir / f"{e['source_id']}.npz"
                sha = hashlib.sha256(npz.read_bytes()).hexdigest()
                if sha != rec.get("arrays_sha256") or sha != e["npz_sha256"]:
                    stops.append("NPZ SHA mismatch " + e["source_id"])
                row["npz_sha256"] = sha
                row["native_duration_s"] = rec["native_duration_seconds"]
                diag = rec["native_adapter_parity"]["dense_diagnostics"]
                row["sampled_max_physical_ratio"] = {d["grid"]: max(v["physical_max_ratio"] for v in d["ratios"].values())
                                                     for d in diag}
            if eligible:
                sha = row["npz_sha256"]
                mk = B.get((sha, "physical", "A"))
                nr = B.get((sha, "native", "A"))
                tg = tight.get(sha)
                t1 = T1.get(sha)
                row["MODEL_K3"] = mk["final"] if mk else "MISSING"
                row["MODEL_K3_producer"] = mk["producer"] if mk else None
                row["MODEL_K3_leaves"] = mk["leaves"] if mk else None
                row["MODEL_K3_max_depth"] = mk["max_depth_used"] if mk else None
                row["MODEL_K3_ms_p50"] = statistics.median(mk["repeats_ms_total"]) if mk else None
                row["NATIVE_REPORTED_K3"] = nr["final"] if nr else "MISSING"
                row["NATIVE_REPORTED_ms_p50"] = statistics.median(nr["repeats_ms_total"]) if nr else None
                gv = None
                if tg:
                    phys = tg["contracts"]["physical"]["nontrivial_residuals"]
                    gv = sum(x["class"] == "GENUINE_VIOLATION" for x in phys)
                    row["exact_genuine_violations"] = gv
                    row["exact_worst_witness"] = max((x["peak_lower"] for x in phys if x["class"] == "GENUINE_VIOLATION"),
                                                     default=None)
                    row["exact_unresolved"] = sum(x["class"] == "UNRESOLVED" for x in phys)
                else:
                    row["exact_genuine_violations"] = "MISSING"
                if t1:
                    row["T1"] = t1["status"]
                    row["T1_reported"] = t1["reported_status"]
                    row["T1_eps_s"] = t1["eps_s_certified_max"]
                    row["T1_reported_max_util"] = {f: t1["reported"][f]["max_utilization"] for f in ("v", "a", "j")}
                    row["T1_offline_s"] = t1["elapsed_s"]
                    row["T1_stages_s"] = t1["stages_s"]
                    row["T1_times"] = t1["times_enumerated"]
                    row["T1_relative_deviation"] = t1["relative_deviation_vs_limits"]
                else:
                    row["T1"] = "MISSING"
                # ---- soundness cross-checks ----
                if row["MODEL_K3"] == "MISSING" or row["NATIVE_REPORTED_K3"] == "MISSING" or row["T1"] == "MISSING" \
                        or gv == "MISSING" or gv is None:
                    stops.append("missing evidence " + e["source_id"])
                if "REJECTED_BY_VERIFIER" in (row["MODEL_K3"], row["NATIVE_REPORTED_K3"]):
                    stops.append("verifier disagreement " + e["source_id"])
                if row["MODEL_K3"] == "CERTIFIED" and isinstance(gv, int) and gv > 0:
                    stops.append("FALSE MODEL_K3 CERTIFIED (exact witness exists) " + e["source_id"])
                if row["MODEL_K3"] == "VIOLATED" and isinstance(gv, int) and gv == 0:
                    stops.append("MODEL_K3 VIOLATED without exact witness " + e["source_id"])
                if row["T1"] not in ("AUDITED_CONSISTENT", "MISSING"):
                    stops.append("T1 integrity failure " + e["source_id"])
                if row["NATIVE_REPORTED_K3"] == "CERTIFIED":
                    if row.get("T1_reported") != "REPORTED_WITHIN_LIMITS_ALL_FLOAT32_TIMES":
                        stops.append("FALSE NATIVE_REPORTED CERTIFIED (exhaustive production exceedance) " + e["source_id"])
                    if any(v > 1.0 for v in row["sampled_max_physical_ratio"].values()):
                        stops.append("FALSE NATIVE_REPORTED CERTIFIED (sampled exceedance) " + e["source_id"])
            rows.append(row)
        el = [r for r in rows if r["eligible"]]
        n = len(el)

        def count(col, val):
            return sum(r.get(col) == val for r in el)
        unknown_mk = sum(str(r.get("MODEL_K3", "")).startswith(UNKNOWN_PREFIX) for r in el)
        summary = {
            "attempted_source_queries": len(rows), "native_success": sum(r["planner_success"] for r in rows),
            "native_failures": sum(not r["planner_success"] for r in rows), "eligible": n,
            "native_success_rate": proportion(sum(r["planner_success"] for r in rows), len(rows)),
            "MODEL_K3": {"CERTIFIED": count("MODEL_K3", "CERTIFIED"), "VIOLATED": count("MODEL_K3", "VIOLATED"),
                         "VIOLATED_BOUNDARY": count("MODEL_K3", "VIOLATED_BOUNDARY"), "UNKNOWN": unknown_mk,
                         "coverage": proportion(count("MODEL_K3", "CERTIFIED"), n),
                         "status_counts": _counts(el, "MODEL_K3")},
            "NATIVE_REPORTED_K3": {"CERTIFIED": count("NATIVE_REPORTED_K3", "CERTIFIED"),
                                   "UNKNOWN_TIGHTENED_WITNESS": count("NATIVE_REPORTED_K3", "UNKNOWN_TIGHTENED_WITNESS"),
                                   "coverage": proportion(count("NATIVE_REPORTED_K3", "CERTIFIED"), n),
                                   "status_counts": _counts(el, "NATIVE_REPORTED_K3")},
            "T1_AUDITED_FLOAT32_TIMES": {
                "audited_consistent": count("T1", "AUDITED_CONSISTENT"),
                "audit_fraction": proportion(count("T1", "AUDITED_CONSISTENT"), n),
                "reported_within_limits": count("T1_reported", "REPORTED_WITHIN_LIMITS_ALL_FLOAT32_TIMES"),
                "reported_exceeds": count("T1_reported", "REPORTED_EXCEEDS_LIMITS"),
                "times_enumerated_total": sum(r.get("T1_times", 0) for r in el),
                "eps_s": pct([r["T1_eps_s"] for r in el if isinstance(r.get("T1_eps_s"), float)]),
                "offline_audit_s": pct([r["T1_offline_s"] for r in el if "T1_offline_s" in r]),
                "relative_state_deviation_max": {f: max((r["T1_relative_deviation"][f] for r in el if "T1_relative_deviation" in r),
                                                        default=None) for f in ("v", "a", "j")}},
            "exact_genuine_violation_outputs": sum(isinstance(r.get("exact_genuine_violations"), int)
                                                   and r["exact_genuine_violations"] > 0 for r in el),
            "by_scene": {s: {"attempted": sum(r["scene"] == s for r in rows), "eligible": sum(r["scene"] == s for r in el),
                             "model_k3_certified": sum(r["scene"] == s and r.get("MODEL_K3") == "CERTIFIED" for r in el)}
                         for s in sorted({r["scene"] for r in rows})},
        }
        per_arm[arm] = {"summary": summary, "rows": rows}
    # ---------------- paired durations (secondary vs primary) ----------------
    prim = {r["pair_id"]: r for r in per_arm["primary"]["rows"] if r["eligible"]}
    sec = {r["pair_id"]: r for r in per_arm["secondary"]["rows"] if r["eligible"]}
    gains = [100 * (prim[p]["native_duration_s"] - sec[p]["native_duration_s"]) / prim[p]["native_duration_s"]
             for p in sorted(prim) if p in sec]
    paired = paired_bootstrap(gains, seed=20261117, samples=10000) | {
        "definition": "100*(T_0.95 - T_0.99)/T_0.95 per source query; positive = 0.99 faster",
        "pairs_primary_eligible": len(prim), "pairs_with_secondary_success": len(gains)}
    # ---------------- frozen-output method comparison ----------------
    methods = {}
    for (sha, contract, method), r in B.items():
        if sha not in {x["npz_sha256"] for x in prim.values()}:
            continue
        m = methods.setdefault(f"{contract}/{method}", {"n": 0, "certified": 0, "violated": 0, "unknown": 0, "p50s": []})
        m["n"] += 1
        m["certified"] += r["final"] == "CERTIFIED"
        m["violated"] += r["final"] in ("VIOLATED", "VIOLATED_BOUNDARY")
        m["unknown"] += r["final"] not in ("CERTIFIED", "VIOLATED", "VIOLATED_BOUNDARY")
        m["p50s"].append(statistics.median(r["repeats_ms_total"]))
    for m in methods.values():
        p = m.pop("p50s")
        m["per_path_median_ms"] = pct(p)
        m["ci95_median_ms"] = cluster_bootstrap_median(p)
    # ---------------- fresh latency ----------------
    lat = None
    if fresh:
        per_q, pooled, per_q_by_status, stages = [], [], {}, {}
        for sid, r in fresh.items():
            arm_key = a.fresh_arm
            add = r["added_ms_by_arm"].get(arm_key)
            if add is None:
                raise SystemExit("fresh benchmark lacks registered arm " + arm_key)
            med = statistics.median(add)
            per_q.append(med)
            pooled += add
            st = r["repeats"][0][arm_key]["status"]
            per_q_by_status.setdefault(st, []).append(med)
            for s in r["stage_by_arm"].get(arm_key, []):
                for k, v in s.items():
                    stages.setdefault(k, []).append(v)
        p50 = statistics.median(per_q)
        p95 = _percentile(sorted(per_q), 0.95)
        lat = {"arm": a.fresh_arm, "queries": len(per_q), "per_query_median_added_ms": pct(per_q),
               "ci95_median_added_ms": cluster_bootstrap_median(per_q), "pooled_requests_added_ms": pct(pooled),
               "by_status": {k: pct(v) for k, v in per_q_by_status.items()},
               "stage_ms_pooled": {k: pct(v) for k, v in stages.items()},
               "native_ms_per_query_median": pct([statistics.median(r["native_ms"]) for r in fresh.values()]),
               "distinct_output_digest_queries": sum(r["distinct_output_digests"] > 1 for r in fresh.values()),
               "targets": {"p50_le_1ms": p50 <= 1.0, "p95_le_3ms": p95 <= 3.0},
               "label": "PASS" if (p50 <= 1.0 and p95 <= 3.0) else "FAILED_PARTIAL_TARGET",
               "statuses_by_arm_disagree": None}
        other = [k for k in next(iter(fresh.values()))["added_ms_by_arm"] if k != a.fresh_arm]
        lat["secondary_arms"] = {k: pct([statistics.median(r["added_ms_by_arm"][k]) for r in fresh.values()]) for k in other}
    out = {"cohort": a.cohort, "population_flag": FC.POPULATION_FLAG,
           "original_protocol": FC.ORIGINAL_PROTOCOL_STATUS, "query_manifest_sha256": a.expected_sha256,
           "first_k": gen["first_k"], "gpu": gen["gpu"],
           "contracts_note": ("MODEL_K3: exact serialized model, continuous time; NATIVE_REPORTED_K3: conditional "
                              "(A1-A5) FP32 reported values at every native progress; T1: per-output exhaustive "
                              "float32 requested-time audit on this binary/GPU. None is a physical execution claim."),
           "arms": {k: v["summary"] for k, v in per_arm.items()},
           "paired_duration_0p99_vs_0p95": paired, "frozen_output_methods_primary": methods,
           "fresh_latency_primary": lat, "t1_evidence": t1_meta, "fresh_evidence": fresh_meta,
           "hard_stops": stops, "gate": "PASS_SOUNDNESS" if not stops else "HARD_STOP",
           "rows": {k: v["rows"] for k, v in per_arm.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"gate": out["gate"], "stops": stops[:10],
                      "primary": {k: out["arms"]["primary"][k] for k in ("attempted_source_queries", "eligible")},
                      "primary_MODEL_K3": out["arms"]["primary"]["MODEL_K3"]["status_counts"],
                      "primary_NATIVE": out["arms"]["primary"]["NATIVE_REPORTED_K3"]["status_counts"],
                      "secondary_MODEL_K3": out["arms"]["secondary"]["MODEL_K3"]["status_counts"],
                      "latency": lat and {k: lat[k] for k in ("per_query_median_added_ms", "label")}}, indent=1))


def _counts(rows, col):
    out = {}
    for r in rows:
        out[r.get(col)] = out.get(r.get(col), 0) + 1
    return out


if __name__ == "__main__":
    main()

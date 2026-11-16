"""Fail-closed paired analysis for real, frozen, multi-margin TimeOpt outputs.

No benchmark results are invented. A 'native reported' certificate guarantees
reported kernel quantities only under registered arithmetic assumptions; it
does NOT establish derivatives of the physical-time trajectory.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from decimal import Decimal

from .margin_campaign import as_margin, check_manifest, digest


def margin_key(value) -> str:
    """Single canonical string for a margin (3 decimals): '0.95' and '0.950' agree.

    Regression (2026-11-17): the manifest summary used '0.950' while pair grouping
    used str(Decimal('0.95')) == '0.95', raising KeyError for every analysis."""
    s = format(as_margin(value).quantize(Decimal("0.001")), "f")
    return s[:-1] if s.endswith("0") else s  # registered names: 0.90 0.95 0.975 0.99 1.00

PASS = "CERTIFIED"
KNOWN = {"CERTIFIED", "VIOLATED", "UNKNOWN_DEPTH", "UNKNOWN_NODES",
         "UNKNOWN_UNIFORM", "UNKNOWN_BOUNDARY", "VIOLATED_BOUNDARY",
         "NUMERICAL_ERROR", "INVALID_INPUT", "UNKNOWN_TIGHTENED_WITNESS",
         "REJECTED_BY_VERIFIER"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _median(values):
    return statistics.median(values) if values else None


def _percentile(sorted_values: list[float], alpha: float):
    if not sorted_values:
        return None
    pos = alpha * (len(sorted_values) - 1)
    i = int(pos)
    weight = pos - i
    if i + 1 == len(sorted_values):
        return sorted_values[-1]
    return sorted_values[i] * (1 - weight) + sorted_values[i + 1] * weight


def paired_bootstrap(values: list[float], *, seed=20261117, samples=4000):
    """Bootstrap SOURCE-QUERY PAIRS (not replicate timings) for median gain."""
    if not values:
        return {"n": 0, "median": None, "ci95": None, "bootstrap_unit": "pair_id"}
    if not all(math.isfinite(x) for x in values):
        raise ValueError("nonfinite gains")
    rng = random.Random(seed)
    n = len(values)
    medians = sorted(statistics.median(values[rng.randrange(n)] for _ in range(n))
                     for _ in range(samples))
    return {"n": n, "median": statistics.median(values),
            "ci95": [_percentile(medians, 0.025), _percentile(medians, 0.975)],
            "bootstrap_unit": "pair_id", "resamples": samples, "seed": seed}


def index_benchmark(file: Path | None, *, method="A") -> dict:
    if file is None:
        return {}
    data = _read_json(file)
    if data.get("scope") != "FROZEN_OUTPUT_FULL_CERTIFICATE_CPU_EXPLORATORY":
        raise ValueError("unknown benchmark evidence scope")
    result = {}
    for row in data["rows"]:
        if row.get("method") != method or row.get("contract") not in {"physical", "native"}:
            continue
        sha = row["source_npz_sha256"]
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("missing frozen NPZ SHA")
        status = row["final"]
        if status not in KNOWN:
            raise ValueError(f"unknown certifier status: {status}")
        key = (sha, row["contract"])
        if key in result:
            raise ValueError("duplicate benchmark row for source SHA / contract")
        times = row.get("repeats_ms_total", [])
        if not isinstance(times, list) or not times or any(
            not isinstance(x, (float,int)) or not math.isfinite(x) or x < 0 for x in times
        ):
            raise ValueError("missing or invalid complete certificate latency")
        result[key] = {"status": status, "latency_p50_ms": float(_median(times)),
                       "times_ms": times,
                       "source": "FROZEN_OUTPUT_FULL_CERTIFICATE_CPU_EXPLORATORY",
                       "digest": sha}
    return result


def _validate_record(row, record, *, verify_arrays: bool):
    if record.get("source_id") != row["source_id"]:
        raise ValueError("record source_id does not match frozen query")
    if record.get("source_query_sha256") != row["source_query_sha256"]:
        raise ValueError("record query hash mismatch")
    if record.get("source_query") is not None and record["source_query"] != row:
        raise ValueError("record source query differs from frozen query")
    if record.get("solver_condition") is not None and record["solver_condition"] != row["solver_condition"]:
        raise ValueError("record condition mismatch")
    if record.get("status") == "FAILED_RETAINED":
        return {"planner_status": "FAILED_RETAINED", "native_duration_s": None,
                "arrays_sha256": None, "source_array_verification": "NOT_APPLICABLE"}
    if record.get("planner_success") is not True:
        raise ValueError("nonfailed record lacks native success")
    if not str(record.get("status", "")).startswith("NATIVE_SUCCESS_PARITY_"):
        raise ValueError("native successful record has unexpected status")
    duration = record.get("native_duration_seconds")
    if not isinstance(duration, (int,float)) or isinstance(duration, bool) or not (
        math.isfinite(duration) and duration > 0
    ):
        raise ValueError("invalid native trajectory duration")
    sha = record.get("arrays_sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("missing NPZ SHA")
    verification = "REPORTED_SOURCE_SHA_ONLY"
    if verify_arrays:
        path = Path(record.get("arrays_path", ""))
        if not path.is_file():
            raise ValueError("missing frozen source arrays to verify: " + str(path))
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("frozen source NPZ SHA changed")
        verification = "REVERIFIED_SOURCE_NPZ_SHA"
    return {"planner_status": record["status"],
            "native_duration_s": float(duration),
            "arrays_sha256": sha, "source_array_verification": verification}


def build_analysis(manifest: list[dict], records_dir: Path, *,
                   benchmark_file: Path | None = None,
                   verify_arrays=False, seed=20261117, bootstrap=4000) -> dict:
    desc = check_manifest(manifest)
    if bootstrap < 100:
        raise ValueError("bootstrap must have at least 100 resamples")
    index = index_benchmark(benchmark_file)
    found = set()
    rows = []
    for query in manifest:
        record_path = records_dir / (query["source_id"] + ".json")
        d = {
            "source_id": query["source_id"],
            "pair_id": query["pair_id"],
            "margin": query["planning_margin"],
            "scene": query["scene_id"],
            "magnitude": query["magnitude_stratum"],
            "query_sha256": query["source_query_sha256"],
            "native_reported_contract": "REPORTED_VALUES_NOT_TIME_DERIVATIVES",
            "time_law_status": "BLOCKED_NATIVE_TIME_INVERSION",
        }
        if record_path.exists():
            record = _read_json(record_path)
            d.update(_validate_record(query, record, verify_arrays=verify_arrays))
            found.add(record_path.name)
        else:
            d.update({"planner_status": "NOT_RUN", "native_duration_s": None,
                      "arrays_sha256": None, "source_array_verification": "NOT_RUN"})
        for contract, key in (("physical", "model_k3"), ("native", "native_reported")):
            entry = index.get((d["arrays_sha256"], contract)) if d["arrays_sha256"] else None
            # In a tightened-limit contract, a producer's violation witness does NOT
            # prove violation against original physical limits. No false claims.
            if contract == "native" and entry is not None and entry["status"] == "VIOLATED":
                raise ValueError("unqualified tightened-limit violation in native contract")
            d[key + "_status"] = entry["status"] if entry else "NOT_RUN"
            d[key + "_latency_ms"] = entry["latency_p50_ms"] if entry else None
        rows.append(d)
    expected = {q["source_id"] + ".json" for q in manifest}
    unexpected = sorted(x.name for x in records_dir.glob("*.json") if x.name not in expected and x.name != "summary.json")
    if unexpected:
        raise ValueError("unregistered records present (do not silently pool): "+repr(unexpected[:10]))
    margins = [margin_key(m) for m in desc["margins"]]
    by_margin = {}
    for margin in margins:
        group = [r for r in rows if margin_key(r["margin"]) == margin_key(margin)]
        native = [r for r in group if r["native_duration_s"] is not None]
        med_duration = _median([r["native_duration_s"] for r in native])
        by_margin[margin] = {
            "attempted": len(group), "not_run": sum(r["planner_status"] == "NOT_RUN" for r in group),
            "native_success": len(native), "native_failed": len(group)-len(native)-sum(
                r["planner_status"] == "NOT_RUN" for r in group),
            "duration_p50_s_success_only": med_duration,
            "model_certified": sum(r["model_k3_status"] == PASS for r in group),
            "model_violated": sum(r["model_k3_status"] in ("VIOLATED", "VIOLATED_BOUNDARY") for r in group),
            "model_unknown": sum(r["model_k3_status"] not in ("CERTIFIED", "VIOLATED", "VIOLATED_BOUNDARY", "NOT_RUN")
                                 for r in group),
            "native_reported_certified": sum(r["native_reported_status"] == PASS for r in group),
            "native_reported_unknown_tightened": sum(
                r["native_reported_status"] == "UNKNOWN_TIGHTENED_WITNESS" for r in group),
            "certification_not_run": sum(r["native_reported_status"] == "NOT_RUN" for r in group),
            "native_reported_latency_p50_ms": _median([
                r["native_reported_latency_ms"] for r in group
                if r["native_reported_latency_ms"] is not None]),
            "full_physical_time_certified": 0,
            "time_law_proven": False,
        }
    pairs: dict[str, dict] = {}
    for row in rows:
        m = margin_key(row["margin"])
        pairs.setdefault(row["pair_id"], {})[m] = row
    base = margin_key("0.95")
    comparisons = {}
    for margin in margins:
        m = margin_key(margin)
        if m == base:
            continue  # the baseline is not compared with itself
        gains = []
        pair_rows = []
        counts = {"baseline_only_success": 0, "candidate_only_success": 0,
                  "both_failed_or_not_run": 0}
        for pid, group in sorted(pairs.items()):
            if base not in group or m not in group:
                raise ValueError(f"pair {pid} lacks a registered arm")
            anchor, arm = group[base], group[m]
            a, b = anchor["native_duration_s"], arm["native_duration_s"]
            if a is None or b is None:
                if a is not None:
                    counts["baseline_only_success"] += 1
                elif b is not None:
                    counts["candidate_only_success"] += 1
                else:
                    counts["both_failed_or_not_run"] += 1
                continue
            gain = 100 * (a-b) / a
            gains.append(gain)
            pair_rows.append({"pair_id": pid, "baseline_duration_s": a,
                              "candidate_duration_s": b, "gain_pct": gain,
                              "baseline_model_status": anchor["model_k3_status"],
                              "candidate_model_status": arm["model_k3_status"],
                              "baseline_native_reported_status": anchor["native_reported_status"],
                              "candidate_native_reported_status": arm["native_reported_status"]})
        comparisons[margin] = {"duration_gain_pct_paired_success_only": paired_bootstrap(
            gains, seed=seed, samples=bootstrap),
            "missingness": counts, "matched_success_pairs": pair_rows,
            "interpretation": "positive gain means candidate faster; conditional on BOTH native solves succeeding"}
    return {"schema_version": 1,
            "scope": "EXPLORATORY_MATCHED_MARGIN_DURATION_AND_CERTIFIABILITY_NOT_NATIVE_TIME_LAW_PROOF",
            "readme": "Unobserved evidence is NOT_RUN; never infer native physical-time derivative safety from reported-value certificates.",
            "manifest": desc, "source_files_verified": bool(verify_arrays),
            "benchmark_source": str(benchmark_file) if benchmark_file else None,
            "denominators": {"source_queries": len(rows), "pairs": desc["pairs"],
                             "attempted_gpu_queries": sum(r["planner_status"] != "NOT_RUN" for r in rows)},
            "summary_by_margin": by_margin, "paired_to_0p95": comparisons, "rows": rows}

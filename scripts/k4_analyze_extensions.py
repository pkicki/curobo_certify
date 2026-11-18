#!/usr/bin/env python3
"""K4 extension analysis: identified Panda model (Gaz et al. 2019) and preregistered synthetic payloads.

Inputs are k4_certify_outputs JSON files (one per contract / payload / friction variant), the generation
ledger(s), and optional native sampled torque diagnostics. Every contract is reported separately; nothing is
pooled across contracts, payloads or margin arms. Native sampled torques (cuRobo float32 GPU dynamics) are only
joined for variants whose dynamics equal the native planner's (cuRobo contract, with the payload hook when a
payload is used); for the identified model, between-sample violations are judged against the same model's
rigorous point torques at 24 points/span (dense_point_max_util). MODEL_K3 is dynamics-independent: when a run used
--skip-k3, K3 is joined from the reference run on identical NPZ bytes (same SHA-256) and the K3_PLUS_K4
column is recomputed from the joined K3 and the run's own verified K4.

Variants are given as NAME=CERT_JSON. Ledgers: --ledger ARM_SOURCE=GEN_JSON[:PAYLOAD].
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_formal_cohort import pct, proportion  # noqa: E402

COLS = ("MODEL_K3", "MODEL_K4", "K3_PLUS_K4", "PLANNING_K4")


def combine(k3, k4):
    if k3 == "CERTIFIED" and k4 == "CERTIFIED":
        return "CERTIFIED"
    if k3 in ("VIOLATED", "VIOLATED_BOUNDARY") or k4 in ("VIOLATED_TORQUE", "VIOLATED_X"):
        return "VIOLATED"
    return "UNKNOWN"


def load_rows(paths):
    meta, rows = None, []
    for p in paths:
        d = json.loads(Path(p).read_text())
        m = {k: d.get(k) for k in ("contract_sha256", "model_scope", "payload", "friction_bound")}
        if meta is not None and m != meta:
            raise SystemExit(f"mixed contracts in one variant: {p}")
        meta = m
        rows += d["rows"]
    return meta, {r["npz_sha256"]: r for r in rows}


def summarize(rows, ledger, k3ref, native):
    out, stops = {}, []
    for arm in sorted({e["arm"] for e in ledger}):
        L = [e for e in ledger if e["arm"] == arm and e.get("repeat", 0) == 0]
        elig = [e for e in L if (e["selected_eligible"] if arm == "primary" else e["planner_success"])]
        R = []
        for e in elig:
            c = rows.get(e["npz_sha256"])
            if c is None:
                stops.append("missing certification " + e["source_id"])
                continue
            k3 = c["MODEL_K3"]
            if k3 == "NOT_RECOMPUTED":
                k3 = (k3ref.get(e["npz_sha256"]) or {}).get("MODEL_K3", "MISSING_K3")
                if k3 == "MISSING_K3":
                    stops.append("missing K3 join " + e["source_id"])
            if "REJECTED_BY_VERIFIER" in (k3, c["MODEL_K4"], c["PLANNING_K4"]):
                stops.append("verifier disagreement " + e["source_id"])
            n = native.get(e["npz_sha256"]) if native else None
            samp = max(g["max_util_physical"] for g in n["grids"].values()) if n else None
            between = (c["MODEL_K4"] == "VIOLATED_TORQUE" and samp is not None and samp <= 1.0)
            grid_phys = n["grids"]["native_grid_512"]["max_util_physical"] if n else None
            grid_plan = n["grids"]["native_grid_512"]["max_util_planning"] if n else None
            samp_plan = max(g["max_util_planning"] for g in n["grids"].values()) if n else None
            R.append({"source_id": e["source_id"], "pair_id": e["pair_id"], "npz_sha256": e["npz_sha256"],
                      "duration": c["native_duration_s"], "MODEL_K3": k3, "MODEL_K4": c["MODEL_K4"],
                      "K3_PLUS_K4": combine(k3, c["MODEL_K4"]), "PLANNING_K4": c["PLANNING_K4"],
                      "k4_bound_util": max(c["K4_physical"]["bound_util"] or [0]),
                      "rigorous_point_max_util": c["dense_point_max_util_physical"],
                      "native_sampled_max_util_physical": samp, "between_samples_violation_physical": between,
                      "native_sampled_max_util_planning": samp_plan,
                      "native_grid512_max_util_physical": grid_phys, "native_grid512_max_util_planning": grid_plan,
                      # violation hidden from TimeOpt's own 512-sample enforcement grid
                      "between_grid512_violation_physical": c["MODEL_K4"] == "VIOLATED_TORQUE" and grid_phys is not None and grid_phys <= 1.0,
                      "between_grid512_violation_planning": c["PLANNING_K4"] == "VIOLATED_TORQUE" and grid_plan is not None and grid_plan <= 1.0,
                      "between_all_samples_violation_planning": c["PLANNING_K4"] == "VIOLATED_TORQUE" and samp_plan is not None and samp_plan <= 1.0,
                      "planning_witness": c["K4_planning"]["witness"],
                      "between_dense_rigorous_points": (c["MODEL_K4"] == "VIOLATED_TORQUE"
                                                         and c["dense_point_max_util_physical"] <= 1.0),
                      "witness": c["K4_physical"]["witness"]})
        n_el = len(R)
        out[arm] = {"attempted": len(L), "native_success": sum(e["planner_success"] for e in L),
                    "native_failures": sum(not e["planner_success"] for e in L), "eligible": n_el,
                    **{col: {"CERTIFIED": sum(r[col] == "CERTIFIED" for r in R),
                             "VIOLATED": sum(str(r[col]).startswith("VIOLATED") for r in R),
                             "UNKNOWN": sum(not (r[col] == "CERTIFIED" or str(r[col]).startswith("VIOLATED")) for r in R),
                             "coverage": proportion(sum(r[col] == "CERTIFIED" for r in R), n_el)} for col in COLS},
                    "between_samples_violations_physical": sum(r["between_samples_violation_physical"] for r in R),
                    **{k: sum(r[k] for r in R) for k in ("between_grid512_violation_physical", "between_grid512_violation_planning",
                                                          "between_all_samples_violation_planning")},
                    "planning_contract_note": ("PLANNING_K4 uses cuRobo's 0.95-scaled torque limits; this arm was planned with "
                                               "its own TimeOpt scale, so PLANNING_K4 violations here are expected" if arm != "primary" else
                                               "PLANNING_K4 = this arm's own TimeOpt torque limits (0.95-scaled, float32)"),
                    "between_dense_rigorous_points": sum(r["between_dense_rigorous_points"] for r in R),
                    "native_sampled_available": sum(r["native_sampled_max_util_physical"] is not None for r in R),
                    "rigorous_point_max_util": pct([r["rigorous_point_max_util"] for r in R]),
                    "k4_bound_util": pct([r["k4_bound_util"] for r in R]),
                    "rows": R}
    return out, stops


def paired_payload_durations(res, base_analysis, payload_population):
    import random
    import statistics
    pop = {r["source_id"]: r for r in json.loads(Path(payload_population).read_text())}
    an = json.loads(Path(base_analysis).read_text())
    base = {r["source_id"]: r["duration"] for arm in ("primary", "secondary") for r in an["arms"][arm]["rows"]}
    out = {}
    for v, d in res.items():
        if not d["contract"].get("payload") or not v.startswith("cu_"):
            continue
        for arm, x in d["arms"].items():
            def b(r):
                s = pop[r["source_id"]]["base_source_id"]
                return s if arm == "primary" else s.replace("margin_0p950", "margin_0p990")
            g = [100 * (r["duration"] - base[b(r)]) / base[b(r)] for r in x["rows"] if b(r) in base]
            if not g:
                continue
            rng, k = random.Random(20261117), len(g)
            meds = sorted(statistics.median(g[rng.randrange(k)] for _ in range(k)) for _ in range(10000))
            out[f"{v}_{arm}"] = {"n": k, "median_pct_slower_than_no_payload": statistics.median(g),
                                 "ci95": [meds[250], meds[9749]], "min": min(g), "max": max(g)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", nargs="+", required=True, help="NAME=cert.json[,cert2.json...]")
    ap.add_argument("--variant-ledger", nargs="+", required=True, help="NAME=gen.json[:PAYLOAD]")
    ap.add_argument("--variant-native", nargs="*", default=[], help="NAME=native_torque.json")
    ap.add_argument("--k3-reference", nargs="+", type=Path, required=True, help="cert JSON(s) with recomputed MODEL_K3")
    ap.add_argument("--base-analysis", type=Path, help="k4_analyze_population output (no-payload Dev40) for paired durations")
    ap.add_argument("--payload-population", type=Path, help="K4-Panda-PayloadDev40.json (base_source_id mapping)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    k3ref = {}
    for p in a.k3_reference:
        for r in json.loads(p.read_text())["rows"]:
            if r["MODEL_K3"] != "NOT_RECOMPUTED":
                k3ref[r["npz_sha256"]] = r
    ledgers = {}
    for s in a.variant_ledger:
        name, spec = s.split("=", 1)
        path, _, pay = spec.partition(":")
        g = json.loads(Path(path).read_text())
        ledgers[name] = (g.get("payloads") or g.get("conditions"))[pay]["ledger"] if pay else g["ledger"]
    natives = {}
    for s in a.variant_native:
        name, path = s.split("=", 1)
        natives[name] = {r["npz_sha256"]: r for r in json.loads(Path(path).read_text())["rows"]}
    res, stops = {}, []
    for s in a.variant:
        name, paths = s.split("=", 1)
        meta, rows = load_rows(paths.split(","))
        arms, st = summarize(rows, ledgers[name], k3ref, natives.get(name))
        res[name] = {"contract": meta, "arms": arms}
        stops += [f"{name}: {x}" for x in st]
    out = {"scope": "K4_EXTENSIONS_IDENTIFIED_MODEL_AND_SYNTHETIC_PAYLOADS", "variants": res, "hard_stops": stops,
           "gate": "PASS_SOUNDNESS" if not stops else "HARD_STOP"}
    if a.base_analysis and a.payload_population:
        out["payload_duration_vs_no_payload"] = paired_payload_durations(res, a.base_analysis, a.payload_population)
        out["payload_duration_note"] = ("paired with the no-payload K4-Panda-Dev40 output of the same frozen source query and "
                                        "same margin arm (primary 0.95 / secondary 0.99); only pairs where both succeeded; "
                                        "source-query bootstrap 10,000")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({"gate": out["gate"], "stops": stops[:5],
                      **{n: {arm: {c: f'{s[c]["CERTIFIED"]}/{s["eligible"]} V{s[c]["VIOLATED"]} U{s[c]["UNKNOWN"]}' for c in COLS}
                             | {"between_native": s["between_samples_violations_physical"], "between_dense": s["between_dense_rigorous_points"],
                                "grid512_phys": s["between_grid512_violation_physical"], "grid512_plan": s["between_grid512_violation_planning"],
                                "allsamp_plan": s["between_all_samples_violation_planning"]}
                             for arm, s in v["arms"].items()} for n, v in res.items()}}, indent=1))


if __name__ == "__main__":
    main()

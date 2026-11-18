#!/usr/bin/env python3
"""K4 corrected-model comparison: TimeOpt planning with cuRobo's model vs with the identified+hand model.

Both populations are judged under the SAME identified+hand(+payload) contract (MODEL_K4, physical limits;
friction-bound variant separately). Pairs share the frozen source query (pool order): cuRobo-model outputs are
K4-Panda-Dev40 primary repeat 0 (no payload) and K4-Panda-PayloadDev40 primary (3 / 5 kg); corrected outputs are
K4-Panda-CorrectedDev40. Coverage: exact Clopper-Pearson; durations: paired source-query bootstrap (10,000).
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_formal_cohort import pct, proportion  # noqa: E402

PAIRS = {"IDENT_HAND": ("identified_hand", "corr_none"), "IDENT_HAND_3KG": ("ih_3kg", "corr_3kg"),
         "IDENT_HAND_5KG": ("ih_5kg", "corr_5kg")}


def boot(v, seed=20261118, n=10000):
    if not v:
        return None
    rng, k = random.Random(seed), len(v)
    m = sorted(statistics.median(v[rng.randrange(k)] for _ in range(k)) for _ in range(n))
    return {"n": k, "median": statistics.median(v), "ci95": [m[int(0.025 * n)], m[int(0.975 * n) - 1]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ext", type=Path, required=True, help="k4_analyze_extensions output incl. corr_* variants")
    ap.add_argument("--friction", type=Path, help="k4_analyze_extensions output for the friction-bound corr_* variants")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ex = json.loads(a.ext.read_text())
    fr = json.loads(a.friction.read_text()) if a.friction else None
    pool = lambda sid: int(sid.split("-")[3])  # noqa: E731  k4-<x>-<tag>-<pool>-...
    res = {}
    for cond, (old, new) in PAIRS.items():
        O, N = ex["variants"][old]["arms"]["primary"], ex["variants"][new]["arms"]["primary"]
        od = {pool(r["source_id"]): r for r in O["rows"]}
        nd = {pool(r["source_id"]): r for r in N["rows"]}
        common = sorted(set(od) & set(nd))
        dur = [100 * (nd[k]["duration"] - od[k]["duration"]) / od[k]["duration"] for k in common]
        flips = {f"{od[k]['MODEL_K4']}->{nd[k]['MODEL_K4']}": 0 for k in common}
        for k in common:
            flips[f"{od[k]['MODEL_K4']}->{nd[k]['MODEL_K4']}"] += 1
        rec = {}
        for tag, X in (("planned_with_curobo_model", O), ("planned_with_corrected_model", N)):
            rec[tag] = {"attempted": X["attempted"], "native_success": X["native_success"], "eligible": X["eligible"],
                        **{c: {k: X[c][k] for k in ("CERTIFIED", "VIOLATED", "UNKNOWN", "coverage")}
                           for c in ("MODEL_K3", "MODEL_K4", "K3_PLUS_K4", "PLANNING_K4")},
                        "rigorous_point_max_util": X["rigorous_point_max_util"],
                        "between_grid512_violation_physical": X.get("between_grid512_violation_physical"),
                        "between_grid512_violation_planning": X.get("between_grid512_violation_planning"),
                        "between_all_samples_violation_planning": X.get("between_all_samples_violation_planning")}
        if fr and new in fr["variants"]:
            F = fr["variants"][new]["arms"]["primary"]
            rec["planned_with_corrected_model"]["MODEL_K4_friction_bound"] = {
                k: F["MODEL_K4"][k] for k in ("CERTIFIED", "VIOLATED", "UNKNOWN", "coverage")}
        rec["paired_on_both_successful"] = len(common)
        rec["MODEL_K4_transitions_old_to_new"] = flips
        rec["duration_pct_slower_corrected_vs_curobo_model"] = boot(dur)
        rec["duration_pct_distribution"] = pct(dur) if dur else None
        res[cond] = rec
        print(cond, json.dumps({t: {c: f'{rec[t][c]["CERTIFIED"]}/{rec[t]["eligible"]}' for c in ("MODEL_K4", "K3_PLUS_K4", "PLANNING_K4")}
                                | {"native_success": f'{rec[t]["native_success"]}/{rec[t]["attempted"]}'}
                                for t in ("planned_with_curobo_model", "planned_with_corrected_model")}),
              "dur", rec["duration_pct_slower_corrected_vs_curobo_model"] and round(rec["duration_pct_slower_corrected_vs_curobo_model"]["median"], 3))
    out = {"scope": "K4_CORRECTED_MODEL_PLANNING_COMPARISON", "judged_under": "identified+URDF hand (+ hypothetical payload)",
           "conditions": res, "hard_stops": ex["hard_stops"] + (fr["hard_stops"] if fr else []),
           "gate": "PASS_SOUNDNESS" if not ex["hard_stops"] and not (fr and fr["hard_stops"]) else "HARD_STOP"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()

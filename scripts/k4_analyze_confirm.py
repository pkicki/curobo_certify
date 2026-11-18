#!/usr/bin/env python3
"""K4-Panda-Confirm150 analysis: hypotheses H1-H8 exactly as preregistered (docs/reports/k4/confirm/PREREGISTRATION.md).

Inputs (runs/k4/confirm): native/generation_summary.json, cert/part_plan_<C>_*.json (planning-model K3+K4),
cert/part_ident_<C>_*.json (identified+hand [+payload] K4), bl/part_bl_<C>_*.json (rigorous bracket + sampled checks),
native_torque_<C>.json, toppra/cert_*.json. Writes a JSON report with every count, CI and test, plus a HARD_STOP list.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_formal_cohort import pct, proportion  # noqa: E402

CONDS = ("CA_CUROBO", "CB_CUROBO_3KG", "CC_CORR_3KG", "CD_CORR_3KG_NI063")
MEANINGFUL = 1e-4
THRESHOLDS = (1e-6, 1e-4, 1e-3, 1e-2)
NATIVE_TOL = 1e-4


def rows_of(pattern):
    return [r for f in sorted(glob.glob(pattern)) for r in json.loads(Path(f).read_text())["rows"]]


def mcnemar_one_sided(b, c):
    """P(X >= b | X ~ Bin(b + c, 1/2)): evidence that the 'b' discordance dominates."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(b, n + 1)) / 2 ** n


def truth(r):
    if r["truth_lb"] > 1:
        return "VIOLATED"
    if r.get("truth_ub") is not None and r["truth_ub"] <= 1 and r.get("truth_ub_verified") is not False:
        return "CERTIFIED"
    return "UNDECIDED"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=ROOT / "runs/k4/confirm")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    D = a.dir
    gen = json.loads((D / "native/generation_summary.json").read_text())
    stops, rep = [], {"scope": "K4_CONFIRM150_ANALYSIS", "preregistration": "docs/reports/k4/confirm/PREREGISTRATION.md",
                      "meaningful_excess": MEANINGFUL, "conditions": {}}
    data = {}
    nonconf = []
    ef = D / "escalation.json"
    escal = json.loads(ef.read_text()) if ef.exists() else {}
    for C in CONDS:
        L = gen["conditions"][C]["ledger"]
        elig = {e["npz_sha256"]: e for e in L if e["selected_eligible"]}
        plan = {r["npz_sha256"]: r for r in rows_of(str(D / f"cert/part_plan_{C}_*.json"))}
        ident = {r["npz_sha256"]: r for r in rows_of(str(D / f"cert/part_ident_{C}_*.json"))}
        bl = {r["npz_sha256"]: r for r in rows_of(str(D / f"bl/part_bl_{C}_*.json"))}
        nat = {r["npz_sha256"]: r for r in json.loads((D / f"native_torque_{C}.json").read_text())["rows"]}
        miss = [s for s in elig if s not in plan or s not in ident or s not in bl or s not in nat]
        if miss:
            stops.append(f"{C}: missing evidence for {len(miss)} outputs")
        rows = []
        for s, e in elig.items():
            if s in miss:
                continue
            p, i, b, n = plan[s], ident[s], bl[s], nat[s]
            t = truth(b)
            excess = b["truth_lb"] - 1
            s32 = n["grids"]["S32768"]["max_util_planning"]
            tgrid = n["grids"]["native_grid_512"]["max_util_planning"]
            # ---- H1 soundness ----
            if b["K4"] == "CERTIFIED" and b["truth_lb"] > 1:
                stops.append(f"H1 {C} {e['source_id']}: verified CERTIFIED but rigorous lb > 1")
            if p["PLANNING_K4"] == "CERTIFIED" and s32 > 1 + NATIVE_TOL:
                stops.append(f"H1 {C} {e['source_id']}: PLANNING_K4 CERTIFIED but native S32768 {s32}")
            if p["MODEL_K4"] == "CERTIFIED" and n["grids"]["S32768"]["max_util_physical"] > 1 + NATIVE_TOL:
                stops.append(f"H1 {C} {e['source_id']}: MODEL_K4 CERTIFIED but native S32768 physical")
            for col in ("MODEL_K3", "MODEL_K4", "PLANNING_K4"):
                if p[col] == "REJECTED_BY_VERIFIER":   # inconclusive at protocol refinement 8; escalated separately
                    key = f"{e['source_id']}|{col}"
                    esc = escal.get(key)
                    nonconf.append({"key": key, "escalation": esc})
                    if esc is None or not (esc.get("python_refine12") and esc.get("compiled_refine12")):
                        stops.append(f"H1 {C} {e['source_id']}: {col} verifier non-confirmation not resolved by escalation")
            if (p["PLANNING_K4"] == "CERTIFIED") != (b["K4"] == "CERTIFIED") and b["K4"] in ("CERTIFIED", "VIOLATED_TORQUE") \
                    and p["PLANNING_K4"] in ("CERTIFIED", "VIOLATED_TORQUE"):
                stops.append(f"H1 {C} {e['source_id']}: Python tm1 ({p['PLANNING_K4']}) vs compiled ({b['K4']}) disagree")
            rows.append({"source_id": e["source_id"], "pool_order": e["pool_order"], "scene": e["scene"], "magnitude": e["magnitude"],
                         "MODEL_K3": p["MODEL_K3"], "MODEL_K4": p["MODEL_K4"], "PLANNING_K4": p["PLANNING_K4"],
                         "K3_PLUS_K4": p["K3_PLUS_K4"], "IDENTIFIED_K4": i["MODEL_K4"], "truth": t, "excess": excess,
                         "lb": b["truth_lb"], "ub": b.get("truth_ub"), "native_S32768": s32, "native_timeopt_grid": tgrid,
                         "duration": p["native_duration_s"], "k4_seconds": b["k4_producer_seconds_1thread"] + b["k4_verifier_seconds_1thread"]})
        data[C] = rows
        n_el = len(rows)
        cnt = lambda col, v: sum(r[col] == v for r in rows)  # noqa: E731
        rep["conditions"][C] = {
            "attempted": len(L), "native_success": sum(e["planner_success"] for e in L), "eligible": n_el,
            "first_k_reached": len(elig) >= 150,
            **{col: {"CERTIFIED": cnt(col, "CERTIFIED"), "VIOLATED": sum(str(r[col]).startswith("VIOLATED") for r in rows),
                     "OTHER": sum(not (r[col] == "CERTIFIED" or str(r[col]).startswith("VIOLATED")) for r in rows),
                     "coverage": proportion(cnt(col, "CERTIFIED"), n_el)}
               for col in ("MODEL_K3", "MODEL_K4", "PLANNING_K4", "K3_PLUS_K4", "IDENTIFIED_K4")},
            "planning_truth": {k: sum(r["truth"] == k for r in rows) for k in ("VIOLATED", "CERTIFIED", "UNDECIDED")},
            "planning_excess_at": {str(th): sum(r["excess"] >= th for r in rows) for th in THRESHOLDS},
            "excess_meaningful": pct([r["excess"] for r in rows if r["excess"] >= MEANINGFUL]) if any(r["excess"] >= MEANINGFUL for r in rows) else None,
            "k4_seconds_1thread": pct([r["k4_seconds"] for r in rows]),
            "duration_s": pct([r["duration"] for r in rows])}
    H = {}
    # H2 coverage
    for C in ("CA_CUROBO", "CC_CORR_3KG"):
        cov = rep["conditions"][C]["MODEL_K4"]["coverage"]
        H[f"H2_{C}"] = {"coverage": cov, "decision": "GO" if (cov["one_sided_95_lower_exact"] or 0) >= 0.95 else "NO-GO"}
    # H3 / H4 model mismatch and fix
    def viol_ident(C):
        return proportion(sum(str(r["IDENTIFIED_K4"]).startswith("VIOLATED") for r in data[C]), len(data[C]))
    h3 = viol_ident("CB_CUROBO_3KG")
    H["H3_model_mismatch_CB"] = {"proportion": h3, "claim_real_risk": (h3["two_sided_95_exact"] or [0])[0] > 0.05}
    h4 = viol_ident("CC_CORR_3KG")
    cb = {r["pool_order"]: r for r in data["CB_CUROBO_3KG"]}
    cc = {r["pool_order"]: r for r in data["CC_CORR_3KG"]}
    common = sorted(set(cb) & set(cc))
    b_ = sum(1 for k in common if str(cb[k]["IDENTIFIED_K4"]).startswith("VIOLATED") and not str(cc[k]["IDENTIFIED_K4"]).startswith("VIOLATED"))
    c_ = sum(1 for k in common if not str(cb[k]["IDENTIFIED_K4"]).startswith("VIOLATED") and str(cc[k]["IDENTIFIED_K4"]).startswith("VIOLATED"))
    H["H4_fix_CC"] = {"proportion": h4, "zero_violations": h4["k"] == 0, "paired_n": len(common),
                      "discordant_CBviol_CCok": b_, "discordant_CBok_CCviol": c_, "p_one_sided": mcnemar_one_sided(b_, c_),
                      "claim": h4["k"] == 0 and mcnemar_one_sided(b_, c_) < 0.05}
    # H5 coarse grid (meaningful planning violations), paired CD vs CC
    cd = {r["pool_order"]: r for r in data["CD_CORR_3KG_NI063"]}
    common = sorted(set(cc) & set(cd))
    m = lambda r: r["excess"] >= MEANINGFUL  # noqa: E731
    b_ = sum(1 for k in common if m(cd[k]) and not m(cc[k]))
    c_ = sum(1 for k in common if not m(cd[k]) and m(cc[k]))
    H["H5_coarse_grid"] = {"CD_rate": proportion(sum(m(r) for r in data["CD_CORR_3KG_NI063"]), len(data["CD_CORR_3KG_NI063"])),
                           "CC_rate": proportion(sum(m(r) for r in data["CC_CORR_3KG"]), len(data["CC_CORR_3KG"])),
                           "paired_n": len(common), "discordant_CD_only": b_, "discordant_CC_only": c_,
                           "p_one_sided": mcnemar_one_sided(b_, c_), "claim": mcnemar_one_sided(b_, c_) < 0.05}
    # H6 dense native sampling agreement (pooled)
    allr = [r for C in CONDS for r in data[C]]
    dec = [r for r in allr if r["truth"] != "UNDECIDED"]
    dis = [r for r in dec if (r["truth"] == "VIOLATED") != (r["native_S32768"] > 1)]
    H["H6_dense_sampling"] = {"decided": len(dec), "undecided": len(allr) - len(dec), "disagreement": proportion(len(dis), len(dec)),
                              "misses": [r["source_id"] for r in dis if r["truth"] == "VIOLATED"],
                              "false_alarms": [r["source_id"] for r in dis if r["truth"] == "CERTIFIED"],
                              "meaningful_misses": [{"source_id": r["source_id"], "excess": r["excess"], "native": r["native_S32768"]}
                                                    for r in dis if r["truth"] == "VIOLATED" and r["excess"] >= MEANINGFUL]}
    # H7 TOPP-RA
    tp = rows_of(str(D / "toppra/cert_*.json"))
    ok = [r for r in tp if r["K4"] != "TOPPRA_FAILED"]
    mv = [r for r in ok if r["truth_lb"] - 1 >= MEANINGFUL]
    pr = proportion(len(mv), len(ok))
    H["H7_toppra"] = {"attempted": len(tp), "toppra_failed": len(tp) - len(ok), "meaningful_violation": pr,
                      "any_violation_K4": sum(r["K4"] == "VIOLATED_TORQUE" for r in ok),
                      "excess_meaningful": pct([r["truth_lb"] - 1 for r in mv]) if mv else None,
                      "duration_vs_timeopt_pct": pct([100 * (r["duration_s"] - r["native_duration_s"]) / r["native_duration_s"]
                                                      for r in ok if r.get("duration_s")]),
                      "claim_systematic": (pr["two_sided_95_exact"] or [0])[0] >= 0.25}
    # H8 enforcement-grid misses
    for C in ("CC_CORR_3KG", "CD_CORR_3KG_NI063"):
        R = data[C]
        mm = [r for r in R if r["excess"] >= MEANINGFUL and r["native_timeopt_grid"] <= 1]
        H[f"H8_{C}"] = {"missed_by_timeopt_grid": proportion(len(mm), len(R)),
                        "excess": pct([r["excess"] for r in mm]) if mm else None}
    rep["hypotheses"] = H
    rep["verifier_nonconfirmations_refine8"] = nonconf
    rep["hard_stops"] = stops
    rep["gate"] = "PASS_SOUNDNESS" if not stops else "HARD_STOP"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(json.dumps({"gate": rep["gate"], "stops": stops[:10],
                      **{k: {kk: vv for kk, vv in v.items() if kk in ("decision", "claim", "claim_real_risk", "claim_systematic", "p_one_sided")}
                         for k, v in H.items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()

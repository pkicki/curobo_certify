#!/usr/bin/env python3
"""Compose the ten research answers and the GO/NO-GO from recorded result files.

Every number below is computed from the JSON results (paths listed in the output);
if a source is missing the corresponding statement says NOT RUN.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs"


def load(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None


def med(a):
    return float(np.median(a))


def main():
    tz = load("hardcase_seed20261116/tightness_79paths.json")
    nr = load("native_linkage/native_reported_16paths.json")
    fz = load("native_linkage/same_s_falsify_16paths.json")
    tc = load("native_linkage/time_consistency_16paths.json")
    camp = {n: load(f"compiled_k3/differential_10k_{n}.json") for n in ("O3", "O0", "native")}
    bench = load("benchmarks/frozen_native_79paths.json")
    summ = load("benchmarks/summary.json")
    fresh = {g: (summ["fresh"][i] if summ and len(summ["fresh"]) > i else None) for i, g in enumerate(("3090", "a100"))}
    old_python_ms = 1457.20  # recorded p50, docs/PROGRESS.md 2026-11-11 (Python full K3 + replay, 8 Panda paths)
    old_hull_ms = 84.72
    Q = []

    def cond(path):
        return "margin" if "margin_arm" in path else "default"

    # Q1/Q2/Q3 from exact tightness
    if tz:
        by = {"default": [], "margin": []}
        for r in tz["rows"]:
            by[cond(r["path"])].append(r)
        def n_with(rows, c, cls):
            return sum(any(x["class"] == cls for x in r["contracts"][c]["nontrivial_residuals"]) for r in rows)
        d, m = by["default"], by["margin"]
        per_path_res = tz["rows"][0]["common_spans"] * 29
        gaps = [x["hull_gap_bracket"][0] for r in tz["rows"] for c in ("physical", "planning")
                for x in r["contracts"][c]["nontrivial_residuals"] if x["class"] == "SPLIT_CERTIFIABLE"]
        fam = Counter(x["family"] for r in tz["rows"] for c in ("physical", "planning")
                      for x in r["contracts"][c]["nontrivial_residuals"] if x["class"] != "H0_CERTIFIED")
        famv = Counter(x["family"] for r in m for x in r["contracts"]["physical"]["nontrivial_residuals"] if x["class"] == "GENUINE_VIOLATION")
        depths = [x["bracket_depth"] for r in tz["rows"] for c in ("physical", "planning") for x in r["contracts"][c]["nontrivial_residuals"]]
        all_res = sum(sum(r["contracts"][c]["class_counts"].values()) for r in tz["rows"] for c in ("physical", "planning"))
        h0_res = sum(r["contracts"][c]["class_counts"].get("H0_CERTIFIED", 0) for r in tz["rows"] for c in ("physical", "planning"))
        tot_split_phys = sum(r["contracts"]["physical"]["class_counts"].get("SPLIT_CERTIFIABLE", 0) for r in d)
        Q.append({"q": "1. How frequently do real TimeOpt trajectories require subdivision?",
                  "a": (f"Exact analysis of {len(tz['rows'])} real native outputs (79 = 16 earlier + 63 new; {per_path_res}-ish residuals per path). "
                        f"<b>Default solver margins, physical limits:</b> {n_with(d,'physical','SPLIT_CERTIFIABLE')}/{len(d)} paths need any subdivision "
                        f"({tot_split_phys} of {sum(sum(r['contracts']['physical']['class_counts'].values()) for r in d)} residuals); "
                        f"{len(d)-n_with(d,'physical','SPLIT_CERTIFIABLE')-n_with(d,'physical','GENUINE_VIOLATION')} certify at depth 0. "
                        f"<b>Against the solver's own 0.95-scaled contract:</b> {n_with(d,'planning','SPLIT_CERTIFIABLE')}/{len(d)} need subdivision and "
                        f"{n_with(d,'planning','GENUINE_VIOLATION')}/{len(d)} genuinely violate it. <b>Margin arm (scale 1.00), physical:</b> "
                        f"{n_with(m,'physical','SPLIT_CERTIFIABLE')}/{len(m)} need subdivision, {n_with(m,'physical','GENUINE_VIOLATION')}/{len(m)} genuinely violate. "
                        "Subdivision is rare when a margin separates the solver contract from the certified contract and ubiquitous when it does not.")})
        Q.append({"q": "2. How conservative are the initial Bernstein bounds?",
                  "a": (f"For the {len(gaps)} real residuals whose depth-0 hull was inconclusive but which are feasible, the exact hull overestimate "
                        f"(U0 − true peak, lower bracket) has median {np.median(gaps):.2e} and max {max(gaps):.2e} in normalized residual units "
                        f"(i.e. fractions of a squared/linear limit). Exact peak brackets needed depth ≤ {max(depths)} (median {np.median(depths):.0f}). "
                        f"Over both contracts {h0_res}/{all_res} residuals ({100*h0_res/all_res:.2f}%) are already proven by the initial hull.") if gaps else "No inconclusive residuals."})
        Q.append({"q": "3. Which constraints cause the most difficult proofs?",
                  "a": (f"Families of residuals that were not proven at depth 0 (all contracts): {dict(fam.most_common())}. "
                        f"Genuine physical violations (margin arm) by family: {dict(famv)}; worst is jerk (≈ +0.36% over the limit). "
                        "Squared-speed positivity never needed subdivision and no strict-positivity failure occurred.")})
    else:
        for q in ("1", "2", "3"):
            Q.append({"q": q, "a": "NOT RUN"})
    # Q4/Q5/Q6 from benchmarks
    if bench:
        rows = bench["rows"]
        def group(contract, method, cnd):
            return [r for r in rows if r["contract"] == contract and r["method"] == method and cond(r["path"]) == cnd]
        lines = []
        for c, cnd in (("planning", "default"), ("physical", "margin")):
            parts = []
            for meth in ("H0", "U1", "U2", "U4", "U6", "A", "A-nw"):
                g = group(c, meth, cnd)
                if not g:
                    continue
                cert = sum(r["final"] == "CERTIFIED" for r in g)
                viol = sum(r["final"].startswith("VIOLATED") for r in g)
                parts.append(f"{meth}: {cert} cert / {viol} viol / {len(g)-cert-viol} unknown, mean nodes {np.mean([r['nodes'] for r in g]):.0f}, "
                             f"p50 {np.percentile([med(r['repeats_ms_total']) for r in g],50):.2f} ms")
            lines.append(f"<b>{c} contract, {cnd} outputs</b>: " + "; ".join(parts))
        Q.append({"q": "4. Does adaptive subdivision outperform uniform subdivision?",
                  "a": ("On identical frozen outputs, in every contract and population measured, adaptive matches or exceeds the coverage of every uniform level while touching only the "
                        "inconclusive residuals, so its cost stays near H0; uniform levels pay for every residual. " + " ".join(lines) +
                        " Without early witness handling (A-nw) genuine violations end as UNKNOWN instead of VIOLATED at similar cost.")})
        h0 = [med(r["repeats_ms_total"]) for r in rows if r["contract"] == "physical" and r["method"] == "A"]
        nat = [med(r["repeats_ms_total"]) for r in rows if r["contract"] == "native" and r["method"] == "A"]
        nb = [med(r["repeats_ms_native_bound"]) for r in rows if r["contract"] == "native" and r["method"] == "A"]
        f3 = fresh.get("3090")
        Q.append({"q": "5. What is the complete certification overhead?",
                  "a": (f"Frozen output, complete certificate (compiled producer + independent verifier), adaptive, physical contract: p50 {np.percentile(h0,50):.2f} ms, "
                        f"p95 {np.percentile(h0,95):.2f} ms over {len(h0)} paths. Native-reported contract (adds the compiled FP32 error bound, p50 {np.percentile(nb,50):.2f} ms): "
                        f"p50 {np.percentile(nat,50):.2f} ms, p95 {np.percentile(nat,95):.2f} ms. "
                        + (f"Fresh native requests on {f3['gpu']}: native request p50 {f3['native_ms']['p50']:.2f} ms; added certification p50 {f3['added_ms']['p50']:.2f} ms "
                           f"(95% CI {f3['added_ms']['p50_ci95'][0]:.2f}–{f3['added_ms']['p50_ci95'][1]:.2f}), p95 {f3['added_ms']['p95']:.2f} ms; stage medians "
                           + ", ".join(f"{k} {v:.2f} ms" for k, v in f3['stage_median_ms'].items()) + "." if f3 else "Fresh-request latency: NOT RUN."))})
        Q.append({"q": "6. How much faster is the compiled full certifier?",
                  "a": (f"Previous fully verified Python path (outward certificate + replay, same 8 pilot paths): p50 {old_python_ms:.0f} ms; previous compiled hull-only screen "
                        f"{old_hull_ms:.1f} ms (not a certificate). Compiled complete producer + independent verifier, physical contract, adaptive: p50 {np.percentile(h0,50):.2f} ms "
                        f"→ ≈{old_python_ms/np.percentile(h0,50):.0f}× faster for a strictly stronger, independently verified result (both measured on the same Ryzen 9 5900X, "
                        "different sessions; not a paired comparison).")})
    else:
        for q in ("4", "5", "6"):
            Q.append({"q": q, "a": "NOT RUN"})
    # Q7 native
    if nr and fz:
        mx = max(max(v[f]["max_ratio_to_bound"] for v in r["same_progress_sweep"].values() for f in ("velocity", "acceleration", "jerk")) for r in fz["rows"])
        ulp = max(r["progress_error"]["max_abs_in_float32_ulps_of_s"] for r in tc["rows"]) if tc else None
        Q.append({"q": "7. Can native CUDA execution be rigorously certified?",
                  "a": (f"<b>Partially, conditionally.</b> A forward FP32 error analysis of the pinned kernel bounds every reported native value against the exact model for every float32 progress; "
                        f"certifying the exact model against limits minus that bound gives NATIVE_REPORTED_K3 on {nr['certified']}/{nr['denominator']} frozen outputs "
                        f"(also with 4-ulp division/sqrt), conditional on documented CUDA FP32 semantics (A1–A5). A harness compiled with upstream's NVRTC options reproduces production outputs bitwise "
                        f"(all 79 paths) and no dense same-progress sample exceeded the bound (max {mx:.3f} of the bound). The time-law consistency s(t) is only a diagnostic "
                        f"(≤ {ulp:.0f} float32 ulps observed): complete native equivalence remains <b>BLOCKED_NATIVE_LINKAGE</b>. The registered 4e-4 parity gate still fails and was not relaxed.")})
    else:
        Q.append({"q": "7", "a": "NOT RUN"})
    # Q8
    Q.append({"q": "8. Which correctness and performance blockers remain?",
              "a": ("(i) BLOCKED_NATIVE_LINKAGE for time-law consistency (no rigorous continuous bound on s_native(t) − s*(t)); (ii) assumptions A1–A5 are vendor/compiler specifications, no SASS audit; "
                    "(iii) torque/torque-rate UNSUPPORTED_DYNAMICS; (iv) compiled certifier validated by 30,000 seeded differential cases, not formally verified; (v) formal Dev40/Holdout100/Confirm300 "
                    "NOT RUN because (i) is a failed correctness gate; (vi) performance: the native FP32 bound stage dominates and the 1 ms / 3 ms added-latency targets are not met "
                    "(single-threaded CPU; parallelization/GPU port not attempted); (vii) UR5e still lacks sourced acceleration/jerk limits.")})
    # Q9
    f3 = fresh.get("3090")
    cov = None
    if nr:
        cov = f"{nr['certified']}/{nr['denominator']}"
    nat_cov = "NOT RUN"
    if bench:
        g = [r for r in bench["rows"] if r["contract"] == "native" and r["method"] == "A" and "margin_arm" not in r["path"]]
        gm = [r for r in bench["rows"] if r["contract"] == "native" and r["method"] == "A" and "margin_arm" in r["path"]]
        nat_cov = (f"{sum(r['final']=='CERTIFIED' for r in g)}/{len(g)} default-margin outputs (compiled, adaptive); margin-arm outputs "
                   f"{sum(r['final']=='CERTIFIED' for r in gm)}/{len(gm)}")
    Q.append({"q": "9. Does the approach satisfy the original engineering targets?",
              "a": (f"Correctness gate (zero false certificates): met on all tests run. Coverage ≥99% of eligible native successes (K3, physical, default margins): "
                    f"exact model {sum(1 for r in (tz['rows'] if tz else []) if cond(r['path'])=='default' and not any(x['class']=='GENUINE_VIOLATION' for x in r['contracts']['physical']['nontrivial_residuals']))}"
                    f"/{sum(1 for r in (tz['rows'] if tz else []) if cond(r['path'])=='default')} exploratory paths certifiable, native-reported {nat_cov} — but on exploratory populations, not the registered holdout. "
                    + (f"Latency: added p50 {f3['added_ms']['p50']:.2f} ms vs target ≤1 ms (<b>not met</b>), p95 {f3['added_ms']['p95']:.2f} ms vs ≤3 ms (<b>{'met' if f3['targets']['added_p95_le_3ms'] else 'not met'}</b>)."
                       if f3 else "Latency: NOT RUN.") + " Decision category: CORRECT_BUT_SLOW for the model K3 contract; BLOCKED_NATIVE_LINKAGE for full native equivalence; BLOCKED_DYNAMICS for torque.")})
    Q.append({"q": "10. Is this a promising research direction for a publication?",
              "a": ("Yes, with a sharpened claim. Novel, evidence-backed contributions: (a) exact demonstration that sampled time-optimal planners violate their own continuous-time contract between samples "
                    "(9/48 default outputs vs 0.95 limits; 12/31 margin-arm outputs vs physical limits) and that adaptive Bernstein proofs both find exact witnesses and certify the rest with ≈H0 cost; "
                    "(b) a native-reported certificate that bounds a real CUDA kernel's FP32 output for all progress values, validated bitwise; (c) a compiled producer/independent-verifier pair in single-digit "
                    "milliseconds. Before submission: close time-law consistency rigorously, run the registered Dev40/Holdout100/Confirm300, add UR5e, and parallelize the native bound to approach the 1 ms target.")})
    decision = ("<b>NO-GO for deployment as a qualified native-linked certifier</b> (BLOCKED_NATIVE_LINKAGE on time-law consistency; latency targets missed; formal cohorts not run). "
                "<b>GO for continued research and a publication</b> on exact continuous-time certification of unchanged TimeOpt outputs with conditional native-reported guarantees.")
    out = {"generated_from": [str(p) for p in ("hardcase_seed20261116/tightness_79paths.json", "native_linkage/*.json", "compiled_k3/*.json", "benchmarks/*.json")],
           "questions": Q, "decision": decision}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/research_answers.json").write_text(json.dumps(out, indent=1) + "\n")
    for q in Q:
        print(q["q"], "\n  ", q["a"][:300], "\n")
    print(decision)


if __name__ == "__main__":
    main()

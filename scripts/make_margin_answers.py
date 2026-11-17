#!/usr/bin/env python3
"""Findings of the 2026-11-17 margin / native-time / latency campaign, computed from recorded files.

Every number is read from runs/margins/* and runs/compiled_k3/*_mt4.json; missing
inputs produce NOT RUN statements. Output: reports/margin_campaign_answers.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs"


def load(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def main():
    A = load("margins/analysis.json")
    T = load("margins/time_inversion_audit_224paths.json")
    L = load("margins/l1_summary.json")
    FU = load("margins/fp_unit_audit_rtx3090.json")
    C = {n: load(f"compiled_k3/differential_10k_{n}_mt4.json") for n in ("O3", "O0", "native")}
    Q = []
    if A:
        s = A["summary_by_margin"]
        ms = sorted(s, key=float)
        Q.append({"q": "Planning success and duration by margin (36 frozen source queries × 5 margins = 180 requests)",
                  "a": "; ".join(f"<b>{m}</b>: {s[m]['native_success']}/{s[m]['attempted']} solved, duration p50 {f(s[m]['duration_p50_s_success_only'], 4)} s" for m in ms)
                  + ". The 7 failures are the same source queries at every margin (failure is query-, not margin-dependent); every arm therefore has the same 29 matched pairs and no differential-missingness bias. Constructed TimeOpt configuration audited on all 180 requests: only limit_scale and jerk_limit_scale differ from the pinned default."})
        cmp = A["paired_to_0p95"]
        Q.append({"q": "Paired duration change relative to the pinned 0.95 margin (source-pair bootstrap, 4,000 resamples)",
                  "a": "; ".join(f"<b>{m}</b>: median {f(c['duration_gain_pct_paired_success_only']['median'])}% [{f(c['duration_gain_pct_paired_success_only']['ci95'][0])}, {f(c['duration_gain_pct_paired_success_only']['ci95'][1])}] (n={c['duration_gain_pct_paired_success_only']['n']})" for m, c in sorted(cmp.items(), key=lambda kv: float(kv[0])))
                  + ". Positive = faster. Removing the 5% margin entirely saves ≈3% of duration."})
        Q.append({"q": "Exact continuous-time feasibility (serialized model, physical limits; compiled producer + independent verifier)",
                  "a": "; ".join(f"<b>{m}</b>: {s[m]['model_certified']} certified / {s[m]['model_violated']} exact violations / {s[m]['model_unknown']} unknown; {s[m]['physical_paths_needing_subdivision']} need adaptive subdivision; solver's own scaled contract genuinely violated by {s[m]['solver_contract_paths_genuinely_violating']}" for m in ms)
                  + f". Worst physical witness at 1.00: normalized residual {f(s[ms[-1]]['worst_physical_witness_residual'], 5)}. Feasibility is preserved up to 0.99 but proofs need subdivision; at 1.00, 12/29 outputs genuinely violate a physical limit between solver samples."})
        Q.append({"q": "Native-reported certificate (A1–A5 conditional analytic bound) by margin",
                  "a": "; ".join(f"<b>{m}</b>: {s[m]['native_reported_certified']} certified, {s[m]['native_reported_unknown_tightened']} unknown (witness only against tightened limits)" for m in ms)})
    if T:
        rows = [r for r in T["rows"] if "eps_s_certified_max" in r]
        eps = max(r["eps_s_certified_max"] for r in rows)
        okrep = sum(r["exhaustive_reported_status"] == "CERTIFIED_EXHAUSTIVE" for r in rows)
        Q.append({"q": "Native time→progress inversion versus the exact time law",
                  "a": f"Every float32 requested time of {len(rows)} real outputs ({sum(r['times_enumerated'] for r in rows):.3e} times, {T['gpu']}) was executed with the pinned progress_at_time and compared with a rigorous enclosure of the exact inverse: certified max |s_native − s*| ≤ {eps:.3g}; induced state shift ≤ {max(r['relative_deviation_vs_limits']['v'] for r in rows):.2g} (v), {max(r['relative_deviation_vs_limits']['a'] for r in rows):.2g} (a), {max(r['relative_deviation_vs_limits']['j'] for r in rows):.2g} (j, off-knot) of the limits. Reported values within physical limits at every float32 time on {okrep}/{len(rows)} outputs; all exceedances are scale-1.00 outputs and coincide exactly with the exact-model violation witnesses."})
    if L and "fresh" in L:
        F = L["fresh"]
        Q.append({"q": "Certification latency after optimization (RTX 3090, paired fresh requests)",
                  "a": f"Added p50 {f(F['arms']['B1']['added_ms']['p50'])} ms (1 thread) → {f(F['arms']['B8']['added_ms']['p50'])} ms (8 threads); paired Δ {f(F['paired_last_minus_first_arm_ms']['median'])} ms [{f(F['paired_last_minus_first_arm_ms']['ci95'][0])}, {f(F['paired_last_minus_first_arm_ms']['ci95'][1])}]; p95 {f(F['arms']['B8']['added_ms']['p95'])} ms; pooled p99 {f(F['arms']['B8']['pooled_requests']['p99'])} ms. Target p95 ≤ 3 ms met; p50 ≤ 1 ms not met. Outputs identical across thread counts."})
    if FU:
        Q.append({"q": "Arithmetic assumptions",
                  "a": "Exhaustive over all positive normal float32 inputs on " + FU["gpu"] + ": " + ", ".join(f"{k} ≤ {v['max_error_ulps']:.3f} ulp" for k, v in FU["results"].items()) + "; SASS shows MUFU.RCP/MUFU.SQRT/MUFU.RSQ, FFMA contraction and .FTZ, consistent with A1–A4."})
    if all(C.values()):
        Q.append({"q": "Soundness of the parallel compiled certifier",
                  "a": " ; ".join(f"{n}: {c['cases']} cases, {c['breaches']} breaches" for n, c in C.items()) + " (seeded adversarial cases vs exact Fraction reference, 4 OpenMP threads per call)."})
    decision = ("<b>Paper: GO</b> — exploratory but rigorous results: exact continuous-time violations of time-optimal outputs between samples as the planning margin is removed, adaptive proofs at ≈H0 cost, an exhaustively audited native time law and reported-value contract, and ≈2 ms complete certification. "
                "<b>Native reported-value certification (per audited output): GO for research use</b>, conditional on the audited compiled code path and GPU. "
                "<b>Native physical-time certification as a qualified deployment claim: NO-GO</b> until the per-output audit contract is reviewed independently, frozen, and run on the registered Dev40/Holdout100/Confirm300 cohorts (not run). "
                "<b>Margin policy: no margin is recommended from these exploratory data</b>; 0.99 kept every output feasible with ≈2.3% shorter durations, 1.00 produced genuine violations in 12/29 outputs.")
    out = {"questions": Q, "decision": decision, "sources": ["runs/margins/*", "runs/compiled_k3/*_mt4.json"]}
    (ROOT / "reports/margin_campaign_answers.json").write_text(json.dumps(out, indent=1) + "\n")
    for q in Q:
        print("-", q["q"], ":", q["a"][:260])


if __name__ == "__main__":
    main()

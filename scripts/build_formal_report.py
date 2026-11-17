#!/usr/bin/env python3
"""Build the offline interactive formal-qualification report (single self-contained HTML).

Inputs are the committed/ignored REAL evidence only (no synthetic numbers):
  runs/formal_panda/<cohort>/{analysis.json, t1_v2.json, tightness.json, fresh_rtx3090.json, native/*.npz}
  docs/reports/formal_panda/FREEZE.json, docs/reports/t1_v2_reissue_224paths_summary.json,
  runs/margins/time_inversion_audit_224paths.json (withdrawn v1 eps, shown only as the correction),
  optional A100 replication JSONs.
Per eligible output the compiled producer is re-run (physical contract, frozen policy A) only to
extract the proof-tree leaves for display; its status must equal the recorded benchmark status.
Native curves are the saved S8193 production samples: SAMPLED DIAGNOSTICS, never proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify import compiled_k3 as K  # noqa: E402

COHORTS = ("Panda-Dev40", "Panda-Holdout100", "Panda-Confirm300")
FAM = ("positive_x", "velocity", "acceleration_plus", "acceleration_minus", "jerk")


def r6(x):
    return None if x is None else float(f"{x:.6g}")


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def curves(npz: Path, n=256):
    with np.load(npz) as z:
        st = z["diagnostic_8193_states"].astype(np.float64)
        t = z["diagnostic_8193_times"].astype(np.float64)
        lim = [np.minimum(np.abs(z[k][0]), np.abs(z[k][1])).astype(np.float64)
               for k in ("velocity_limits", "acceleration_limits", "jerk_limits")]
    idx = np.linspace(0, len(t) - 1, n).round().astype(int)
    out = {"t": [round(float(x), 5) for x in t[idx]]}
    for fi, f in enumerate(("v", "a", "j")):
        u = np.max(np.abs(st[fi + 1]) / lim[fi][None, :], axis=1)
        # keep the per-window maximum so peaks are not lost by decimation
        w = [float(u[max(0, i - 4):i + 5].max()) for i in idx]
        out[f] = [round(x, 4) for x in w]
    return out


def proof_tree(P, npz: Path):
    with np.load(npz) as z:
        lim = {n: np.minimum(np.abs(z[k][0]), np.abs(z[k][1])).astype(np.float64)
               for n, k in (("velocity", "velocity_limits"), ("acceleration", "acceleration_limits"), ("jerk", "jerk_limits"))}
        r = P.certify(z["effective_geometry"], z["timing_controls"], z["requested_start"][0], z["requested_goal"][0], lim,
                      max_depth=12, max_nodes=8192)
    L = r["leaves"]
    split = {}
    for lf in L:
        if lf["depth"] > 0:
            key = f"{int(lf['span'])}/{FAM[int(lf['family'])]}/{int(lf['joint'])}"
            split.setdefault(key, []).append([int(lf["depth"]), int(lf["bits"]), r6(float(lf["upper"]))])
    worst = {}
    for lf in L:
        f = FAM[int(lf["family"])]
        worst[f] = max(worst.get(f, -1e300), float(lf["upper"]))
    return r["status"], {"leaves": int(len(L)), "nodes": int(r["nodes"]), "max_depth": int(L["depth"].max()) if len(L) else 0,
                         "split": split, "worst_upper": {k: r6(v) for k, v in worst.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/formal_qualification_report.html")
    ap.add_argument("--a100", type=Path, nargs="*", default=[])
    a = ap.parse_args()
    data = {"cohorts": {}, "paths": [], "ledger": []}
    with tempfile.TemporaryDirectory() as d:
        K.build("k3_certifier.cpp", Path(d) / "c.so")
        P = K.Producer(Path(d) / "c.so")
        for c in COHORTS:
            base = ROOT / "runs/formal_panda" / c
            an = load(base / "analysis.json")
            if an is None:
                data["cohorts"][c] = {"status": "NOT_RUN"}
                continue
            t1 = {r["source_npz_sha256"]: r for r in load(base / "t1_v2.json")["rows"]}
            tight = {r["source_npz_sha256"]: r for r in load(base / "tightness.json")["rows"]}
            fresh = load(base / "fresh_rtx3090.json")
            cs = {k: an[k] for k in ("gate", "hard_stops", "first_k", "paired_duration_0p99_vs_0p95",
                                     "fresh_latency_primary", "frozen_output_methods_primary", "query_manifest_sha256")}
            cs["arms"] = an["arms"]
            if fresh:
                cs["fresh_per_query"] = [{"id": r["source_id"], "native": [round(x, 3) for x in r["native_ms"]],
                                          "B8": [round(x, 3) for x in r["added_ms_by_arm"]["B8"]],
                                          "B1": [round(x, 3) for x in r["added_ms_by_arm"]["B1"]],
                                          "status": r["repeats"][0]["B8"]["status"], "digests": r["distinct_output_digests"]}
                                         for r in fresh["rows"]]
            note = ROOT / "docs/reports/formal_panda" / c / "latency_contamination_note.json"
            if note.exists():
                cs["latency_note"] = json.loads(note.read_text())
            data["cohorts"][c] = cs
            for arm, rows in an["rows"].items():
                for r in rows:
                    data["ledger"].append({"cohort": c, "arm": arm, **{k: r.get(k) for k in (
                        "source_id", "pair_id", "pool_order", "scene", "magnitude", "native_status", "eligible",
                        "MODEL_K3", "NATIVE_REPORTED_K3", "T1", "T1_reported", "failure_reason")}})
                    if not r.get("eligible"):
                        continue
                    sha = r["npz_sha256"]
                    npz = base / "native" / f"{r['source_id']}.npz"
                    st, tree = proof_tree(P, npz)
                    if st != r["MODEL_K3"] and not (st == "VIOLATED" and r["MODEL_K3"] == "VIOLATED"):
                        raise SystemExit(f"proof replay status {st} != recorded {r['MODEL_K3']} for {r['source_id']}")
                    tr = t1[sha]
                    tg = tight[sha]["contracts"]["physical"]
                    data["paths"].append({
                        "id": r["source_id"], "cohort": c, "arm": arm, "scene": r["scene"], "magnitude": r["magnitude"],
                        "sha": sha, "duration": r["native_duration_s"], "MODEL_K3": r["MODEL_K3"],
                        "NATIVE_REPORTED_K3": r["NATIVE_REPORTED_K3"], "T1": tr["status"], "T1_reported": tr["reported_status"],
                        "eps": tr["eps_s_certified_max"], "eps_at": tr["at_time"], "t1_s": round(tr["elapsed_s"], 3),
                        "t1_stages": {k: round(v, 3) for k, v in tr["stages_s"].items()}, "times": tr["times_enumerated"],
                        "reported": {f: {"u": r6(tr["reported"][f]["max_utilization"]), "t": tr["reported"][f]["at_time"],
                                         "joint": tr["reported"][f]["joint"]} for f in ("v", "a", "j")},
                        "reldev": {k: r6(v) for k, v in tr["relative_deviation_vs_limits"].items()},
                        "hull0": {k: r6(v) for k, v in tg["max_initial_hull_upper"].items()},
                        "nontrivial": [{"span": x.get("span"), "family": x.get("family"), "joint": x.get("joint"),
                                        "class": x["class"], "U0": r6(x.get("U0")), "Lstar": r6(x.get("peak_lower")),
                                        "Ustar": r6(x.get("peak_upper")), "depth": x.get("bracket_depth"), "s": x.get("s_interval")}
                                       for x in tg["nontrivial_residuals"]],
                        "sampled_S32768": r6(tg["sampled_max_utilization_S32768"]) if not isinstance(tg["sampled_max_utilization_S32768"], dict)
                        else {k: r6(v) for k, v in tg["sampled_max_utilization_S32768"].items()},
                        "proof": tree, "curves": curves(npz), "ms_model": r6(r["MODEL_K3_ms_p50"]),
                        "ms_native": r6(r["NATIVE_REPORTED_ms_p50"])})
    fz = load(ROOT / "docs/reports/formal_panda/FREEZE.json")
    data["freeze"] = {k: fz[k] for k in ("protocol", "population_flag", "original_dual_robot_protocol", "git_head_at_freeze",
                                          "upstream_curobo_sha", "cohorts", "acceptance", "policy", "opened_cohorts",
                                          "candidate_designation", "reproducibility", "t1_binary_audit",
                                          "compiled_k3_builds", "a100_replication_preregistered")
                      if k in fz}
    re = load(ROOT / "docs/reports/t1_v2_reissue_224paths_summary.json")
    v1 = {r["source_npz_sha256"]: r["eps_s_certified_max"] for r in
          load(ROOT / "runs/margins/time_inversion_audit_224paths.json")["rows"]}
    data["reissue"] = {k: re[k] for k in re if k != "rows"} | {
        "pairs": [[v1[r["source_npz_sha256"]], r["eps_s_certified_max"]] for r in re["rows"]]}
    data["a100"] = [load(p) for p in a.a100]
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    tpl = (ROOT / "web/formal_report_template.html").read_text()
    html = tpl.replace("/*__DATA__*/null", payload)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html)
    print(f"{a.out} {len(html)/1e6:.2f} MB sha256 {hashlib.sha256(html.encode()).hexdigest()} paths={len(data['paths'])}")


if __name__ == "__main__":
    main()

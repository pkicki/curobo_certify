#!/usr/bin/env python3
"""Exhaustive per-output float32 time-law audit, version 2 (scope T1_AUDITED_FLOAT32_TIMES).

For each frozen native output and EVERY float32 requested time t in [+0, T_native]
(T_native = native time[intervals], the production non-tail domain; all bit patterns
0x00000000..bits(T_native), i.e. +0, positive subnormals and normals, endpoint included):

 A. production identity: the UPSTREAM production launcher
    (curobo trajectory_timeopt.sample -> sample_with_progress_kernel, compiled by cuRobo's own
    cache/flags/source list) and the audit harness (progress_at_time + mirrored tail) are run
    on the same times; all 4 x dof output words must be bitwise identical (count reported).
    The production launch is first checked to reproduce the saved S8193/S32768 production
    diagnostics bit for bit (argument layout control).
 B. reported values: max |v|/v_lim, |a|/a_lim, |j|/j_lim of the PRODUCTION outputs over all
    times (decision |x| <= lim made exactly in float32), joint/time argmax, and the number of
    non-finite q/v/a/j words.
 C. time inversion: certified bound on |s_native(t) - s*(t)| against a rigorous CPU
    enclosure of the exact time law (time_law_grid), fail-closed kernel inverse_audit_v2
    (NaN/out-of-range progress, empty bracket, harness-vs-reference progress mismatch all
    counted; processed-thread count must equal the enumerated domain size).

Additionally the inversion (C only) is enumerated on (T_native, T_exact_hi] when the native duration
is shorter than the certified exact-model duration (q proximity in the tail window).
Excluded (documented, production semantics): -0.0 and negative times (production tail: q at clamped
progress, v=a=j=0), t > max(T_native, T_exact_hi) (s* = 1, production tail v=a=j=0 exactly), NaN/Inf
(production outputs NaN). Reported v/a/j for t > T_native are exactly 0 by upstream source
(timeopt_path_kernel.cuh, `tail && k > 0 ? 0.0f`), not by enumeration.
Offline cost is recorded per stage; it is NOT part of the online certification latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from curobo_certify import time_law_grid as TG  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402
from native_same_s_falsify import HARNESS, native_inputs, run_progress  # noqa: E402
from native_time_inversion_audit import fup, lipschitz  # noqa: E402

AUDIT = ROOT / "src/curobo_certify/native/time_inverse_audit_v2.cuh"
SCOPE = "T1_AUDITED_FLOAT32_TIMES_V2"


def f32bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits2f(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class AuditV2:
    def __init__(self, torch, device):
        from curobo._src.context import get_runtime
        from curobo._src.curobolib.backends.cuda_core_backend import trajectory_timeopt as TT
        from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import TrajectoryKernelCfg
        self.torch, self.device, self.TT = torch, device, TT
        cfg = TrajectoryKernelCfg()
        self.cache = get_runtime().get_cuda_core_cache()
        self.flags = cfg.get_compile_flags()
        sources = [cfg.kernel_dir / "timeopt" / "timeopt_common.cuh",
                   cfg.kernel_dir / "timeopt" / "timeopt_path_kernel.cuh", HARNESS, AUDIT]
        prod = [cfg.kernel_dir / "timeopt" / "timeopt_path_kernel.cuh", cfg.kernel_dir / "timeopt" / "timeopt_path_helpers.cuh"]
        self.source_hashes = {p.name: sha(p) for p in sources + prod}
        self.kernels, self.qualified = {}, {}
        for short in ("inverse_audit_v2", "progress_chunk_v2"):
            name = "curobo_certify_harness_v2::" + short
            self.kernels[short] = self.cache.get_or_compile_kernel(
                source_files=sources, kernel_name=name, include_dirs=cfg.get_include_dirs(), compile_flags=self.flags)
            self.qualified[short] = name

    def launch(self, name, n, *args):
        from cuda.core import LaunchConfig
        from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
        stream = self.cache.get_stream_wrapper(self.torch.cuda.current_stream(self.device))
        launch_kernel(self.qualified[name], stream, LaunchConfig(grid=(n + 255) // 256, block=256), self.kernels[name],
                      *[x.data_ptr() if isinstance(x, self.torch.Tensor) else x for x in args])

    def production(self, inp, times_1d, start_offset=1):
        """Upstream production sampler (sample_with_progress_kernel via cuRobo's own launcher)."""
        torch = self.torch
        n = times_1d.shape[0]
        dof = inp["dof"]
        out = torch.empty((4, 1, n, dof), dtype=torch.float32, device=self.device)
        prog = torch.empty((1, n), dtype=torch.float32, device=self.device)
        self.TT.sample(inp["controls"][None].contiguous(), inp["start"][None].contiguous(), inp["goal"][None].contiguous(),
                       inp["timing"][None].contiguous(), inp["polynomial"], inp["nodes"], inp["weights"],
                       inp["time"][None].contiguous(), times_1d[None].contiguous(),
                       torch.ones(1, dtype=torch.bool, device=self.device),
                       torch.zeros(1, dtype=torch.bool, device=self.device), out, prog, start_offset)
        return out, prog


def audit_path(A: AuditV2, path: Path, log2_cells: int, chunk: int, check_samples: int) -> dict:
    torch, dev = A.torch, A.device
    stages = {}
    t_all = time.perf_counter()
    raw = path.read_bytes()
    with np.load(path, allow_pickle=False) as z:
        arr = {k: z[k].copy() for k in z.files}
    row = {"path": path.stem, "source_npz_sha256": hashlib.sha256(raw).hexdigest()}
    t0 = time.perf_counter()
    grid = TG.build_grid(arr["timing_controls"], log2_cells)
    stages["grid_s"] = time.perf_counter() - t0
    row.update({"grid_status": grid.status, "grid_cells": grid.n_cells})
    if grid.status != "BOUNDED_EXACT_MODEL":
        row["status"] = "FAILED_TO_AUDIT_GRID"
        return row
    tab_sha = hashlib.sha256(b"".join(np.ascontiguousarray(v).tobytes() for v in
                                      (grid.T_lo, grid.T_hi, grid.fmin, grid.fmax))).hexdigest()
    inp = native_inputs(arr, torch, dev)
    # --- A0: production argument-layout control on saved production diagnostics -------------
    t0 = time.perf_counter()
    saved_ok = {}
    for key in ("diagnostic_8193", "diagnostic_32768"):
        ts = torch.as_tensor(arr[key + "_times"].astype(np.float32), device=dev)
        out, _ = A.production(inp, ts)
        torch.cuda.synchronize(dev)
        saved_ok[key] = bool(np.array_equal(out[:, 0].cpu().numpy().view(np.uint32),
                                            arr[key + "_states"].astype(np.float32).view(np.uint32)))
    stages["saved_production_replay_s"] = time.perf_counter() - t0
    row["production_replays_saved_diagnostics_bitwise"] = saved_ok
    # --- enumeration ---------------------------------------------------------------------------
    tab = {k: torch.as_tensor(np.ascontiguousarray(v), device=dev) for k, v in
           (("T_lo", grid.T_lo), ("T_hi", grid.T_hi), ("fmin", grid.fmin), ("fmax", grid.fmax))}
    Tn = float(arr["native_timing_time_grid"][-1])
    if not (np.isfinite(Tn) and Tn > 0):
        row["status"] = "FAILED_TO_AUDIT_DURATION"
        return row
    last = f32bits(Tn)
    total = last + 1
    dof = inp["dof"]
    lim32 = {n: torch.as_tensor(np.minimum(np.abs(arr[k][0]), np.abs(arr[k][1])).astype(np.float32), device=dev)
             for n, k in (("v", "velocity_limits"), ("a", "acceleration_limits"), ("j", "jerk_limits"))}
    lim64 = {n: v.double() for n, v in lim32.items()}
    keys = {k: torch.zeros(1, dtype=torch.int64, device=dev) for k in ("any", "ahead", "behind")}
    counters = torch.zeros(5, dtype=torch.int64, device=dev)
    times_buf = torch.empty(chunk, dtype=torch.float32, device=dev)
    hout = torch.empty(4 * chunk * dof, dtype=torch.float32, device=dev)
    hprog = torch.empty(chunk, dtype=torch.float32, device=dev)
    mismatch_words = 0
    nonfinite = 0
    exceed = {"v": 0, "a": 0, "j": 0}
    best = {f: (-1.0, None, None) for f in ("v", "a", "j")}
    chunks = 0
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    start = 0
    while start < total:
        cnt = min(chunk, total - start)
        A.launch("progress_chunk_v2", cnt, inp["controls"], inp["start"], inp["goal"], inp["timing"], inp["polynomial"],
                 inp["nodes"], inp["weights"], inp["time"], np.uint32(start), np.uint32(cnt), times_buf, hout, hprog,
                 np.int32(dof), np.int32(inp["knots"]), np.int32(inp["intervals"]), np.int32(1))
        ts = times_buf[:cnt]
        pout, _ = A.production(inp, ts)
        h = hout[: 4 * cnt * dof].view(4, cnt, dof)
        p = pout[:, 0]
        mismatch_words += int((p.view(torch.int32) != h.view(torch.int32)).sum().item())
        nonfinite += int((~torch.isfinite(p)).sum().item())
        for fi, f in ((1, "v"), (2, "a"), (3, "j")):
            av = p[fi].abs()
            exceed[f] += int((av > lim32[f]).sum().item())
            r = (av.double() / lim64[f])
            m = int(torch.argmax(r).item())
            val = float(r.view(-1)[m].item())
            if val > best[f][0]:
                best[f] = (val, float(ts[m // dof].item()), m % dof)
        A.launch("inverse_audit_v2", cnt, inp["polynomial"], inp["timing"], inp["nodes"], inp["weights"], inp["time"],
                 np.int32(inp["intervals"]), np.uint32(start), np.uint32(cnt), hprog, tab["T_lo"], tab["T_hi"],
                 tab["fmin"], tab["fmax"], np.int32(log2_cells), keys["any"], keys["ahead"], keys["behind"], counters)
        torch.cuda.synchronize(dev)  # surfaces asynchronous launch errors per chunk
        start += cnt
        chunks += 1
    # Beyond-duration window (T_native, T_exact]: when the native duration is SHORTER than the exact
    # model duration, production reports the tail (v=a=j=0, q at the native progress) while the
    # reference q*(t) is still moving. The time inversion (q proximity) is audited there too, up to
    # the first float32 >= the certified upper bound of T_exact (s* = 1 from there on).
    T_exact_hi = float(grid.T_hi[-1])
    stop_bits = f32bits(float(np.float32(T_exact_hi))) if np.float32(T_exact_hi) >= T_exact_hi else \
        f32bits(float(np.nextafter(np.float32(T_exact_hi), np.float32(np.inf))))
    beyond_total = max(0, stop_bits - last)
    beyond_counters = torch.zeros(5, dtype=torch.int64, device=dev)
    bstart = last + 1
    while bstart <= stop_bits:
        cnt = min(chunk, stop_bits - bstart + 1)
        A.launch("inverse_audit_v2", cnt, inp["polynomial"], inp["timing"], inp["nodes"], inp["weights"], inp["time"],
                 np.int32(inp["intervals"]), np.uint32(bstart), np.uint32(cnt), np.uint64(0), tab["T_lo"], tab["T_hi"],
                 tab["fmin"], tab["fmax"], np.int32(log2_cells), keys["any"], keys["ahead"], keys["behind"],
                 beyond_counters)
        torch.cuda.synchronize(dev)
        bstart += cnt
    stages["gpu_enumeration_s"] = time.perf_counter() - t0
    cnts = [int(c) for c in counters.cpu().tolist()]
    bcnts = [int(c) for c in beyond_counters.cpu().tolist()]

    def unpack(t):
        v = int(t.item()) & 0xFFFFFFFFFFFFFFFF
        return bits2f(v >> 32), bits2f(v & 0xFFFFFFFF)
    eps, t_at = unpack(keys["any"])  # +Inf when any fail-closed counter fired
    ahead, t_ahead = unpack(keys["ahead"])
    behind, t_behind = unpack(keys["behind"])
    # --- independent replay of the argmax and random CPU cross-check ---------------------------
    t0 = time.perf_counter()
    from native_same_s_falsify import Harness  # noqa: F401  (same harness used by run_progress)
    rng = np.random.default_rng(20261117)
    probe = np.concatenate([[t_at, t_ahead, t_behind, 0.0, bits2f(1), bits2f(0x007FFFFF), bits2f(0x00800000), Tn,
                             bits2f(last - 1)], rng.random(check_samples) * Tn]).astype(np.float32)
    probe_t = torch.as_tensor(probe, device=dev)
    hp = _harness_progress(A, inp, probe_t)
    cpu_max = 0.0
    argmax_replay = None
    for idx, (t, s) in enumerate(zip(probe, hp)):
        lo, hi = TG.inverse_bracket(grid, float(t))
        e = max(float(s) - lo, hi - float(s), 0.0)  # far-end distance: sound upper bound (R0 U1)
        if idx == 0:
            argmax_replay = e
        cpu_max = max(cpu_max, e)
    stages["cpu_crosscheck_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    spans = extract_timeopt_spans(arr["effective_geometry"], arr["timing_controls"])
    L = lipschitz(spans)
    stages["lipschitz_s"] = time.perf_counter() - t0
    lim = {n: np.minimum(np.abs(arr[k][0]), np.abs(arr[k][1])).astype(float)
           for n, k in (("v", "velocity_limits"), ("a", "acceleration_limits"), ("j", "jerk_limits"))}
    dev_bounds = {k: [fup(eps * v) for v in L[k]] for k in ("q", "v", "a", "j")}
    dev_bounds["j_with_knot_jump"] = [fup(d + jj) for d, jj in zip(dev_bounds["j"], L["jump_j"])]
    integrity = {
        "processed_equals_domain": cnts[0] == total, "processed": cnts[0], "domain_size": total,
        "beyond_duration_inverse_times": beyond_total, "beyond_processed": bcnts[0],
        "beyond_counters_clean": bcnts[0] == beyond_total and bcnts[1] == bcnts[2] == bcnts[4] == 0,
        "bad_progress": cnts[1], "empty_bracket": cnts[2], "progress_bit_mismatch": cnts[3],
        "bracket_lookup_failure": cnts[4], "production_harness_mismatch_words": mismatch_words,
        "production_saved_diagnostics_bitwise": all(saved_ok.values()),
        "argmax_cpu_replay": argmax_replay,
        # consistency checks (not proofs): CPU and GPU brackets differ only by directed-rounding details
        "argmax_replay_consistent": bool(np.isfinite(eps)) and abs(argmax_replay - eps) <= 1e-6 * eps + 1e-300,
        "cpu_crosscheck_max": cpu_max, "crosscheck_consistent": bool(np.isfinite(eps)) and cpu_max <= eps * (1 + 1e-6) + 1e-300,
        "eps_finite": bool(np.isfinite(eps)),
    }
    integrity_ok = (integrity["processed_equals_domain"] and integrity["beyond_counters_clean"] and cnts[1] == cnts[2] == cnts[3] == cnts[4] == 0
                    and mismatch_words == 0 and integrity["production_saved_diagnostics_bitwise"]
                    and integrity["crosscheck_consistent"] and integrity["argmax_replay_consistent"]
                    and integrity["eps_finite"])
    reported_ok = nonfinite == 0 and all(v == 0 for v in exceed.values())
    row.update({
        "status": ("AUDITED_CONSISTENT" if integrity_ok else "FAILED_TO_AUDIT_INTEGRITY"),
        "reported_status": ("REPORTED_WITHIN_LIMITS_ALL_FLOAT32_TIMES" if integrity_ok and reported_ok
                            else "REPORTED_EXCEEDS_LIMITS" if integrity_ok else "NOT_ESTABLISHED"),
        "time_grid_tables_sha256": tab_sha, "times_enumerated": total, "chunks": chunks,
        "native_duration_s": Tn, "model_duration_bracket_s": list(grid.duration),
        "grid_summation_error_s": grid.summation_error, "grid_max_cell_remainder": grid.max_cell_remainder,
        "eps_s_certified_max": eps, "at_time": t_at, "ahead_max": ahead, "ahead_at": t_ahead,
        "behind_max": behind, "behind_at": t_behind, "integrity": integrity,
        "reported": {f: {"max_utilization": best[f][0], "at_time": best[f][1], "joint": best[f][2],
                         "times_x_joints_exceeding": exceed[f]} for f in ("v", "a", "j")},
        "reported_nonfinite_words": nonfinite, "lipschitz": L, "state_deviation_bounds": dev_bounds,
        "relative_deviation_vs_limits": {k: max(d / l for d, l in zip(dev_bounds[k], lim[k])) for k in ("v", "a", "j")},
        "stages_s": stages, "elapsed_s": time.perf_counter() - t_all})
    return row


def _harness_progress(A, inp, times_t):
    torch = A.torch
    n = times_t.shape[0]
    out = torch.empty((4, n, inp["dof"]), dtype=torch.float32, device=A.device)
    prog = torch.empty(n, dtype=torch.float32, device=A.device)
    # same progress_at_time as the audit kernels, at arbitrary (non-consecutive) times
    from native_same_s_falsify import Harness
    if not hasattr(A, "_h"):
        A._h = Harness(A.device)
    A._h.launch("progress_eval", n, inp["controls"], inp["start"], inp["goal"], inp["timing"], inp["polynomial"],
                inp["nodes"], inp["weights"], inp["time"], times_t, out, prog, np.int32(n), np.int32(inp["dof"]),
                np.int32(inp["knots"]), np.int32(inp["intervals"]), np.int32(1))
    torch.cuda.synchronize(A.device)
    return prog.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--log2-cells", type=int, default=22)
    ap.add_argument("--chunk", type=int, default=1 << 24)
    ap.add_argument("--check-samples", type=int, default=20000)
    ap.add_argument("--require-gpu", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    dev = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(dev)
    if a.require_gpu and gpu != a.require_gpu:
        raise SystemExit(f"WRONG_GPU {gpu!r}")
    A = AuditV2(torch, dev)
    rows = []
    for path in a.npz:
        row = audit_path(A, path, a.log2_cells, a.chunk, a.check_samples)
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("path", "status", "reported_status", "times_enumerated",
                                                   "eps_s_certified_max", "elapsed_s")}), flush=True)
    out = {"scope": SCOPE, "gpu": gpu, "capability": list(torch.cuda.get_device_capability(dev)),
           "driver_cuda": torch.version.cuda, "compile_flags": A.flags, "source_sha256": A.source_hashes,
           "grid_module_sha256": sha(ROOT / "src/curobo_certify/time_law_grid.py"),
           "driver_script_sha256": sha(Path(__file__)), "platform": platform.platform(),
           "excluded_domain": ["-0.0 and negative times", "t > T_native (production tail: v=a=j=0)", "NaN/Inf"],
           "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(_finite(out), indent=1, allow_nan=False) + "\n")


def _finite(v):
    """JSON without NaN/Inf: non-finite floats become explicit strings."""
    if isinstance(v, float) and not np.isfinite(v):
        return "NONFINITE_" + repr(v)
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    return v


if __name__ == "__main__":
    main()

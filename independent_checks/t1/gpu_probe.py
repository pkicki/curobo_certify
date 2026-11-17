#!/usr/bin/env python3
"""R0 GPU probes of the T1 audit kernels (run in the pinned cuRobo env, RTX 3090 only).

    CUDA_VISIBLE_DEVICES=1 conda run -n curobo-timeopt-c301cb \
        python independent_checks/t1/gpu_probe.py --out <json> [--paths N]

Probes (each a short GPU job):
  P1 production sample_with_progress_kernel (upstream python entry `sample`) vs the audit
     harness progress_eval, bitwise, on a contiguous chunk of 2^20 float32 times around the
     audited argmax time and on special inputs (+0, -0, subnormals, T_native, next float,
     2*T_native, +Inf, NaN, -1).
  P2 inverse_audit chunk replay: the audit kernel's packed atomicMax key on the chunk vs an
     exact (Fraction, directed-rounding-emulated) replay from harness progress values and
     the reviewed CPU tables.
  P3 reported_audit chunk replay against ratios computed from the PRODUCTION kernel outputs.
  P4 fault injection: NaN in a timing control -> what do inverse_audit / reported_audit report?
  P5 inconsistent tables (T_hi < T_lo, i.e. empty bracket) -> is anything flagged?
  P6 subnormal requested times: kernel result on all 2^23-1 positive subnormal t; FTZ probes of
     float->double and __double2float_ru with the same NVRTC options; PTX of inverse_audit.
  P7 native s at each audited path's recorded argmax times (input for the CPU independent
     epsilon check, run separately without GPU).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

AUDIT = ROOT / "src/curobo_certify/native/time_inverse_audit.cuh"


def bits32(x) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def unpack(v: int):
    v &= 0xFFFFFFFFFFFFFFFF
    return f32(v >> 32), v >> 32, v & 0xFFFFFFFF


# ---- exact directed-rounding emulation of the kernel's bracket arithmetic -----------------

def _round_dir(q: Q, up: bool) -> float:
    f = float(q)  # correctly rounded to nearest by CPython
    fq = Q(f)
    if up and fq < q:
        f = float(np.nextafter(f, np.inf))
    if (not up) and fq > q:
        f = float(np.nextafter(f, -np.inf))
    return f


def _f32_ru(q: Q) -> int:
    """bits of the smallest float32 >= q (q >= 0), no flush (IEEE semantics)."""
    f = np.float32(float(q))
    if Q(float(f)) < q:
        f = np.nextafter(f, np.float32(np.inf))
    return bits32(float(f))


def kernel_e_exact(t: float, s: float, Tlo, Thi, fmin, fmax, N: int):
    """Exact replay of inverse_audit's e for one (t, s) with IEEE directed rounding (no FTZ)."""
    td = t
    if not td > 0:
        lo = hi = 0.0
    else:
        k = int(np.searchsorted(Thi, td, side="right")) - 1
        j = int(np.searchsorted(Tlo, td, side="left")) - 1
        if k >= N or k < 0:
            lo = 1.0 if k >= N else 0.0
        else:
            sk, sk1 = k / N, (k + 1) / N
            d = _round_dir(Q(td) - Q(float(Thi[k])), False)
            q = _round_dir(Q(d) / Q(float(fmax[k])), False)
            v = _round_dir(Q(sk) + Q(q), False)
            v = max(v, sk)
            lo = min(v, sk1)
        if j >= N:
            hi = 1.0
        elif j < 0:
            hi = 0.0
        else:
            sj, sj1 = j / N, (j + 1) / N
            d = _round_dir(Q(td) - Q(float(Tlo[j])), True)
            q = _round_dir(Q(d) / Q(float(fmin[j])), True)
            v = _round_dir(Q(sj) + Q(q), True)
            hi = min(v, sj1, 1.0)
    ahead = _round_dir(Q(s) - Q(hi), True)
    behind = _round_dir(Q(lo) - Q(s), True)
    ea, eb = max(ahead, 0.0), max(behind, 0.0)
    return max(ea, eb), ea, eb, lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audit-json", type=Path, default=ROOT / "runs/margins/time_inversion_audit_224paths.json")
    ap.add_argument("--chunk-paths", type=int, default=4)
    ap.add_argument("--log2-cells", type=int, default=22)
    ap.add_argument("--skip-p7", action="store_true")
    a = ap.parse_args()
    import torch
    import native_same_s_falsify as NS
    from native_same_s_falsify import Harness, native_inputs, run_progress
    from curobo_certify import time_law_grid as TG
    from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import TrajectoryKernelCfg
    from curobo._src.curobolib.backends.cuda_core_backend import trajectory_timeopt as PROD

    dev = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(dev)
    assert "3090" in gpu, f"refusing to run on {gpu}; use CUDA_VISIBLE_DEVICES=1 (RTX 3090)"

    class AuditHarness(Harness):  # same construction as scripts/native_time_inversion_audit.py
        def __init__(self, device):
            super().__init__(device)
            cfg = TrajectoryKernelCfg()
            self.sources = [cfg.kernel_dir / "timeopt" / "timeopt_common.cuh",
                            cfg.kernel_dir / "timeopt" / "timeopt_path_kernel.cuh", NS.HARNESS, AUDIT]
            self.source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.sources}
            for short in ("inverse_audit", "reported_audit"):
                name = "curobo_certify_harness::" + short
                self.kernels[short] = self.cache.get_or_compile_kernel(
                    source_files=self.sources, kernel_name=name, include_dirs=cfg.get_include_dirs(),
                    compile_flags=self.flags)
                self.qualified[short] = name

    h = AuditHarness(dev)
    audit = json.loads(a.audit_json.read_text())
    rows = {r["path"]: r for r in audit["rows"]}
    npzmap = {}
    for f in ROOT.glob("runs/**/*.npz"):
        npzmap.setdefault(f.stem, []).append(f)

    def npz_for(row):
        for f in npzmap.get(row["path"], []):
            if hashlib.sha256(f.read_bytes()).hexdigest() == row["source_npz_sha256"]:
                return f
        raise FileNotFoundError(row["path"])

    def load(row):
        with np.load(npz_for(row), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def production(arr, inp, times):
        n = len(times)
        controls = inp["controls"][None].contiguous()
        start = inp["start"][None].contiguous()
        goal = inp["goal"].contiguous()
        timing = inp["timing"][None].contiguous()
        tm = inp["time"][None].contiguous()
        req = torch.as_tensor(np.asarray(times, np.float32)[None], device=dev).contiguous()
        out = torch.empty((4, 1, n, inp["dof"]), dtype=torch.float32, device=dev)
        prog = torch.empty((1, n), dtype=torch.float32, device=dev)
        succ = torch.ones(1, dtype=torch.bool, device=dev)
        stat = torch.zeros(1, dtype=torch.bool, device=dev)
        PROD.sample(controls, start, goal, timing, inp["polynomial"], inp["nodes"], inp["weights"], tm, req,
                    succ, stat, out, prog, 1)
        torch.cuda.synchronize(dev)
        return out[:, 0].cpu().numpy()

    def run_inverse(inp, tab, bstart, cnt, log2):
        res = {k: torch.zeros(1, dtype=torch.int64, device=dev) for k in ("any", "ahead", "behind")}
        h.launch("inverse_audit", cnt, inp["polynomial"], inp["timing"], inp["nodes"], inp["weights"], inp["time"],
                 np.int32(inp["intervals"]), np.uint32(bstart), np.uint32(cnt), tab["T_lo"], tab["T_hi"],
                 tab["fmin"], tab["fmax"], np.int32(log2), res["any"], res["ahead"], res["behind"])
        return {k: unpack(int(v.item())) for k, v in res.items()}

    def run_reported(arr, inp, bstart, cnt):
        lim64 = {n: torch.as_tensor(np.minimum(np.abs(arr[k][0]), np.abs(arr[k][1])).astype(np.float64), device=dev)
                 for n, k in (("v", "velocity_limits"), ("a", "acceleration_limits"), ("j", "jerk_limits"))}
        mr = torch.zeros(3, dtype=torch.int64, device=dev)
        nf = torch.zeros(1, dtype=torch.int64, device=dev)
        h.launch("reported_audit", cnt, inp["controls"], inp["start"], inp["goal"], inp["timing"], inp["polynomial"],
                 inp["nodes"], inp["weights"], inp["time"], np.int32(inp["intervals"]), np.int32(inp["dof"]),
                 np.int32(inp["knots"]), np.int32(1), np.uint32(bstart), np.uint32(cnt),
                 lim64["v"], lim64["a"], lim64["j"], mr, nf)
        return [unpack(int(mr[i].item())) for i in range(3)], int(nf.item())

    def tables(grid):
        return {k: torch.as_tensor(np.ascontiguousarray(v), device=dev) for k, v in
                (("T_lo", grid.T_lo), ("T_hi", grid.T_hi), ("fmin", grid.fmin), ("fmax", grid.fmax))}

    report = {"gpu": gpu, "compile_flags_passed": h.flags, "source_sha256": h.source_hashes, "probes": {}}

    # choose chunk paths: max-eps path + a margin-1.00 violator + two others
    ordered = sorted(audit["rows"], key=lambda r: -r["eps_s_certified_max"])
    picks = [ordered[0]["path"]]
    picks += [r["path"] for r in audit["rows"] if r["exhaustive_reported_status"] != "CERTIFIED_EXHAUSTIVE"][:1]
    picks += [r["path"] for r in audit["rows"] if r["path"].endswith("0p950")][:a.chunk_paths - len(picks)]
    p1, p2, p3 = [], [], []
    for name in picks:
        row = rows[name]
        arr = load(row)
        inp = native_inputs(arr, torch, dev)
        Tn = float(arr["native_timing_time_grid"][-1])
        grid = TG.build_grid(arr["timing_controls"], a.log2_cells)
        tab = tables(grid)
        centre = bits32(row["at_time"])
        bstart, cnt = max(0, centre - (1 << 19)), 1 << 20
        times = np.array([f32(b) for b in range(bstart, bstart + cnt)], np.float32)
        # P1 production vs harness
        prod = production(arr, inp, times)
        hout, hs = run_progress(h, inp, times)
        neq = (prod.view(np.uint32) != hout.view(np.uint32))
        dom = times <= np.float32(Tn)
        mism = int(np.sum(neq[:, dom]))
        mism_tail = int(np.sum(neq[:, ~dom]))
        tail_fams = {fam: int(np.sum(neq[i][~dom])) for i, fam in enumerate(("q", "v", "a", "j"))}
        specials = {"+0": 0.0, "-0": -0.0, "min_subnormal": f32(1), "max_subnormal": f32(0x7FFFFF),
                    "min_normal": f32(0x800000), "T_native": Tn, "next_after_T_native": f32(bits32(Tn) + 1),
                    "2T_native": 2 * Tn, "+inf": float("inf"), "nan": float("nan"), "-1": -1.0}
        st = np.array(list(specials.values()), np.float32)
        sp_prod = production(arr, inp, st)
        sp_h, sp_s = run_progress(h, inp, st)
        sp = {}
        for i, key in enumerate(specials):
            sp[key] = {"harness_s": float(sp_s[i]),
                       "prod_eq_harness_bitwise": bool(np.array_equal(sp_prod[:, i].view(np.uint32), sp_h[:, i].view(np.uint32))),
                       "prod_v_max_abs": float(np.nanmax(np.abs(sp_prod[1, i]))) if np.isfinite(sp_prod[1, i]).all() else "nonfinite",
                       "prod_j_first": float(sp_prod[3, i, 0]), "harness_j_first": float(sp_h[3, i, 0])}
        p1.append({"path": name, "chunk_bits": [bstart, bstart + cnt - 1], "values": int(prod.size),
                   "times_in_domain": int(dom.sum()), "times_beyond_T_native": int((~dom).sum()),
                   "bit_mismatches_prod_vs_harness_in_domain": mism,
                   "bit_mismatches_beyond_T_native": mism_tail, "beyond_T_native_mismatch_by_family": tail_fams,
                   "specials": sp})
        # P2 inverse_audit replay
        k = run_inverse(inp, tab, bstart, cnt, a.log2_cells)
        # RN prefilter, then exact emulation on the top candidates
        td = times.astype(np.float64)
        kk = np.searchsorted(grid.T_hi, td, side="right") - 1
        jj = np.searchsorted(grid.T_lo, td, side="left") - 1
        N = grid.n_cells
        kc, jc = np.minimum(kk, N - 1), np.minimum(jj, N - 1)
        lo = np.where(kk >= N, 1.0, np.minimum(np.maximum(kc / N + (td - grid.T_hi[kc]) / grid.fmax[kc], kc / N), (kc + 1) / N))
        hi = np.where(jj >= N, 1.0, np.minimum(np.minimum(jc / N + (td - grid.T_lo[jc]) / grid.fmin[jc], (jc + 1) / N), 1.0))
        sd = hs.astype(np.float64)
        e_rn = np.maximum(np.maximum(sd - hi, lo - sd), 0.0)
        top = np.argsort(-e_rn)[:64]
        best = None
        for i in top:
            e, ea, eb, _, _ = kernel_e_exact(float(times[i]), float(hs[i]), grid.T_lo, grid.T_hi, grid.fmin, grid.fmax, N)
            key = (_f32_ru(Q(e)), bits32(times[i]))
            best = key if best is None or key > best else best
        p2.append({"path": name, "gpu_key_any": {"e": k["any"][0], "e_bits": k["any"][1], "t_bits": k["any"][2]},
                   "replay_key_any": {"e": f32(best[0]), "e_bits": best[0], "t_bits": best[1]},
                   "exact_match": (k["any"][1], k["any"][2]) == best,
                   "audited_eps_full_path": row["eps_s_certified_max"], "audited_at_time_bits": centre})
        # P3 reported_audit replay from PRODUCTION outputs
        mr, nf = run_reported(arr, inp, bstart, cnt)
        lims = [np.minimum(np.abs(arr[kq][0]), np.abs(arr[kq][1])).astype(np.float64)
                for kq in ("velocity_limits", "acceleration_limits", "jerk_limits")]
        fam = []
        for f in range(3):
            ratio = np.abs(prod[f + 1].astype(np.float64)) / lims[f][None, :]
            rmax = ratio.max(1)
            topi = np.argsort(-rmax)[:32]
            bestk = None
            for i in topi:
                r = max(Q(abs(float(prod[f + 1, i, jx]))) / Q(float(lims[f][jx])) for jx in range(prod.shape[2]))
                key = (_f32_ru(Q(_round_dir(r, True))), bits32(times[i]))
                bestk = key if bestk is None or key > bestk else bestk
            fam.append({"gpu": {"ratio": mr[f][0], "t_bits": mr[f][2]}, "replay_from_production": {"ratio": f32(bestk[0]), "t_bits": bestk[1]},
                        "exact_match": (mr[f][1], mr[f][2]) == bestk})
        p3.append({"path": name, "families": dict(zip(("velocity", "acceleration", "jerk"), fam)),
                   "gpu_nonfinite": nf, "production_nonfinite": int(np.sum(~np.isfinite(prod)))})
    report["probes"]["P1_production_vs_harness"] = p1
    report["probes"]["P2_inverse_audit_chunk_replay"] = p2
    report["probes"]["P3_reported_audit_chunk_replay"] = p3

    # P4 NaN injection, P5 empty brackets (on the first pick)
    row = rows[picks[0]]
    arr = load(row)
    grid = TG.build_grid(arr["timing_controls"], a.log2_cells)
    tab = tables(grid)
    centre = bits32(row["at_time"])
    bstart, cnt = centre - (1 << 15), 1 << 16
    inp = native_inputs(arr, torch, dev)
    base = run_inverse(inp, tab, bstart, cnt, a.log2_cells)
    bad = dict(arr)
    tcn = arr["timing_controls"].copy()
    s_c = float(run_progress(h, inp, np.array([f32(centre)], np.float32))[1][0])
    nan_ctrl = min(31, int(s_c * 29) + 1)
    tcn[nan_ctrl] = np.nan
    bad["timing_controls"] = tcn
    inp_nan = native_inputs(bad, torch, dev)
    nan_inv = run_inverse(inp_nan, tab, bstart, cnt, a.log2_cells)
    _, s_nan = run_progress(h, inp_nan, np.array([f32(centre)], np.float32))
    nan_rep, nan_nf = run_reported(bad, inp_nan, bstart, cnt)
    report["probes"]["P4_nan_injection"] = {
        "path": picks[0], "times": cnt, "baseline_e_any": base["any"][0], "nan_control_index": nan_ctrl,
        "progress_at_centre_before": s_c,
        "nan_timing_progress_value": float(s_nan[0]),
        "nan_inverse_audit_e_any": nan_inv["any"][0], "nan_inverse_audit_e_ahead": nan_inv["ahead"][0],
        "nan_inverse_audit_e_behind": nan_inv["behind"][0],
        "nan_reported_audit_ratios": [r[0] for r in nan_rep], "nan_reported_audit_nonfinite": nan_nf,
        "interpretation": "see memo: progress_at_time clamps with fminf/fmaxf, which discard NaN, so s stays finite; "
                          "inverse_audit would map a NaN s to e=0 (latent fail-open), reported_audit counts NaN v/a/j."}
    swapped = dict(tab)
    swapped["T_hi"] = tab["T_lo"] * 0.5     # T_hi < T_lo everywhere except index 0: brackets become empty/inverted
    emp = run_inverse(inp, swapped, bstart, cnt, a.log2_cells)
    sw2 = dict(tab)
    sw2["fmin"], sw2["fmax"] = tab["fmax"], tab["fmin"]
    emp2 = run_inverse(inp, sw2, bstart, cnt, a.log2_cells)
    report["probes"]["P5_inconsistent_tables"] = {
        "baseline_e_any": base["any"][0], "T_hi_halved_e_any": emp["any"][0],
        "fmin_fmax_swapped_e_any": emp2["any"][0],
        "kernel_has_error_flag_output": False,
        "interpretation": "the kernel returns a number for inconsistent tables; lo>hi is never detected or reported."}

    # P6 subnormal times + FTZ semantics probes + PTX
    sub = run_inverse(inp, tab, 1, 0x7FFFFF, a.log2_cells)
    sub_times = np.array([f32(1), f32(0x400000), f32(0x7FFFFF)], np.float32)
    _, sub_s = run_progress(h, inp, sub_times)
    from cuda.core import Program, ProgramOptions, LaunchConfig, launch
    cache = h.cache
    opts = ProgramOptions(std="c++17", arch=cache.arch, ftz=True, fma=True, prec_div=False, prec_sqrt=False,
                          device_code_optimize=True, lineinfo=True)
    probe_src = r'''
extern "C" __global__ void ftz_probe(const unsigned int *bits, double *as_double, unsigned int *ru_bits,
                                     const double *dv, int n) {
  int i = threadIdx.x; if (i >= n) return;
  float t = __uint_as_float(bits[i]);
  as_double[i] = double(t);
  ru_bits[i] = __float_as_uint(__double2float_ru(dv[i]));
}'''
    kern = Program(probe_src, code_type="c++", options=opts).compile("cubin").get_kernel("ftz_probe")
    pb = torch.tensor([1, 0x7FFFFF, 0x800000], dtype=torch.int32, device=dev)
    dv = torch.tensor([1e-40, 1e-45, 2e-38], dtype=torch.float64, device=dev)
    o1 = torch.zeros(3, dtype=torch.float64, device=dev)
    o2 = torch.zeros(3, dtype=torch.int32, device=dev)
    stream = cache.get_stream_wrapper(torch.cuda.current_stream(dev))
    launch(stream, LaunchConfig(grid=1, block=32), kern, pb.data_ptr(), o1.data_ptr(), o2.data_ptr(), dv.data_ptr(), np.int32(3))
    torch.cuda.synchronize(dev)
    code = cache._read_sources(h.sources, TrajectoryKernelCfg().get_include_dirs())
    inc = [str(i) for i in TrajectoryKernelCfg().get_include_dirs()]
    from cuda import pathfinder
    for lib in ("cudart", "nvrtc"):
        d = pathfinder.find_nvidia_header_directory(lib)
        if d:
            inc.append(d)
    popts = ProgramOptions(std="c++17", arch=cache.arch, include_path=inc, ftz=True, fma=True, prec_div=False,
                           prec_sqrt=False, device_code_optimize=True, lineinfo=True)
    ptx = Program(code, code_type="c++", options=popts).compile(
        "ptx", name_expressions=("curobo_certify_harness::inverse_audit",)).code
    ptx = ptx.decode() if isinstance(ptx, (bytes, bytearray)) else str(ptx)
    m = re.search(r"\.entry\s+\S*inverse_audit\S*\((.*?)\n}\n", ptx, re.S)
    body = m.group(0) if m else ""
    cvt = sorted(set(re.findall(r"cvt\.[a-z0-9.]+", body)))
    dops = sorted(set(re.findall(r"\b(?:add|sub|div|rcp|fma)\.r[mpnz]\.f64", body)))
    report["probes"]["P6_subnormal_and_ftz"] = {
        "inverse_audit_on_all_positive_subnormal_t": {"e_any": sub["any"][0], "t_bits": sub["any"][2]},
        "harness_s_at_subnormal_t": [float(v) for v in sub_s],
        "double_of_subnormal_float_under_ftz": [float(v) for v in o1.cpu().numpy()],
        "double2float_ru_bits_under_ftz_for_[1e-40,1e-45,2e-38]": [int(v) & 0xFFFFFFFF for v in o2.cpu().numpy()],
        "inverse_audit_ptx_cvt_ops": cvt, "inverse_audit_ptx_directed_f64_ops": dops,
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest()}

    # P7 native s at recorded argmax times for all audited paths
    if not a.skip_p7:
        p7 = []
        for r in audit["rows"]:
            arr = load(r)
            inp = native_inputs(arr, torch, dev)
            ts = np.array([r["at_time"], r["ahead_at"], r["behind_at"]], np.float32)
            _, s = run_progress(h, inp, ts)
            p7.append({"path": r["path"], "t": [float(x) for x in ts], "s_native": [float(x) for x in s],
                       "eps": r["eps_s_certified_max"], "ahead": r["ahead_max"], "behind": r["behind_max"]})
        report["probes"]["P7_native_s_at_argmax"] = p7
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in report["probes"].items() if k != "P7_native_s_at_argmax"}, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()

# R0 independent review of the T1 native time-law audit

**Date:** 2026-11-17
**Verdict: R0 FAIL. Do not release formal cohorts.** One soundness defect was found. The audited "certified ε" is not an upper bound: there is a proven counterexample on every one of the 224 paths. There are also several fail-open, traceability and scope gaps. The CPU time-law enclosure (`time_law_grid`) and the inverse bracket held up under every independent check I ran. The defect is in how the GPU kernel turns a sound bracket into ε. After the fix in §3.1, the corrected ε is still small (§2.3). The headline figure must be restated and the remaining gaps closed before R0 is re-run.

## 0. Reviewer identity, independence and method

* **Reviewer:** an isolated Claude subagent (Claude Code, model Opus 5.5) running as the gate-R0 reviewer. I had no part in writing the code under review. I received only the task brief and the repository. I did not import, copy or adapt any formula from `time_law_grid.py`, `time_law_enclosures.py` or `native_error_bound.py` into the checker. The cross-check modules call the reviewed code only in order to test it.
* **Git:** HEAD `7b9c348a83dc64911f7fdbf4a47ad2878e1179b4`, branch `exp/native-t1-formal-qualification-plan-20261117`. Between the code-under-review commit `fe3dd7c` and HEAD, nothing under `src/`, `scripts/`, `fixtures/` or `runs/` changed (`git diff --stat fe3dd7c HEAD -- src scripts fixtures runs` is empty). Only planning docs changed.
* **Upstream:** NVlabs/curobo `c301cb00a87012c2808b1a60649c4065c9b3c95a` at `/home/pkicki/curobo-pinned-c301cb00…`. Its `git rev-parse HEAD` matches, and the tree is clean.
* **Hardware and toolchain:** NVIDIA GeForce RTX 3090, driver 535.146.02. Every GPU job asserted `"3090" in torch.cuda.get_device_name()`, and nothing ran on the A100. GPU env `curobo-timeopt-c301cb`: Python 3.11, torch 2.5.1, numpy 2.1.3. CPU: Python 3.12, numpy 1.26.4, mpmath 1.3.
* **Method:**
  1. I wrote an independent exact model and a rigorous enclosure from the mathematics. See `independent_checks/t1/`, described in §4.
  2. I ran exact-rational containment checks of the reviewed tables and of `inverse_bracket` on all 224 audited NPZs.
  3. I ran GPU probes against the production upstream entry point `trajectory_timeopt.sample`.
  4. I replayed chunk-level GPU maxima exactly.
  5. I ran fault injection.
  6. I re-enumerated all 224 paths with a corrected reduction kernel. This is a review artifact, not a registered re-run.
  7. I inspected the source and PTX of the kernels, the drivers and the cuRobo compile and launch path.

Terminology used below:
* **unsound:** could certify something false.
* **fail-open:** a diagnostic or control that does not flag a failure.
* **doc/scope:** documentation, contract wording or traceability.

## 1. Proof-obligation matrix (plan §C)

| # | Obligation | Verdict | Evidence |
|---|---|---|---|
| **R0.1** | **Exact time-law grid** | | |
| 1.1 | Exact float32 → rational controls and common-knot cubic representation | PASS | I decode float32 bits by hand (`exact_spline.f32_bits_to_rational`) and evaluate the clamped cubic by de Boor at 4 exact points per span, then solve the exact Vandermonde system. The result equals the repo extraction (`exact_timing_table`) coefficient for coefficient: **0 mismatches on 224/224 paths** (`cpu_review_results.json`, `model_mismatch_paths: 0`) and on random controls (`test_identity_with_repo_exact_model`). It agrees with the upstream `CubicTimingBasis` float32 export to ≤2e-6 relative (`test_matches_upstream_float32_table`), and that fixture is bitwise identical to production `CubicTimingBasis(32).polynomial`, nodes and weights. C2 holds and the endpoints are x(0)=c₀, x(1)=c₃₁ (`test_c2_and_endpoints_random`). `frac_iv` is outward (time_law_grid.py:50-52). Note: the exact model uses exact knots i/29, while production builds its table from binary64 `linspace` knots. That is a definition, and the native table error is charged in `native_error_bound`. |
| 1.2 | x>0 on every span and cell (endpoints, joints, real cases) | PASS | Independent positivity proof on 224/224 paths (status `BOUNDED_EXACT_MODEL`: ρ<½ on every Taylor cell, so x ≥ x₀(1−ρ) > 0). The reviewed grid is `BOUNDED_EXACT_MODEL` on 224/224. Both fail closed when x has an internal zero (`test_internal_root_fails_closed`, `test_zero_at_internal_knot_fails_closed`). The reviewed grid is conservative: it fails closed (UNKNOWN_POSITIVITY) even at 2^22 when x≈1e-6 at a knot (three controls 1e-6), where my checker proves positivity. That is acceptable. |
| 1.3 | Structural C2/knot; strict-rest vs stationary-geometry endpoint | PASS (time law) / note | C2: exact jump test. Strict-rest module `endpoint_root_integral`: closed forms ∫du/√(4u) and 6·asinh(√½) are contained, and NONINTEGRABLE, NEGATIVE and IDENTICALLY_ZERO are classified correctly (`TestStrictRestEndpointModule`). It is not used by any real output. The T1 driver does not itself check the stationary-geometry endpoint contract (q′(0)=q′(1)=0). It relies only on x>0, which is all the time law needs. |
| 1.4 | Interval Horner, nextafter direction, sqrt/div direction | PASS | Inspection of time_law_grid.py:33-83: every RN result is followed by nextafter in the safe direction, which is sound for each binary64 op (RN then one ulp outward). Independent exact containment: **168,436 T_lo, 168,436 T_hi, 168,212 fmin and 168,212 fmax entries at 2^22, all PROVED, 0 violated, 0 undecided** (224 paths, ~752 indices each: every knot-adjacent cell, both ends, 512 random). The 2^10, 2^12 and 2^16 tables were checked in full on a real path, and 2^8 in full on a benign synthetic path. Hardening (unreachable here): `_inv_sqrt` (l.83) returns a negative "upper" bound if `lo` equals the smallest subnormal (dn(√·)=−0⁻). |
| 1.5 | Midpoint remainder h³/24·sup\|f″\| | PASS (doc nit) | f″ = ¾x^{-5/2}x′² − ½x^{-3/2}x″ is correct, and the remainder holds for C² f on knot-free cells. The comment at l.140 ("h^3/24 exact") is false: RN(1/3) < 1/3 by a relative 2^-54. The following `up()` adds a full ulp (≥ the 2^{e-53} needed), so the product is still an upper bound. |
| 1.6 | Cells crossing timing knots | PASS for log2_cells ≥ 5; latent defect below | Straddling cells use the hull of the two adjacent polynomials, each extended over the whole cell (l.145-157). That is a sound superset. **Latent:** `m0/m1` (l.108-110) consider only the first and last span. For N<29 a cell contains 2 knots and the middle span is ignored (`test_F_small_grid_multi_knot_cells_ignored`). log2_cells ≤ 4 is not rejected. A 3000-trial random search found no false bound (most such grids fail closed). The registered 2^22 configuration is unaffected. |
| 1.7 | T_lo/T_hi prefix monotonicity, finiteness | PASS | Full-array checks on 224 paths: finite, nondecreasing, T_lo≤T_hi, fmin≤fmax, fmin>0, T[0]=0. Argument: RN is monotone, cumsum of positive terms is nondecreasing, and dn/up/max(·,0) preserve order. |
| 1.8a | Accumulation safety (math) | PASS | `np.cumsum` is sequential (verified bitwise against a Python loop), so \|fl(S_k)−S_k\| ≤ γ_{k−1}Σ\|t_i\| ≤ γ_N Σ hi_int. `E` uses `np.sum` (pairwise). Any summation order gives S ≤ S′/(1−γ_N) with γ_N = 4.66e-10 < 1e-9, so the `(1+1e-9)` factor dominates. It also covers the two RN products (2^-52 ≪ 5.3e-10 of slack). lo_int ≤ hi_int, so the same E covers T_lo. I re-derived this independently, and the exact-sum enclosure (integer prefix sums, no FP) agrees with containment everywhere. |
| 1.8b | Justified bound in code, not a fudge; fail-closed intermediates | FAIL (doc/hardening) | The `*(1+1e-9)` at l.163 has no stated justification (the argument in 1.8a is missing from the code and docs). Intermediates are not all checked: only `lo_int>0` and `isfinite(hi_int)` are tested. NaN in `lo_int` would pass `lo_int <= 0`. T_lo/T_hi are not re-checked for finiteness or monotonicity after summation. |
| 1.9 | inverse_bracket inequalities, branches, t=0, T_exact, T_native ≶ T_exact, saturation, index −1/N | PASS | Inspection (l.170-188) and the GPU twin (cuh l.48-71): k ≥ 0 because T_hi[0]=0<t; k=N ⇒ T(1)≤t ⇒ s*=1; the cap at s_{j+1} is valid because j is the largest index with T_lo<t on a monotone T_lo. Independent exact decisions: **36,288 bracket endpoints, all PROVED, 0 violated, 0 undecided, 0 empty**. The tested times included t = 0, 2^-149, the largest subnormal, 2^-126, 1e-30, T_native ±1 ulp, the audited argmax times, 64 random times, and float32 times around the independent T(1) upper bound (saturation [1,1] is proved). |
| 1.10 | Adversarial suite | PASS (2^22 sampled only) | `tests/test_independent_t1_checker.py`: constant, affine (exact rational closed form), positive cubic (1+s)³ (exact rational), (1+s)² (mpmath.iv log), near-zero minimum ε=1e-8 (mpmath.iv asinh), internal double root, x≈1e-6 and x=0 at an internal knot, knot straddles, subnormal times, saturation, resolutions 2^8/2^10/2^12/2^16 in full, 2^22 sampled. A full exact check at 2^22 would take ~15 min per path with this checker, so it was not done. |
| 1.11 | Randomized rational families, exact inverse spot checks | PASS | Random log-uniform float32 controls in [1e-3, 1e2]: full 2^12 table containment and inverse checks (fail-closed grids skipped). Real paths: 224 × 81 times each. |
| **R0.2** | **Exhaustive GPU enumeration** | | |
| 2.1 | Bit domain = [+0, T_native] incl. subnormals, endpoint; exact counts; chunk/uint32 | PASS | Driver l.141-150: bits 0…bits(T_native) inclusive, chunk `min(chunk, total−start)`. bits(T_native) < 2^31, so `bits_start+i` cannot overflow uint32. Grid = ⌈n/256⌉ blocks with an `i<count` guard. `times_enumerated` = bits(T_native)+1 for 224/224, and the sum is 236,385,503,605. A per-thread counter in my corrected kernel reports **processed = expected on 224/224**. |
| 2.2 | −0, NaN, ±Inf, negative, t>T_native excluded/documented | FAIL (doc/scope) | These are not documented. Measured production behaviour (P1 specials): −0 gives output bitwise equal to the harness at +0. NaN and +Inf give NaN outputs. −1 gives s=0 with the zero tail. **t>T_native gives v=a=j=0 (tail branch, timeopt_path_kernel.cuh:30-40)**, but the harness and audit kernels evaluate at s=1, where jerk ≠ 0 (P1: 1,847,881 jerk-only mismatches, all beyond T_native; 0 in-domain). See 3.2. |
| 2.3 | Launch coverage, sync, error checking, reset per path, completed work | FAIL (fail-open control) | Launch errors are raised by `cuda.core.launch`. `Harness.launch` synchronizes after every launch, so asynchronous errors surface in `torch.cuda.synchronize`. Accumulators are freshly zeroed per path. Note: `launch_helper.cudaGetLastError` (runtime API) does not see driver-API launch errors, but cuda.core raises them itself. **The reviewed kernels have no completed-work counter**, and the plan requires "expected total enumerations equal actual completed work". That evidence now exists only from my review kernel (2.1). |
| 2.4 | Audit executes the same progress_at_time/tail, same flags, as production | UNKNOWN | Same unmodified upstream source. Effective NVRTC options are hard-coded in `kernel_cache._compile_kernel` (ftz, fma, prec_div=False, prec_sqrt=False, lineinfo, -O); the `compile_flags` list is only a cache key. The translation unit differs: production compiles `timeopt_path_kernel.cuh` alone, while the audit compiles common+path+harness+audit. `progress_at_time` is inlined separately into each kernel. **`reported_audit` uses the harness's copy of the tail expressions (`tail_values`), not `sample_with_progress_kernel`.** No SASS-level identity argument is given. Evidence, strong but not a proof: P1 found 0 bit mismatches between production `sample()` and harness `progress_eval` over 4×2^20 contiguous float32 times (4,193,169 in-domain times, 117M values) around each audited argmax. P2 and P3 exact key replays matched (see 2.10). |
| 2.5 | largest_le/largest_lt; directed-rounding f64 in source/PTX/SASS | PASS (source, PTX, replay) / SASS not inspected | Binary searches are correct for the required predicates. The PTX of `inverse_audit` contains `sub.rm/rp.f64`, `div.rm/rp.f64`, `add.rm/rp.f64` (P6). An exact Fraction emulation of the IEEE directed operations reproduced the GPU packed key bit for bit on 4 chunks (P2). I did not inspect the SASS of the f64 division sequence. |
| 2.6 | **ε reduced by the kernel is an upper bound of \|s_native − s*\|** | **FAIL (UNSOUND)** | time_inverse_audit.cuh:73-76 reduces `max(s−hi, lo−s, 0)`, the distance to the near end of the bracket. That is a *lower* bound of the error when s ∉ [lo,hi], and 0 when s ∈ [lo,hi]. A sound bound is `max(s−lo, hi−s)`. **Proven counterexamples on all 224 paths:** at each path's own audited argmax time, the independent bracket (width ≤2^-52) gives the true error ≥ s−s*_hi > reported ε. Example: `margin…0000-margin_0p900`, t=0.17535433173179626, s_native=0.6203527450561523. Reported ε=1.904871e-7, while the true \|s−s*\| ≥ 1.911012e-7 (`test_F1_reported_eps_violated_on_real_path`). The driver's CPU cross-check (native_time_inversion_audit.py:182) uses the same formula, so it could not catch this. Corrected values: §2.3. |
| 2.7 | atomicMax packing monotone; upward conversion; NaN/Inf; 64-bit | PASS (finite ≥0, +Inf) / FAIL (fail-open, latent) | For e ≥ 0 the float32 bit pattern is order-preserving, and the low 32 bits make the argmax deterministic. NaN: `ahead>0 ? ahead : 0` maps a NaN s or e to 0, which is silently dropped. This is unreachable today because `progress_at_time` ends in `fmaxf(left,fminf(right,·))` or 0/1, and fminf/fmaxf discard NaN. P4: a NaN timing control gives a finite s, ε jumps to 1.96e-3 and is reported. The guard is still missing. `__double2float_ru` compiles to `cvt.rp.ftz.f32.f64`, so e ∈ (0, 2^-126) rounds to +0, not upward (P6: 1e-40 → bits 0). That is negligible but not "upper-rounded". In reported_audit, NaN ratios are dropped from the maxima (P4: ratios reported 0.0) but counted in `nonfinite` (65,536/65,536), so the path status still fails closed. |
| 2.8 | Empty bracket (lo>hi) detected | FAIL (fail-open) | There is no check. P5: tables with T_hi = ½T_lo give a returned ε=0.053, and swapped fmin/fmax give 2.44e-7, with no flag. Real data had 0 empty brackets on 224/224 (review kernel counter). |
| 2.9 | Reported audit: physical limits, production ops, q finiteness, family/joint/time | FAIL (fail-open diagnostics) | Limits are min(\|lower\|,\|upper\|) per joint from the NPZ (driver l.152). These are the unscaled physical limits: margin-1.00 outputs reach 0.9999. v/a/j finiteness is checked, but **q is not** (cuh l.106-107). The argmax time is recorded per family, but the joint is not. `out[4*16]` has no dof ≤ 16 guard. The `stationary` and `success` branches of production are not modelled (all audited outputs were non-stationary successes, which the bitwise reproduction gates support). It runs the harness copy of the tail, not production (2.4). |
| 2.10 | Chunk maxima cross-checked by a separate replay | PASS (performed in this review) | P2: `inverse_audit` key equals the exact replay (harness s + IEEE directed emulation + reviewed tables) on 4/4 chunks of 2^20, including the full-path argmax. P3: `reported_audit` keys for v/a/j equal the replay from **production** `sample()` outputs on 12/12 family-chunks. The driver's own 5,000-sample CPU check is not independent (same bracket, same near-end formula, 1e-6 relative tolerance). |
| 2.11 | Unordered reduction / nondeterminism | PASS | atomicMax over a total order is order-independent. Per-thread values are deterministic (no atomics or reductions inside the computation). Re-runs reproduced every audited ε bit for bit (legacy column equals the audited value on 224/224). |
| 2.12 | SASS/A1–A5 on the exact audit binaries; FTZ; input normality/subnormals | FAIL (doc) / UNKNOWN | `sass_audit.py` (l.48-50) inventories `same_s_eval`, `progress_eval` and `sample_with_progress_kernel` only, not `inverse_audit` or `reported_audit`. `fp_unit_audit.py` covers positive normal inputs only. **FTZ:** PTX shows `cvt.ftz.f64.f32`, so all 8,388,607 positive subnormal t are converted to td=0, the bracket becomes [0,0] although s*(t)>0, and the kernel reports e=0 (P6). The native s is also 0 for those t, so the true error s*(t) ≤ t·√max x < 1e-37 ≪ ε. The path-level maximum is unaffected, but the per-time statement is false on subnormals and is undocumented. For ε itself, native s is observed, so A1–A5 are not needed. They matter only for E_same-progress, which was outside the depth of this review. |
| **R0.3** | **Contract semantics and traceability** | | |
| 3.1 | Progress bound → q,v,a,j with derivative factors, knot jumps, Lipschitz | PASS (formulas) conditional on 2.6 | `lipschitz()` (driver l.53-95): dv/ds = q″√x + q′x′/(2√x); da/ds = q‴x + 1.5q″x′ + 0.5q′x″; dj/ds = ℓx′/(2√x) + √x ℓ′ with ℓ′ = 2.5q‴x′ + 2q″x″ + 0.5q′x‴. All are correct. ℓ is continuous across timing knots (x is C²), and geometry-knot jumps are added separately. The √x bounds are float-derived and verified by `assert`, which is disabled under `python -O`, so the checks should be explicit. The output L·ε inherits the unsound ε of 2.6. |
| 3.2 | Output when s or t exceeds the model interval; endpoint hold | FAIL (scope) | On **125/224** paths T_native < T_exact (largest gap 1.17e-7 s). For t ∈ (T_native, T_exact], production emits the tail (v=a=j=0) while the reference q(s*(t)) has s*<1. Neither the audit nor the contract covers this. On **96/224** paths T_native > T_exact (gap up to 1.36e-7 s). For t ∈ (T_exact, T_native], the audit compares against model formulas at s*=1 (non-zero one-sided jerk), whereas the physical derivative of the held reference is 0. For ε alone, monotonicity extends the bound to t > T_native (native s=1 and 1−s*(t) ≤ 1−s*(T_native)), but this is not stated. |
| 3.3 | Physical derivatives vs sampled reported arrays | FAIL (doc) | `NATIVE_TIME_LAW_20261117.md` says the samples are "of a well-defined continuous-time trajectory whose physical derivatives are the certified quantities". It also uses the scope `NATIVE_PHYSICAL_TIME_LINKED`. Plan §B.3 rightly forbids that reading. The doc must be brought into line. |
| 3.4 | 224-path ε=2.44e-7 and 200/224 claims; SHAs, counts, binary, hardware | FAIL (ε) / PASS (provenance) / UNKNOWN (200/224, binary) | NPZ SHA-256 matches 224/224, and counts match (2.1). GPU "RTX 3090" is recorded. ε: **not a bound** (2.6). The corrected maximum is in §2.3. 200/224: not re-run exhaustively by me. Chunk replays agree, but the claim rests on 2.4 identity. Traceability gap: the run records source hashes only, with no cubin/PTX hash, driver version or exact NVRTC option set (the recorded `-O3` flag is not what is applied). |
| 3.5 | 0.99: 28/29 analytic vs 29/29 exhaustive | PASS (explained) | The single analytic non-certificate is `margin…0012-margin_0p990`, status UNKNOWN_TIGHTENED_WITNESS. MODEL_K3 is CERTIFIED there, and the exhaustive reported jerk maximum is 0.99536 < 1. These are different contracts, and the analytic bound tightens limits by E_j, so this reads as conservatism, not a bug. No loosening is warranted. |

## 2. Findings

### 2.1 Unsound
**U1. The ε reduced by the GPU audit is not an upper bound (2.6).**
* Location: `src/curobo_certify/native/time_inverse_audit.cuh:73-76`. The CPU cross-check mirrors it at `scripts/native_time_inversion_audit.py:182`.
* Mechanism: the reduction takes the distance to the near end of a sound bracket. The worst-case under-estimate per t is the bracket width hi−lo. Its maximum over all audited times is the "max bracket width" in §2.3 (5.36e-9).
* Effect: the per-path "certified" ε and the headline "certified max ≤ 2.44·10⁻⁷" are false as stated, with a proven counterexample on every path. The magnitude of the error is small, so after correction the scientific picture survives. But the certificate as issued is invalid, and any downstream L·ε state-deviation bound was computed from an invalid ε.

### 2.2 Fail-open diagnostics and controls
* **F1** No completed-work counter in `inverse_audit` or `reported_audit` (2.3).
* **F2** No empty-bracket (lo>hi) detection. Inconsistent tables yield a number, not an error (2.8; P5).
* **F3** NaN progress or NaN e would be mapped to e=0 (unreachable today; 2.7). In `reported_audit`, NaN ratios are dropped from the maxima, although the status still fails through `nonfinite`.
* **F4** `reported_audit` does not check q finiteness, does not record the offending joint, and has no `dof ≤ 16` guard (2.9).
* **F5** The driver's CPU cross-check is not independent: same tables, same near-end formula, a relative tolerance of 1e-6, and only 5,000 samples.
* **F6** `build_grid` does not reject `log2_cells ≤ 4`, where cells hold 2 knots and the middle span is ignored (1.6). Intermediate NaN in `lo_int` is not caught (1.8b).
* **F7** Rigour checks rely on `assert`, which `python -O` removes (driver l.67-69).

### 2.3 Measured impact of U1 (review re-enumeration, not a registered re-run)
`corrected_eps_run.py` uses the reviewed 2^22 tables, the same NVRTC path and the RTX 3090. Results over 224/224 paths:
* processed = expected everywhere;
* empty brackets = 0, non-finite = 0;
* the legacy column reproduces every audited ε bit for bit;
* **corrected max ε over paths = 2.448771e-7 (path margin-20261117-0017-margin_0p990)** (audited 2.4395e-7);
* largest per-path increase = 1.28e-9;
* largest bracket width = 5.36e-9.

Per-path values are in `corrected_eps_part{0..3}.json`.

At each audited argmax time, the independent rigorous error exceeds the reported ε on 224/224 paths. The largest independent value at an argmax time is 2.444139e-7. Independent errors at the argmax times are all ≤ the corrected ε (see the consistency line in §5).

### 2.4 Documentation and scope
* **D1** The admitted domain is not stated: −0, negative, NaN/Inf and t > T_native behave as measured in 2.2 but are undocumented.
* **D2** The tail and hold mismatch at T_native ≠ T_exact (3.2).
* **D3** "Physical derivatives" wording (3.3).
* **D4** Subnormal t under FTZ: the bracket is [0,0] while s*>0. The magnitude is negligible but undocumented (2.12).
* **D5** The SASS and A1–A5 audit does not cover the audit kernels (2.12).
* **D6** Missing binary and toolchain hashes. The recorded `compile_flags` do not reflect the options NVRTC actually received (3.4).
* **D7** The "h³/24 exact" comment and the unjustified `(1+1e-9)` factor. Both are sound, as shown in 1.5 and 1.8a, but the justification must be written down.
* **D8** `reported_audit` is a harness re-implementation of the production tail (2.4).

## 3. Recommended fixes
1. **U1:** in `inverse_audit`, reduce `e = max(__dsub_ru(sd, lo), __dsub_ru(hi, sd))`, the far-end distance. Keep the old quantities only as labelled diagnostics ("observed distance to bracket"). Apply the same fix to the driver's CPU cross-check. Re-issue every ε and every L·ε bound, and restate the headline. `independent_checks/t1/corrected_inverse_audit.cuh` is a drop-in reference.
2. Add fail-closed counters to both kernels: processed threads (assert equal to bits(T_native)+1), `lo > hi`, `!isfinite(s)`, `!isfinite(e)`, and in `reported_audit` `!isfinite(q)`. Fail the path if any count is wrong or non-zero. Pack the joint index into the reported key, or keep per-joint maxima.
3. Run the exhaustive reported audit on the **production kernel** `sample_with_progress_kernel` itself, in chunks with output buffers and a reduction, rather than on `tail_values`. Alternatively, add a SASS-level identity argument. Either way, record the cubin/PTX SHA-256 of every kernel used, the NVRTC options actually passed, the CUDA/NVRTC and driver versions and the GPU UUID.
4. State the domain precisely: [+0, T_native] float32. Add an explicit treatment of t > T_native: by monotonicity ε covers it, but reported v/a/j there are the zero tail. Add an explicit treatment of the T_native ≠ T_exact windows: either define the contract's reference as hold-at-T_native, or audit the window explicitly. Document the subnormal-t FTZ behaviour, or exclude subnormal t, or bound s*(t) ≤ t·√max x there.
5. `build_grid`: reject `log2_cells < 5`, check `isfinite` and `>0` on `lo_int`, re-check finiteness and monotonicity of T_lo/T_hi, guard `_inv_sqrt` against `dn()` reaching 0, write the γ_N/(1+1e-9) justification from 1.8a into the code, and fix the "exact" comment.
6. Extend `sass_audit.py` to `inverse_audit` and `reported_audit`. Replace `assert` with explicit checks in `lipschitz()`.
7. Amend `NATIVE_TIME_LAW_20261117.md` (scope wording, D3) and re-run R0 after the fixes. These findings came from exploratory data, so no holdout was exposed.

## 4. Independent artifacts (new files only)
* `independent_checks/t1/exact_spline.py`: bit-level float32 decoding, exact de Boor evaluation, exact interpolation of the span cubics.
* `independent_checks/t1/taylor_time_law.py`: rigorous T(s) enclosure. It uses a Taylor model of (1+u)^{-1/2} with Lagrange remainder, exact integer polynomial algebra, isqrt √ enclosures, outward dyadic rounding at 2^-200 and exact integer prefix sums. It also gives an inverse bracket by bisection, and the exact range of the cubic via critical points.
* `independent_checks/t1/crosscheck.py`: PROVED/VIOLATED/UNDECIDED containment of the reviewed tables and brackets.
* `independent_checks/t1/run_review.py`: CPU run over the 224 paths, written to `cpu_review_results.json`.
* `independent_checks/t1/gpu_probe.py`: probes P1–P7, written to `gpu_probe_results.json`.
* `independent_checks/t1/corrected_inverse_audit.cuh` and `corrected_eps_run.py`: the corrected re-enumeration, written to `corrected_eps_part*.json`.
* `tests/test_independent_t1_checker.py`: 25 tests.

## 5. Commands and results
```
PYTHONPATH=src:. /home/pkicki/anaconda3/bin/python -m unittest tests.test_independent_t1_checker -v      # 25 tests OK
PYTHONPATH=src:. /home/pkicki/anaconda3/bin/python independent_checks/t1/run_review.py --out docs/reports/r0_independent_t1_review/cpu_review_results.json --jobs 12
CUDA_VISIBLE_DEVICES=1 conda run -n curobo-timeopt-c301cb python independent_checks/t1/gpu_probe.py --out docs/reports/r0_independent_t1_review/gpu_probe_results.json
CUDA_VISIBLE_DEVICES=1 conda run -n curobo-timeopt-c301cb python independent_checks/t1/corrected_eps_run.py --first {0,56,112,168} --count 56 --out docs/reports/r0_independent_t1_review/corrected_eps_part{0..3}.json
```
Results:
* Unit tests: 25 tests, OK, about 16 s.
* CPU review (224 paths): model mismatches 0. T_lo/T_hi/fmin/fmax containment: 168,436 / 168,436 / 168,212 / 168,212 PROVED, 0 violated, 0 undecided. Inverse bracket endpoints: 36,288 PROVED, 0 violated, 0 undecided, 0 empty. Independent errors at the argmax times exceed the reported ε on 224/224 paths, and the largest is 2.444139e-7.
* GPU probes: P1 found 0 in-domain mismatches. P2 matched 4/4 chunks exactly and P3 matched 12/12. P4, P5 and P6 are as described in the matrix.
* Corrected re-enumeration: 236,385,503,605 times processed, equal to the expected count. 0 empty brackets, 0 non-finite. The legacy column equals the audited ε on 224/224. The corrected maximum ε is 2.448771e-7.
* Consistency: no independent argmax error exceeds that path's corrected ε (0/672).

## 6. SHA-256 of reviewed files
Repository (HEAD 7b9c348a83dc64911f7fdbf4a47ad2878e1179b4; identical to fe3dd7c for all of these except the two planning docs):

| file | sha256 |
|---|---|
| `src/curobo_certify/time_law_grid.py` | `a2d9bdceb2bb4a5fb96f66eaf0d641722dffaa0c0a6f16c284ffb7fa651cc385` |
| `src/curobo_certify/time_law_enclosures.py` | `bd4eb2c227d86bcd883f3bd473ade8de436721fc36a7aa07651aa96fb6a3d253` |
| `src/curobo_certify/native_error_bound.py` | `22eb144138fbd8d52a75fb29e6d1d2c7e577b73f5e8011726574e344147090b6` |
| `src/curobo_certify/spline_polynomials.py` | `9af0b96b6008dbb8ce58f7129b7ac5e1eed7c900649a7a5db728293f10aa95da` |
| `src/curobo_certify/native/time_inverse_audit.cuh` | `042e504e02656edb83c409fd7d764fa62785f685f416d2e2e98326e9364c1287` |
| `src/curobo_certify/native/same_s_harness.cuh` | `206cd38634b54d6044dbe700fb102b2e999dcd520ea97b2bb9209d973b4ad958` |
| `scripts/native_time_inversion_audit.py` | `d1159e0371762838bb534ae3411ac7548e64915d60fc4c2490f774845d75ab44` |
| `scripts/native_same_s_falsify.py` | `bbbaef43e10db923ed221d8586d2c881bcb2ad91d36eda775bee58e1c75500da` |
| `scripts/sass_audit.py` | `48def34ff8890ed9eabe8b18bbe7462f225fd3848258fe456ce5eca8c84091c0` |
| `scripts/fp_unit_audit.py` | `dfaaacf6239d673f811a149c95cd681a244a467303e0b8633e37698a2b2032fc` |
| `fixtures/timeopt_timing_polynomial_c301cb.npy` | `fd99551a46947c0bbfd00fe2699fec3d4534d74333909e17335ff6ac3940d5af` |
| `docs/NATIVE_TIME_LAW_20261117.md` | `18af021d52901b8785fb726fbbeb4c2dfbacea3859b02db8a8aaac3a00c5f21b` |
| `docs/INDEPENDENT_T1_FORMAL_QUALIFICATION_PLAN_20261117.md` | `cce158f5962b64226f27b29b27f1195094d7b0ae1100d6c44345f09828157907` |
| `runs/margins/time_inversion_audit_224paths.json` | `d58123b29ff71853c18a18ecd7f63bf480b3176b06bf3971dd038835d368999d` |
| `runs/margins/time_inversion_audit.log` | `6d57a3ab8060d1d04d470e93fb805e31d4edd3224822fad16e6af383085becfa` |

Pinned upstream c301cb00 (`curobo/_src/…`):

| file | sha256 |
|---|---|
| `curobolib/kernels/trajectory/timeopt/timeopt_common.cuh` | `abc4482ee4d568531c7d73bcbc6547cc4c8d4d53934de89ff5f523a4e9fe40ca` |
| `curobolib/kernels/trajectory/timeopt/timeopt_path_kernel.cuh` | `1a7ef9ea4971c22374824d9037609b94581234bf2aa64dc57e435ede60c7df06` |
| `curobolib/kernels/trajectory/timeopt/timeopt_path_helpers.cuh` | `c66a2f8eaa5a3e159b27aaec134852b1c12d8cb010eb309a146ca3e1bb682e77` |
| `trajectory/timeopt/timeopt_basis.py` | `d8906ed7a41c90a8b1580e63e4d055a5c1d67726462ab6154ebc854130cb1dad` |
| `curobolib/backends/cuda_core_backend/trajectory_timeopt.py` | `06b6f9b3fa272a367a52be643e74bfd2bebeafb257abe5c6d47dd8d283600ac9` |
| `curobolib/backends/cuda_core_backend/kernel_cache.py` | `726a5da9cb1db7ae16b706136a63c075158b08571fdaeab182a08490889e0835` |
| `curobolib/backends/cuda_core_backend/launch_helper.py` | `71d9e17542edfeb77a16122a0f9c7733844a9eb3c0278d2568768013e2db1100` |
| `curobolib/backends/cuda_core_backend/trajectory_config.py` | `1ca933be57ed708e4a3a1e86dccead239cae066f2d673ab9750b643e9341df7b` |

Reviewer artifacts (at time of writing):

| file | sha256 |
|---|---|
| `independent_checks/t1/exact_spline.py` | `5d1285e87f7619c30fcdc7ee422e5dcc7795f223a44d8fe25917ed288a87e637` |
| `independent_checks/t1/taylor_time_law.py` | `8d477a0812f52b7d402e6fde7a9971fdb3a8d8bb71ef1cf96c96fb8e09295afb` |
| `independent_checks/t1/crosscheck.py` | `d2b8e451c3229359155fbbcf6e639faf14947c265f0d25f0af2a3937d2d92c1a` |
| `independent_checks/t1/run_review.py` | `cae64f94c99471223d2b143fb230ef0e1c918bc730915157720fa5783a3f87cc` |
| `independent_checks/t1/gpu_probe.py` | `1a9a323a15e291b30a9797898ff5f8bd9f9e58502c6afefbe7ca309cfe684352` |
| `independent_checks/t1/corrected_inverse_audit.cuh` | `ae540d53f75fbe5c8b3ed52d5dd46a759af9ba1d07ed572f672defd555a28d81` |
| `independent_checks/t1/corrected_eps_run.py` | `2a8160c26691ff1320497efcaa52c397c1a4dd62d6b8b3a83536c8d59fa3c1da` |
| `tests/test_independent_t1_checker.py` | `f39b6759b0df44dd82e487a7d35d4e0d16ae58b2c42dbe712b9b6e09c6a10076` |

PTX of `inverse_audit` (reviewer compile, same options): `3b530a48ae015822a6d36aca13ae206d6d711b54c3276cb207039b5ac31ce1d5`. The 224 NPZ SHA-256s are those recorded in `time_inversion_audit_224paths.json`, all re-verified.


# R0 independent review, round 2: T1 v2 (commit a769113)

**Date:** 2026-11-17
**Verdict: R0 PASS for using T1 v2 in the formal cohorts**, under the scope of `docs/T1_V2_CONTRACT_20261117.md`. Before T1 v2 is used on formal cohorts, three conditions apply:

1. Freeze the reviewed file hashes listed in §5 at R1.
2. Gate every audited output on `status == AUDITED_CONSISTENT`, which needs every integrity counter and control to be clean.
3. Apply the three documentation corrections in §3. None of them affects soundness.

The round-1 unsound finding U1 is fixed and independently confirmed. None of the remaining UNKNOWNs affects the soundness of the T1 contract as scoped.

## 0. Reviewer and method
* **Reviewer and independence:** I am the same isolated Claude subagent (Opus 5.5) that did round 1. I did not write any v2 code, and I modified no existing file.
* **Review target:** a769113. HEAD is now `c1797b4`, which adds only frozen formal query pools and `scripts/write_formal_freeze.py`. No reviewed file changed (`git diff --stat a769113 HEAD`), and I did not open the query pools.
* **GPU:** RTX 3090 only. Every probe asserts the device name. I shared the GPU with the coordinator's job, and each of my jobs took 10 s or less except the 3-path corrected run.
* **Method:**
  1. Source inspection of every changed file.
  2. My round-1 independent checker re-run against the modified `time_law_grid`.
  3. My corrected reference kernel re-run on the v2 grid.
  4. New GPU probes: an exhaustive check of `exact_f64`, edge values of `key_ru`, and production −0 vs +0.
  5. A CPU recomputation of the beyond-duration bit range under the driver's numpy version.
  6. The coordinator's GPU tests.
  7. A reviewer-side run of `t1_binary_audit.py`.

## 1. Answers to the specific questions
* **Far-end bound.** It is sound. The kernel (`time_inverse_audit_v2.cuh:122-129`) reduces `e = max(s−lo, hi−s)`, rounded up with `__dsub_ru`, and that is an upper bound on |s−s*| for every s* in [lo, hi]. The driver's CPU cross-check (`native_time_audit_v2.py:229`) uses the same far-end formula; it is labelled a consistency check, not a proof. Independent confirmation is in §2.
* **Fail-closed bypasses.** I found none.
  * There is no early return before the barriers. Invalid threads take `valid=false`, contribute zeros and still reach the barriers (l.73-78, l.132-137).
  * `__syncthreads_count` is reached by every thread. The driver launches blocks of 256, a multiple of 32, so `__shfl_xor_sync(0xffffffff)` has every lane converged and active.
  * Fail-closed conditions force the key to `INFKEY|bits`, which is greater than every finite key. The driver then requires a finite ε and all counters clean (l.255-258), and requires processed to equal the domain size in both passes.
  * `bad_s` (l.86) catches negative values (including −NaN), values above 1, +Inf and +NaN. `!(lo<=hi)` catches both empty brackets and NaN.
  * `key_ru` is checked on the GPU (probe Q2). For 1e-45, 1e-40 and just below 2^-126 it returns the smallest normal (an over-report). For 0 it returns 0, for +Inf it returns +Inf, and for 1e-7 it rounds up.
  * The null `s_ref` in the beyond pass (`np.uint64(0)` passed as the pointer) only disables the progress-identity compare. No identity is claimed there: production emits the tail. The progress used is computed in the same kernel, so the "s_native = 1" in the beyond pass is observed, not assumed. The beyond pass reduces into the same keys and its own counters, and `beyond_counters_clean` is required.
* **`exact_f64`.** It is correct for every nonnegative finite bit pattern. By derivation: subnormals are m·2^-149 (`0x36A00000` encodes 2^-149, and the m·2^-149 product is exact), and normals rebias the exponent by +896 and shift the mantissa left by 29. Exhaustively: on the GPU, compiled with the cuRobo FTZ options from the audit source list, it matches the hardware float-to-double conversion (compiled without FTZ) on **all 2,139,095,040 patterns 0x00000000–0x7F7FFFFF, with 0 mismatches** (Q1). It is only applied to t in the domain and to s with `bad_s==0`.
* **Production identity check.** It is sound.
  * The upstream `sample()` takes batch, knots and dof from `controls.shape`, samples from `requested.shape[1]`, and intervals from `time.shape[1]−1`. The extra unit dimensions on `goal` do not matter because only pointers are passed.
  * The argument layout is validated by a bitwise replay of the saved S8193 and S32768 production diagnostics.
  * `progress_chunk_v2` writes the production layout `[4][count][dof]` with stride `count·dof`. The compare `p.view(int32) != h.view(int32)` is elementwise over identical shapes, and it compares NaN payloads bitwise.
  * The inverse kernel's progress is compared bitwise per time against the same chunk's harness progress (`s_ref`).
  * Result: production outputs = harness tail(s_h), and s_h is the progress whose ε is certified. The hidden production progress word itself is not compared. The contract states this in item 3, and it is scope only: the chain "reported = native evaluation at s_h ≈ model at s_h ≈ model at s*" does not need it.
* **Beyond-duration window bit range.** It is sound, but the code does not do what its comment says. Under numpy 2.1.3 (NEP 50), `np.float32(T_hi) >= T_hi` is evaluated in float32 and is always true, so the `nextafter` branch is dead. On **110/224 paths** the computed stop bit is RN32(T_exact_hi) < T_exact_hi, not "the first float32 ≥ T_exact_hi". It is still sound: no float32 lies strictly between RN32(T_hi) and T_hi, and every float32 t ≥ T_hi ≥ T(1) with t > T_native has s_native = 1 = s*(t), so the error is exactly 0. The window is non-empty on 72/224 paths, with at most 2 times. See `round2_beyond_window_check.py` and `beyond_window_check.json`. Fix the comment or use an exact `Fraction` comparison.
* **Grid summation bound.** It is rigorous. `np.cumsum` is sequential, which I verified in round 1, so each prefix satisfies \|fl(S_k)−S_k\| ≤ γ_{k−1}Σ_{i<k} t_i ≤ γ_N·S_N (Higham 4.4, nonnegative terms). `math.fsum` returns the sum of the binary64 `hi_int` values rounded to nearest, and even if it were only faithfully rounded, `up(fsum)` would still be ≥ S_N. `gamma` is rounded up, and `E = up(gamma·S_hi)`. `lo_int ≤ hi_int` is now checked explicitly, so E also bounds the T_lo prefix error. The h³/24 rounding is now `up(...)`. Re-check: §2.
* **Contract text.** It is accurate and does not overclaim, with three corrections needed (§3).

## 2. Independent confirmations (numbers)
* **Grid tables after the change.** On all 224 paths at 2^22, the v2 `time_law_grid` (sha `f2c410b7…`) passes my exact-rational containment checks:
  * T_lo 168,436 and T_hi 168,436 PROVED;
  * fmin 168,212 and fmax 168,212 PROVED;
  * inverse brackets 18,144 lower + 18,144 upper PROVED;
  * 0 violated, 0 undecided, 0 empty;
  * full-array structure checks OK.

  Output: `cpu_review_v2grid.json`. My 25 round-1 tests pass unchanged against the new grid.
* **ε_v2 vs my corrected ε.** On every reissued path available at review time (21 paths, `runs/t1_v2_reissue/run.log`), **ε_v2 equals my round-1 corrected ε bit for bit**. For example, margin-20261117-0000-margin_0p900 gives 1.9171518772509444e-07 in both, at t=0.17535433173179626. Re-running my corrected reference kernel on the **v2 grid** for the first 3 paths gives identical values again (`corrected_eps_v2grid_first3.json`).
* **Lower-bound check.** My exact-rational checker's rigorous error at each round-1 argmax time is ≤ the corrected ε on 224/224 paths. For 0p900 that error is ≤1.9110e-7 ≤ 1.9172e-7. So ε_v2 is not contradicted at any independently evaluated point.
* **Coordinator GPU tests.** `tests.test_t1_audit_v2_gpu`: 6/6 OK. They cover the ragged-chunk count, subnormal and zero times, the progress-mismatch, empty-bracket and U1 regressions, and bit-exact time conversion.
* **Binary audit (reviewer run).** `t1_binary_audit_reviewer_run.json`: sm_86, NVRTC 12.9.
  * The inverse kernel SASS contains DADD.RM×2, DADD.RP×4, DFMA.RM/RP×2, DMUL.RM/RP×1 and F2F.F32.F64.RP×3.
  * There is no FTZ f32→f64 conversion.
  * Recompilation is deterministic. SASS SHA prefixes: production `b78b0b0c…`, audit `3f95a85b…`.
* **Signed zero.** Production output at −0.0 is bitwise identical to +0.0 on 2 paths (Q3), and +0 is enumerated.

## 3. Obligation matrix, v2 (numbering as in round 1)

| # | Obligation | Round 1 | v2 | Evidence / note |
|---|---|---|---|---|
| 1.1 | exact float32 → rational spline | PASS | PASS | unchanged; 0 mismatches on 224 paths |
| 1.2 | x>0 | PASS | PASS | 224/224 |
| 1.3 | C2/endpoints, strict rest | PASS | PASS | unchanged |
| 1.4 | interval Horner and rounding direction | PASS | PASS | full re-check on the v2 grid (§2) |
| 1.5 | midpoint remainder | PASS (doc nit) | PASS | `up(h³/24)`; comment fixed |
| 1.6 | knot-crossing cells | PASS, latent N<32 | PASS | `log2_cells ∈ [5,26]` enforced (time_law_grid.py:108-109) |
| 1.7 | monotone and finite prefixes | PASS | PASS | now also checked explicitly, failing closed |
| 1.8a | accumulation bound sound | PASS | PASS | fsum/γ_N argument (§1) |
| 1.8b | justified in code; fail-closed intermediates | FAIL (doc) | PASS | justification comment; finiteness, positivity, lo≤hi, fmin≤fmax and monotonicity checks |
| 1.9 | inverse bracket edges | PASS | PASS | 36,288 endpoints PROVED on the v2 grid |
| 1.10 | adversarial suite | PASS | PASS | 25/25 on the v2 grid |
| 1.11 | randomized families | PASS | PASS | |
| 2.1 | bit domain and counts | PASS | PASS | processed counter; driver requires equality |
| 2.2 | excluded inputs documented | FAIL (scope) | PASS (doc nit D1) | documented; the −0 rationale is wrong (see D1) |
| 2.3 | completed-work and launch checks | FAIL | PASS | `__syncthreads_count`; per-chunk sync; integrity gate |
| 2.4 | same code as production | UNKNOWN | PASS (scoped) | production kernel itself; full-domain bitwise compare of outputs; progress compared between audit kernels; the hidden production s word is not compared (scope, contract item 3) |
| 2.5 | directed rounding | PASS (PTX) | PASS | SASS .RM/.RP inventory; deterministic recompile |
| 2.6 | **ε is an upper bound** | **FAIL (UNSOUND)** | **PASS** | far-end bound; equals my corrected ε on 21/21 paths, and on the v2 grid for 3/3 |
| 2.7 | key packing, NaN, FTZ | PASS / latent fail-open | PASS | INF forcing; `key_ru` probe Q2 |
| 2.8 | empty bracket detected | FAIL | PASS | counter plus INF; fault-injection test |
| 2.9 | reported audit (limits, q, joint, production ops) | FAIL | PASS (nit D2) | production outputs; float32-exact compare; q/v/a/j finiteness; joint recorded |
| 2.10 | separate replay of maxima | PASS | PASS | round-1 replays; corrected kernel on the v2 grid |
| 2.11 | reduction determinism | PASS | PASS | warp max with full mask and converged lanes; atomicMax is order-independent |
| 2.12 | SASS / A1–A5 / subnormal t | FAIL (doc) / UNKNOWN | PASS for T1; UNKNOWN outside T1 | audit kernels' SASS inventoried; `exact_f64` exhaustive; subnormal t now gives a sound tiny bound. A1–A5 still matter only for E_same-progress (the analytic NATIVE_REPORTED contract), not for ε or the exhaustive reported values |
| 3.1 | progress → q/v/a/j mapping | PASS (conditional) | PASS | asserts replaced by explicit checks; ε sound |
| 3.2 | tail/hold windows | FAIL (scope) | PASS (nit D3) | inversion audited on (T_native, T_exact_hi]; tail semantics stated; bit range sound (§1) |
| 3.3 | "physical derivatives" wording | FAIL (doc) | PASS | erratum; contract "Not claimed" |
| 3.4 | headline claims and traceability | FAIL / UNKNOWN | PASS for method; final numbers pending | the reissue had 21/224 rows when reviewed, all AUDITED_CONSISTENT; the 224-path ε/200-of-224 figures must be quoted only from the completed v2 run. SASS hash and options recorded |
| 3.5 | 0.99: 28/29 vs 29/29 | PASS | PASS | unchanged explanation |

### Required documentation corrections (none affects soundness)
* **D1** (`T1_V2_CONTRACT_20261117.md`, excluded domain): −0.0 is **not** the tail branch, because `requested < 0.0f` is false for −0. It is the non-tail branch with s=0, and its output is bitwise identical to +0, which is enumerated (Q3). Reword it as "−0.0 is equivalent to +0.0, which is audited". Negative t is correctly described.
* **D2** (driver l.262-263): non-finite production outputs yield the label `REPORTED_EXCEEDS_LIMITS`. That fails closed but is misnamed. Use a distinct `REPORTED_NONFINITE`.
* **D3** (driver l.189-196 comment, contract item 2): the window runs to RN32(T_exact_hi), not to "the first float32 ≥ T_exact_hi", on 110/224 paths (numpy 2 makes the branch dead). Fix the code with an exact rational comparison, or fix the comment, and state why it is sound. In the window, production reports v=a=j=0. Item 2's L·ε statement concerns model values, not reported ones. One added sentence would prevent misreading.

## 4. Remaining UNKNOWNs
None of these affects the soundness of the T1 contract as scoped.

| Item | Effect |
|---|---|
| Hidden production progress word vs harness progress (outputs are bitwise identical) | scope only (contract item 3) |
| Recorded SASS hash comes from a deterministic recompile with identical options, not from the cubin object loaded by the kernel cache | traceability |
| Silicon and driver correctness of the IEEE directed-rounding f64 division sequence | assumption shared by all GPU interval work; scope |
| A1–A5 for the analytic E_same-progress bound | affects the NATIVE_REPORTED analytic contract, not T1 |
| Completed 224-path v2 reissue | results provenance only |

## 5. SHA-256 of reviewed files (review target a769113; HEAD c1797b4 identical for all of these)

| File | SHA-256 |
|---|---|
| `src/curobo_certify/native/time_inverse_audit_v2.cuh` | `fab9450549764262087be1cc41c4eff600921f9f80f950df1bd1a22ef6603d38` |
| `scripts/native_time_audit_v2.py` | `ab7760efe966945911f4684d1500c07bc8aa90a16616684184113952313511f7` |
| `scripts/t1_binary_audit.py` | `19702692727358e98fd8cfa23c5be69809a8fa0986fa165791ef133e59950f27` |
| `src/curobo_certify/time_law_grid.py` | `f2c410b76644d51b11dafd458c3eb3f5b9b1551ead2afeff5e0e1f7c12c655d3` |
| `scripts/native_time_inversion_audit.py` | `6bb9b146925d5e84c15f9a2e00e8edaac1ffe054675f45573f9884a2301e8307` |
| `docs/T1_V2_CONTRACT_20261117.md` | `5190ad0ea8173c2b9e8d37fb27f36e02cf9b125f94b43fb067535229d1e24b2e` |
| `docs/NATIVE_TIME_LAW_20261117.md` | `bc44ca301731c6af1bf464bb211e0d8f89253630f751e18276e4697f9a2b5235` |
| `tests/test_t1_audit_v2_gpu.py` | `c918bfa0fc5bcc49bfcd3c9d621983ec8fcca84a44543d79a5cc789aac187abd` |
| `src/curobo_certify/native/same_s_harness.cuh` (unchanged) | `206cd38634b54d6044dbe700fb102b2e999dcd520ea97b2bb9209d973b4ad958` |
| `scripts/native_same_s_falsify.py` (unchanged) | `bbbaef43e10db923ed221d8586d2c881bcb2ad91d36eda775bee58e1c75500da` |
| `runs/t1_v2_reissue/run.log` (21 rows at review time) | `f4d46bb5c22887a4540ce5227cbda74fb8375066b87bf6f5a27d0371a5c9bb82` |

Pinned upstream files are unchanged from round 1 (hashes in `../REVIEW_MEMO.md` §6).

Reviewer artifacts for this round:
* `independent_checks/t1/round2_probe.py`
* `independent_checks/t1/round2_beyond_window_check.py`
* `docs/reports/r0_independent_t1_review/round2/` containing:
  * `cpu_review_v2grid.json`
  * `corrected_eps_v2grid_first3.json`
  * `round2_probe_results.json`
  * `beyond_window_check.json`
  * `t1_binary_audit_reviewer_run.json`

## 6. Commands
```
PYTHONPATH=src:. python independent_checks/t1/run_review.py --probe /nonexistent --out docs/reports/r0_independent_t1_review/round2/cpu_review_v2grid.json --jobs 10
PYTHONPATH=src:. python -m unittest tests.test_independent_t1_checker            # 25 OK
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src conda run -n curobo-timeopt-c301cb python -m unittest tests.test_t1_audit_v2_gpu -v   # 6 OK
CUDA_VISIBLE_DEVICES=1 conda run -n curobo-timeopt-c301cb python scripts/t1_binary_audit.py --out docs/reports/r0_independent_t1_review/round2/t1_binary_audit_reviewer_run.json
CUDA_VISIBLE_DEVICES=1 conda run -n curobo-timeopt-c301cb python independent_checks/t1/round2_probe.py --out docs/reports/r0_independent_t1_review/round2/round2_probe_results.json
CUDA_VISIBLE_DEVICES=1 conda run -n curobo-timeopt-c301cb python independent_checks/t1/corrected_eps_run.py --first 0 --count 3 --out docs/reports/r0_independent_t1_review/round2/corrected_eps_v2grid_first3.json
CUDA_VISIBLE_DEVICES="" conda run -n curobo-timeopt-c301cb python independent_checks/t1/round2_beyond_window_check.py
```

# Independent T1 review and formal Panda-only qualification: results

Date 2026-11-17. Branch `exp/native-t1-formal-qualification-plan-20261117`.
Binding protocol: `docs/INDEPENDENT_T1_FORMAL_QUALIFICATION_PLAN_20261117.md`.
Interactive report: `reports/formal_qualification_report.html` (offline, 8.5 MB).

## 1. R0: independent review of the native time-law audit

- **Reviewer:** an isolated Claude subagent that took no part in writing the code.
- **Independent checker:** written by the reviewer in `independent_checks/t1/`. It uses exact rationals, integer-isqrt square-root enclosures and a Taylor-model integrator. It does not reuse any formula from `time_law_grid.py`. It has 25 tests.
- **Memos:** `docs/reports/r0_independent_t1_review/REVIEW_MEMO.md`, `round2/REVIEW_MEMO_ROUND2.md` and `round2/ADDENDUM_D1_D3.md`.

**Round 1: FAIL. The bound was unsound (finding U1).**
- The v1 GPU kernel reduced `max(s−hi, lo−s, 0)`. That is the distance from the native progress to the rigorous bracket, which is a lower bound on |s_native − s*|, not an upper bound.
- The published "certified ε ≤ 2.44·10⁻⁷" was therefore not a bound on any of the 224 outputs. The reviewer proved this at each output's recorded argmax time.
- The CPU time-law enclosure and the inverse bracket passed every independent check:
  - 168,436 T_lo and T_hi entries per grid were proved to contain the exact values;
  - 36,288 bracket endpoints were proved;
  - there were 0 violations and 0 undecided cases.
- Several fail-open paths were also found:
  - no completed-work counter;
  - an empty bracket went undetected;
  - a NaN progress mapped to zero error;
  - the reported-value audit ran a harness copy of the output expressions, not production;
  - the excluded domain was undocumented;
  - the recorded compile flags were not the options NVRTC actually received.

**Fix: T1 v2.** The contract is in `docs/T1_V2_CONTRACT_20261117.md`.
- The bound now uses the far end of the bracket, `max(s−lo, hi−s)`, rounded upward.
- It fails closed: it counts processed times, empty brackets, NaN or out-of-range progress, and harness/reference mismatches.
- float32 values are converted to binary64 exactly from their bit patterns, independent of FTZ.
- The reported-value audit runs the production `sample_with_progress_kernel` itself, over every float32 time. Harness and production are compared bitwise on the whole domain.
- The time inversion is also audited on the window between T_native and T_exact.
- The grid summation bound uses fsum with no empirical slack factor.
- The SASS identity is recorded from the exact cache options.

**Round 2: PASS** for use in the formal cohorts.
- ε_v2 equals the reviewer's independently corrected ε bit for bit.
- 224 of 224 argmax errors are bounded, as checked by the exact-rational checker.
- `exact_f64` was checked exhaustively on 2.14·10⁹ bit patterns.

**Re-issued historical result.** Over 224 outputs and 2.364·10¹¹ float32 times:
- all 224 are `AUDITED_CONSISTENT`;
- max ε is 2.448771·10⁻⁷, at most 1.28·10⁻⁹ above the withdrawn v1 values;
- 200 of 224 report values within limits; the 24 exceedances are the margin-1.00 outputs with exact violation witnesses;
- 0 output words differ between production and harness.

The v1 files and results are kept as evidence and carry an erratum.

## 2. R1: frozen population and implementation

**Population.**
- UR5e is still blocked: the pinned cuRobo has no UR5e model, and no sourced acceleration or jerk limits exist. The original dual-robot protocol therefore stays `BLOCKED_UR5E_MODEL_LIMITS`.
- A separately named Panda-only protocol was preregistered instead: **AMENDED_POPULATION_PANDA_ONLY**, with cohorts Panda-Dev40, Panda-Holdout100 and Panda-Confirm300.
- The sample unit is K eligible primary-arm native successes per cohort. They are taken by a deterministic first-K rule from a frozen, oversubscribed, ordered pool of 70, 175 and 525 queries respectively. Every attempt is kept in the ledger, and nothing is replaced.
- Strata are 4 scenes × 3 motion magnitudes, in seeded blocks.
- Seeds are derived per cohort with SHA-256.
- Leakage audit against all 268 earlier Panda queries: PASS. The smallest L∞ goal distance is 0.042 rad.
- Start configuration (planner HOME) and the four scenes are fixed strata shared with all earlier studies.

**Arms.**
- Primary: the unmodified TimeOpt 0.95 constructor.
- Secondary: 0.99 on the same source queries, reported separately.

**Frozen elements.**
- 30 source files, by SHA-256.
- The deterministic compiled K3 producer and verifier (binary SHA `87b5def0…` / `75a62afa…`).
- The SASS identities of the T1 audit and the production sampler (RTX 3090, sm_86, NVRTC 12.9).
- Policies, denominators and thresholds.

These are recorded in `docs/reports/formal_panda/FREEZE.json`, which has a history section. The gate amendments that opened each cohort are recorded in sequence.

**Amendments.**
- One non-soundness tooling fix during Dev (v1.1): `merge_cert_bench.py` now compares build identity instead of temporary paths.
- No algorithm, tolerance or threshold change was made at any stage.

**Nondeterminism.** The pinned planner is not bit-reproducible across processes. A re-solve of one query gave 0.289314 s against the frozen 0.289613 s. The unit of certification is therefore the registered solve's frozen NPZ.

## 3. Results

Primary arm 0.95 versus physical limits:

| Cohort | Attempted / native failures / eligible | MODEL_K3 | NATIVE_REPORTED_K3 | T1 audited, reported within limits | Exact violations | One-sided 95% CP lower bound | Max ε |
|---|---|---|---|---|---|---|---|
| Panda-Dev40 | 43 / 3 / 40 | 40/40 | 40/40 | 40/40 | 0 | 0.9278 | 2.42e-7 |
| Panda-Holdout100 | 122 / 22 / 100 | 100/100 | 100/100 | 100/100 | 0 | 0.9705 | 2.62e-7 |
| Panda-Confirm300 | 357 / 57 / 300 | 300/300 | 300/300 | 300/300 | 0 | **0.99006** | 2.78e-7 |

Secondary arm 0.99, same source queries:

| Cohort | Eligible | MODEL_K3 | NATIVE_REPORTED_K3 | T1 | Paired duration gain vs 0.95 (bootstrap 95% CI, n pairs) |
|---|---|---|---|---|---|
| Dev40 | 40 | 40/40 | 40/40 | 40/40 | 2.40% [1.83, 2.75], 40 |
| Holdout100 | 99 | 99/99 | 96/99 (3 `UNKNOWN_TIGHTENED_WITNESS`) | 99/99 | 2.32% [1.87, 2.61], 99 |
| Confirm300 | 300 | 300/300 | 295/300 (5 `UNKNOWN_TIGHTENED_WITNESS`) | 300/300 | 2.18% [2.06, 2.44], 299 |

**Correctness and soundness**
- **Soundness:** zero hard stops in every cohort:
  - no false CERTIFIED, checked for MODEL_K3 against exact witnesses and for NATIVE_REPORTED against exhaustive and sampled production exceedances;
  - zero verifier disagreements;
  - zero T1 integrity failures;
  - zero NPZ or source hash mismatches.
- **T1 totals:** 879 formal outputs and 6.33·10¹¹ (Confirm) + 2.11·10¹¹ (Holdout) + 0.84·10¹¹ (Dev) float32 times. Production and harness agreed bitwise on every word.
- **Model-state deviation from the inversion error:** the shift L·ε_s of the exact model's v, a and j is at most 9.4·10⁻⁶, 9.9·10⁻⁶ and 4.1·10⁻⁴ of the limits respectively.
- **Secondary-arm unknowns:** the 8 `UNKNOWN_TIGHTENED_WITNESS` outputs reported within limits at every float32 time. This reflects conservatism of the analytic FP32 bound against the tightened limits, not an error. They are counted as UNKNOWN, not as successes.
- **Planning contract** (the solver's own 0.95-scaled limits; secondary and descriptive): outputs genuinely violate it in 14/40, 24/100 and 71/300 cases, each with an exact witness. All the same outputs are certified against the physical limits.

**Coverage inference**
- The preregistered inferential criterion is a one-sided 95% exact lower bound ≥ 0.99. It is met only by Confirm300: 300/300 gives 0.99006.
- This assumes independent source queries. The population shares one start configuration and four fixed scenes; the per-scene counts are in the report.
- The two-sided 95% interval is [0.9878, 1].

**Online fresh added latency** (per-query median of 10 randomized paired repeats, 8 threads, NATIVE_REPORTED contract; targets p50 ≤ 1 ms and p95 ≤ 3 ms):

| Cohort / GPU | All attempted p50 / p95 (registered) | Certified queries only p50 / p95 | Label |
|---|---|---|---|
| Dev40 / RTX 3090 | 2.03 / 2.17 | 2.03 / 2.18 | FAILED_PARTIAL_TARGET (p95 met) |
| Holdout100 / RTX 3090 | 2.03 / 2.06 | 2.04 / 2.06 | FAILED_PARTIAL_TARGET (p95 met) |
| Confirm300 / RTX 3090 | **2.38 / 3.71** | 2.57 / 3.73 | FAILED_PARTIAL_TARGET (**p95 missed as measured**) |
| Confirm subset / A100 | 2.06 / 2.15 (40 queries) | same | p95 met, p50 missed |

- **Stage medians:** plan 7.6–8.5 ms, device→host 0.05 ms, FP32 native bound 1.25–1.7 ms, producer 0.46–0.63 ms, verifier 0.35–0.46 ms.
- **Confirm300 timing contamination:** an unrelated concurrent Claude session (`/home/pkicki/curobo_analysis`) ran CPU-bound jobs during the Confirm300 benchmarks.
  - Load average was 4.97 and 6.76 (5- and 15-minute) at the end of the run.
  - In execution order, the first two-thirds measured p50 2.6–3.0 ms with the bound stage at 1.7–2.0 ms. The last third measured p50 2.07 ms and p95 2.5 ms with the bound stage at 1.28 ms, matching Dev and Holdout.
  - The registered result is kept, not replaced, and is flagged `LATENCY_MEASURED_UNDER_UNCONTROLLED_EXTERNAL_LOAD` in `docs/reports/formal_panda/Panda-Confirm300/latency_contamination_note.json`.
  - The same session ran partly during the Holdout100 timing, but those numbers stayed tight.
- **Run-to-run output differences:** in the fresh benchmark, 2, 7 and 13 queries (Dev, Holdout, Confirm) returned more than one distinct native output across repeats. These are reported, not averaged away.

**Offline T1 audit cost per output**
- RTX 3090: p50 9.2–10.4 s and p95 ≤ 11.7 s, covering grid construction, the saved-diagnostic replay, GPU enumeration, CPU cross-check and Lipschitz bounds.
- A100: p50 6.4 s.

This cost is never part of the online latency.

**Frozen-output proof cost** (CPU, 1 thread, per-path medians, Confirm300 primary):

| Contract / method | Certified | Per-path median (ms) |
|---|---|---|
| physical H0 | 296/300 | 3.20 |
| physical adaptive A | 300/300 | 3.44 |
| physical U6 | 300/300 | 95.1 |
| native-reported A (incl. 1-thread FP32 bound) | 300/300 | 12.7 |

Exact tightness: 272 of 600 Confirm outputs needed adaptive subdivision for at least one residual.

**A100 replication** (preregistered subset: the first 40 eligible primary Confirm300 outputs plus their 40 secondaries)
- The A100 has its own sm_80 SASS identity and its own exhaustive FP32 unit audit, identical to the RTX 3090: reciprocal 0.975 ulp, sqrtf 0.944, rsqrtf 1.51, division by 6 0.667.
- T1 v2 on the A100: 80/80 outputs consistent and within limits. ε and the reported maxima are bit-identical to the RTX 3090 values.
- Fresh requests on the A100: 40/40 certified.
- These are A100 measurements; nothing was transferred from the RTX 3090.

## 4. Verdicts

1. **PAPER: GO.** The paper rests on:
   - a pre-registered, independently reviewed audit, including a documented unsound bug that was found and fixed;
   - 300/300 confirmatory coverage on three distinct contracts, with complete failure ledgers;
   - genuine exact violation witnesses in the margin study;
   - an honest latency result.
2. **MODEL_K3 certification: QUALIFIED.** This applies only to the amended Panda-only population (fixed start, four scenes), TimeOpt 0.95 and physical limits. The confirmatory one-sided bound of 0.99006 assumes independence between source queries.
3. **NATIVE_REPORTED_K3 analytic certificate:**
   - **Primary 0.95 arm: QUALIFIED** within assumptions A1–A5 and the audited binary and GPU (300/300).
   - **Secondary 0.99 arm: PARTIAL** (295/300, 5 UNKNOWN).
4. **Per-output exhaustive T1 float32-time audit: AUDITED_CONDITIONAL.** 879/879 formal outputs passed. The audit costs about 10 s per output offline on the RTX 3090 and about 6 s on the A100.
5. **Online latency targets:**
   - p50 ≤ 1 ms: FAILED in every cohort and on both GPUs.
   - p95 ≤ 3 ms: met on Dev40, Holdout100 and the A100 subset; missed as measured on Confirm300, under contamination.
   - Overall: FAILED_PARTIAL_TARGET.
6. **Deployment: NO-GO.** Reasons:
   - the latency targets are not met;
   - the online contract is NATIVE_REPORTED_K3 only, and T1 is an offline per-output audit;
   - no controller interpolation contract exists.
7. **NOT_CLAIMED:**
   - arbitrary unseen trajectories beyond the population statistic;
   - between-request interpolation and physical-time derivatives of the executed command;
   - dynamics, torque, torque rate and collision;
   - GPUs and binaries other than those audited;
   - the original dual-robot qualification;
   - real-robot tracking.

## 5. Limitations

- Single robot, a single start configuration and four scenes.
- The independence assumption is approximate.
- The planner is nondeterministic across processes.
- Confirm300 latency was measured under external load.
- The T1 audit's progress linkage to production rests on identical outputs, not on the hidden progress word.
- The A1–A5 FP32 assumptions are audited per GPU and binary, not proved for silicon.
- The independent reviewer is an AI agent, not a human expert.

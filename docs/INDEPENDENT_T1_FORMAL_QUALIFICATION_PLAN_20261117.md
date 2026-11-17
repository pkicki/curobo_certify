# Independent T1 audit and gated formal certification qualification
**Date:** 2026-11-17  
**Status:** PLAN ONLY; no new formal cohort has been generated, inspected or run in this planning commit.  
**Parent:** exp/margin-duration-native-time-20261117 @ fe3dd7c47b24c6475a1b847e1a5a9802c7eb86af  
**Work branch:** exp/native-t1-formal-qualification-plan-20261117  
**Execution authority:** The user authorizes the coding agent to implement, test, independently review, freeze and then execute stages that pass preregistered gates. Do not seek routine approval; never bypass a correctness blocker.

## A. Executive decision and evidence

**Decision:** CONDITIONAL GO for formal certification qualification. An independently checked T1 audit and immutable cohort preregistration are *required before* Dev40; Holdout100 and Confirm300 are separately gated. There is **NO GO** for unqualified deployment or for a generic "physical-time derivative of executed robot motion" claim.

The exploratory results at fe3dd7c are:
- 180 source requests = 36 source problems x {0.90, 0.95, 0.975, 0.99, 1.00}. 145 successes, 29 per arm; the same seven source problems fail all arms.
- Exact mathematical model feasibility among 29 successes/arm: 29 / 29 / 29 / 29 / 17, respectively. Margin 1.00 has 12 proven physical-limit violations. Registered paired duration gain vs 0.95: margin 0.975 +1.34%, 0.99 +2.33%, 1.00 +2.93%, n=29; exploratory, *not* formal qualification.
- Soundness of the fast native-reported analytic certificate: 29/29/29/28/5 among these 29/arm, conditional on native arithmetic assumptions. Do not confuse the 0.99 analytic 28/29 with its 29/29 mathematical feasibility or the separately exhaustive-reported 29/29.
- 224 existing outputs exhaustively audited over 2.364e11 nonnegative float32 requested-time encodings with an observed/certified audit bound |s_native-s_exact| ≤ 2.44e-7 and 200/224 reported within physical limits. This guarantee is per-path, per-binary/GPU, and is subject to the CPU interval/grid and GPU audit implementation being sound. The 24 reported overshoots are the margin-1.00 violating outputs.
- Full producer and independent verifier with optimized native-bound stage have *added* fresh request p50 2.04ms and p95 2.06ms (RTX3090, 8 CPU threads), not ≤1ms p50; the formal p95≤3ms engineering goal is provisionally met. Separate exhaustive per-trajectory time-law audit reportedly costs ~5s GPU plus ~10s CPU per path. DO NOT include this offline audit in a 2ms online certification claim.

The old full campaign, docs/CAMPAIGN_PLAN.md, registered Dev40/Holdout100/Confirm300 as 20/50/150 paths per robot Panda + UR5e. **UR5e acceleration/jerk limits and supported model were never verified**. Never silently convert those dual-robot cohorts to Panda-only while preserving their original name/claim. Population decision must be frozen under Gate R1 below.

## B. Non-negotiable assertion/claim taxonomy

Maintain three separately named contracts, acceptance columns, scripts and status fields:
1. MODEL_K3: exact serialized q(s), x(s) continuous-time velocity/acceleration/jerk feasibility (mathematical reference t*=T(s)). Requires positivity, endpoint, regularity and independent proof replay.
2. NATIVE_REPORTED_K3: reported FP32 v/a/j at every admitted native progress s; conditional bound against original physical limits, source and ISA assumptions, model-to-native tightening. An exhaustive reported-time audit is a separate expensive offline validation, not a 2ms online verifier.
3. T1_AUDITED_FLOAT32_TIMES: for **each audited frozen trajectory on the audited executable and GPU**, reported q/v/a/j for every admissible finite nonnegative float32 requested time are enclosed relative to a mathematically defined reference q*(t)=q(s*(t)); supported by a rigorously checked CPU inverse bracket, exact enumeration and native-production identity. **No specified between-request interpolation; not a proof that the true derivative of the discrete native sample stream equals the reported derivative arrays.** Not robot tracking or arbitrary GPU/binary or unseen trajectory certification.

Do not label any contract unqualified PHYSICAL_TIME_CERTIFIED. The old terminology NATIVE_PHYSICAL_TIME_LINKED must carry explicit per-path/sample-domain/assumption qualifiers. Certified by compiled K3 + independent verifier is conditional on compiler/interval arithmetic assumptions, not formal verification of the verifier. Positivity and endpoint behavior must be checked for each path. Torque/torque-rate/collision/dynamics unsupported here. An EXACT MODEL violation does not automatically prove native-sampled violation, and a tightened-limit witness is UNKNOWN_TIGHTENED_WITNESS, not a native violation.

## C. R0 — mandatory independent technical audit of T1 (not reused implementation)

Inspect and cross-check:
- docs/NATIVE_TIME_LAW_20261117.md
- src/curobo_certify/time_law_grid.py, time_law_enclosures.py, native_error_bound.py
- scripts/native_time_inversion_audit.py, sass_audit.py, fp_unit_audit.py, native_same_s_falsify.py
- src/curobo_certify/native/time_inverse_audit.cuh, pinned NVlabs/curobo c301cb00a87012c2808b1a60649c4065c9b3c95a
- source + compiler settings of compiled K3 producer and independent verifier.

An independent reviewer must produce a **separate written proof-obligation review** and at least one independently written mathematical reference/test checker (different formulas/implementation from the code under examination, not just invoking the same utility). Prefer a second isolated reviewer/agent or knowledgeable human. If independent review is unavailable, mark R0 BLOCKED_INDEPENDENT_REVIEW and do not release formal cohorts. Coding agent may prepare artifacts/tests without inspecting them on holdout.

### R0.1 Exact time-law grid correctness
For T(s)=integral_0^s x(u)^(-1/2) du:
- Prove exact float32 input -> rational cubic coefficient enclosure and common-knot representation; prove x>0 for each span, endpoints, joints and every represented case. Check structural C2/knot and strict-rest vs stationary-geometry endpoint contracts.
- Verify interval Horner enclosure, binary64 nextafter semantics for all numpy paths, sqrt/division direction, midpoint remainder h^3 sup|f''|/24, handling of cells crossing timing knots, and T_lo/T_hi prefix monotonicity.
- **Explicitly inspect the accumulation safety** in time_law_grid.py: recursive np.cumsum/np.sum, the gamma_N times sum bound and any heuristic widening such as *(1+1e-9). Require a mathematically justified outward upper bound, not an empirical fudge factor. Ensure every intermediate is finite, nonnegative and monotone or fail closed.
- Independently verify inverse_bracket inequalities, binary search branches, t=0, t=T_exact endpoints, T_native < T_exact and >T_exact, saturating s*(t)=1 after the model duration, no index -1 or N overflow.
- Test constant x, affine x (closed-form integral), positive cubic x, near-zero x and strict-rest endpoint substitution, x≈0 internal knot, degree/root multiplicity, knot straddles, rounding/subnormals, and grid resolutions 2^8 / 2^12 / 2^16 / 2^22. Build independent exact-rational sqrt enclosures via integer isqrt, MPFR/arb intervals or a separately implemented rigorous interval integrator. High-precision decimal alone is diagnostic, not a proof.
- Falsify by adversarial randomized rational polynomial families, including exact inverse enclosure spot checks and interval endpoints, not by dense samples alone.

### R0.2 Exhaustive native-time enumeration / GPU kernel audit
- Prove the float32 bit-domain is exactly the admitted API domain [0,T_native] (positive zeros, positive subnormals, normals, endpoint included); test -0 and out-of-range/NaN/Inf/negative times separately as excluded or documented. Count bit patterns exactly and prove no omitted or duplicated chunk, including start/end boundary and uint32 overflow. Audit block and grid launch coverage, asynchronous stream/synchronization, error checking and reset before each path.
- Prove that the audit calls *the same pinned progress_at_time and tail expressions* as the production sampler with identical configuration (CUDA flags, NVRTC source order, compiled PTX/SASS, driver/device); bitwise production trace parity is a necessary control, not by itself a proof that all paths are identical.
- Independently verify largest_le/largest_lt and inverse-bracket arithmetic with directed-rounded f64 intrinsics in source and SASS. Verify max/argmax packing into atomicMax is monotone for nonnegative finite f32 results, upper-rounded conversion, no dropped NaNs/Inf, proper 64-bit counters, and no overflows. Ensure every CUDA launch was checked for failure, every chunk synchronized where needed, and expected total enumerations equal actual completed work (not just sum of requested blocks).
- Reported_audit must compare the physical limits, use the same production operations, check q as well as v/a/j for finiteness where relevant, and capture *which* family/joint/time exceeded. Cross-check select chunk-level maxima by a separate slower CPU/harness replay. Audit potential unordered GPU reduction effects and nondeterminism; each output is replayed on the target GPU.
- Verify SASS and rounding assumptions A1-A5 on exact production compiled binary and NVRTC flags, including FFMA and .FTZ and input normality. The 2.13e9 positive-normal FP32 instruction audit supports certain operations only; test subnormals, signed zero, underflow/overflow and other input classes by explicit fail-closed preconditions or by bounded arithmetic. Never generalize empirical instruction tests to all hardware or all compiled binaries.

### R0.3 Contract semantics and traceability
- Derive rigorously the progress bound and map it through bounded q(s), v(s), a(s), j(s) with correct derivative factors and jump handling. Check per-span Lipschitz bounds and two-sided knot limits, output limitations when s or t exceeds the model interval, and endpoint hold behavior.
- Distinguish actual physical derivatives of continuous reference q*(t) from reported native arrays evaluated at quantized progress. The latter are sampled quantities; a controller interpolation/hold contract is REQUIRED before claiming physical time derivatives of a commanded trajectory. Do not derive derivative equality from progress proximity alone.
- Independently challenge 224-path 2.44e-7 maximum and 200/224 reported feasible claim; verify each source NPZ SHA, log digest, target compiled binary, hardware identity, requested-time counts and status criteria. Re-run a registered representative audit after fixing any bug.
- Assess whether 0.99's 28/29 analytic native-reported certificates vs 29/29 exhaustive time audit indicate conservatism, a bug, or just different contracts; do not loosen bounds without sound proof.

**R0 PASS** only with reviewer identity/method, signed (as documented) independent findings, a numbered proof-obligation matrix (PASS/FAIL/UNKNOWN), independent tests and exact source/binary/toolchain/hardware hashes. Any FAIL/UNKNOWN affecting soundness or scope -> STOP, repair and re-review, no use of formal cohorts as debugging data.

## D. R1 — freeze contract, code, populations and objective decision rules BEFORE Dev40

Lock after R0 passes: source Git SHA, native pinned cuRobo SHA, compiler flags and versions, NVRTC/SASS and executable hashes, math proof/verifier/bounds, test hashes, scripts, CPU/GPU architecture, OMP_NUM_THREADS/affinity, benchmark warmups/repeats, source query generator, sampling seed, sample identity and scene strata, error-bound pieces, max depth/nodes, policies, limits, tolerance, invalid/unknown handling, and both audit modes (online vs offline).

### Crucial population choice: dual-robot vs Panda-only
The historical two-robot formal population requires Panda and UR5e in a 20/50/150 each split. **Do not use invented UR5e acceleration or jerk limits**. If an independently sourced UR5e model and kinematic limits are actually available and fully validated, retain the historical two-robot split: Dev40=20 Panda+20 UR5e; Holdout100=50+50; Confirm300=150+150. Source-query/scene clustering by robot and original query must prevent leakage.

If UR5e remains blocked, explicitly preregister a **new Panda-only qualification protocol** before generating or inspecting any formal results: Panda-Dev40 / Panda-Holdout100 / Panda-Confirm300 (sample totals 40/100/300 but NOT the original mixed-robot Dev40/Holdout100/Confirm300). Label this as AMENDED_POPULATION_PANDA_ONLY in every output and report; do not claim multi-robot generality. Document reason, comparison limits, and new population freeze SHA. The original dual-robot protocol remains BLOCKED_UR5E_MODEL_LIMITS.

Both protocols use previously unseen source-query groups. Exclude source IDs, seeds, start/goal, scenes or descendent trajectories previously observed in the 79/145 exploratory study. Decide sample unit in advance (source queries attempted vs eligible successful trajectories); **do not silently top up or replace failures** to reach desired success counts. If the registered original count refers to *eligible successful paths*, use predetermined oversubscribed source manifests and a deterministic first-K successful rule, reporting all attempted failures and the deterministic selection mechanism. Freeze raw manifest before any solve. Freeze all eligibility/exclusion and denominator rules.

### Primary/secondary experimental arms
- **Primary (original):** unmodified cuRobo TimeOpt default scale 0.95, against original physical limits, certified mathematical model, independent replay, conditional native-reported analytic bound, T1 audit as an expensive separate mode. This is the formal target, NOT margin 0.99 or 1.00.
- **Secondary (preregistered):** scale 0.99 reruns on the same frozen source-query starts/goals/scenes; record distinct output SHA, paired duration change and native feasibility. Its 29/29 exact mathematical success but 28/29 analytic native reported is exploratory; no automatic safety or success equivalence. Qualification status calculated independently for this arm.
- Margin 1.00 and the previously seen 0.90/0.975 cases are **not** part of primary acceptance; if desired, preregister separate stress studies after locking and never pool them into primary statistics.

### Fixed acceptance/stop criteria
- Soundness: **zero false CERTIFIED**, zero proof-verifier disagreements and zero invalid numerical enclosures or mismatched native/ISA claims across all formal cases. Any exception is a hard stop and must be investigated with the current holdout declared exposed/contaminated.
- Eligibility: all scheduled source requests accounted for, including native planning failures, invalid inputs and UNKNOWN; document source-level attempt rate and eligible-success rate separately.
- Coverage: target ≥99% of eligible trajectories with complete mathematical K3 + independent proof and separately ≥99% native-reported analytic certificates if claiming that contract; an UNKNOWN is NOT a success. Exhaustive offline audit must have 100% enumerated time domain *for each trajectory claimed audited*. Predeclare whether result is descriptive pass fraction or inferential lower confidence limit; report one-sided 95% exact binomial interval and cluster uncertainty. Do NOT pretend n=300 automatically proves ≥99% robustly, especially if eligible count smaller or correlated.
- Fresh added certification latency: p50 ≤1ms (aspiration, prior 2.04ms FAIL), p95 ≤3ms (prior 2.06ms PASS). Report both, and an overall latency label FAILED_PARTIAL_TARGET when median not met; do not move thresholds.
- T1 offline latency/time-law scope: record wall time per audited output including grid construction, GPU enumeration, synchronization, reported-value audit and verifier replay. Never combine its results with online 2ms latency as one service.
- Record physical-time derivative deployment as NOT_ESTABLISHED unless separate interpolated-command semantics and proof exist; passing cohorts alone cannot fill this gap.

**R1 PASS** requires source/binary/config hash-signed freeze manifest and split-leakage audit, immutable unused holdout/confirmation manifests, no changes to rule after data observed. If UR5e blocked and no alternative preregistered Panda protocol, STOP_FORMAL_POPULATION.

## E. R2 — Dev40 (or explicitly Panda-Dev40) only after R0/R1

Run complete registered source population in original deterministic order/registered seeded randomization (avoid GPU concurrency). Retain every planner attempt/failure; all successful native arrays immutable. Run:
- Mathematical exact-model K3 producer + independent verifier, original physical limits; detailed per-family/interior/endpoint/exact witness status.
- Native-reported analytic FP32 bound + producer + independent verifier; explicitly preserve UNKNOWN_TIGHTENED_WITNESS.
- T1 full exhaustive float32-time audit independently for each successfully produced output (offline; log every attempted bit-pattern, compiled binary and time-grid digest).
- Diagnostic S8193/S32768 only as sampled evidence, never proof.
- Frozen-output H0, U1/U2/U4/U6 and adaptive per preregistered comparison budget.
- Fresh native-only vs native+full certificate randomized AB/BA, 50 warmups and ≥10 repeats/query, 8-thread CPU with recorded affinity, sync, clocks/load, output SHA/nondeterminism. Also measure startup/cold runs separately. No unsafe pooled statuses from different source outputs.

On Dev alone troubleshoot environment/performance and correct *only by versioning the candidate*. Any soundness-related code fix requires R0 re-review and R1 renewed freeze; no peeking at untouched holdout. Lock one candidate and policies before Holdout100.

**R2 PASS** if R0 soundness persists, planned coverage/latency measured with honest failures and no unexplained certificate/numerical or source mismatch. If efficiency targets miss but soundness passes, do not mislabel as full deployment qualification; continuation to holdout may still be scientifically valuable with a predeclared partial-performance designation. Version any changes and preserve raw Dev failures.

## F. R3 — Holdout100 (or explicitly Panda-Holdout100)

Open only after R2 fixes are complete and the chosen candidate has a new recorded freeze. Holdout100 is never used for tuning policies/bounds. Apply identical commands/config/version. Stop on any correctness failure or mismatch; do not relabel a contaminated holdout as independent after fixing against it. If invalid, obtain a *new, previously unseen* holdout with a pre-frozen regenerated protocol, explain invalidation, do not recycle old results as confirmatory. Report exact counts, eligible denominators, one-sided CIs and bootstraps clustered by source query/scene, margin arm separation, fresh p50/p95/p99, and complete proof/T1 status.

**R3 PASS** only if zero correctness failures, complete audit provenance, prespecified coverage and validity gates. If a performance target fails, label it partial qualification and whether pre-registered plan permits R4 for research. Any physical-time execution claim remains blocked.

## G. R4 — Confirm300 and second hardware

On a previously untouched Confirm300 (or explicitly Panda-Confirm300), execute the exact locked candidate without optimization. Preserve all 300 source cases or preregistered eligible selection plus all attempted failures. Run the full proof producer/verifier, analytic native reported certifier and per-output T1 exhaustive audit using the **same compiled code and GPU**. Record the expensive offline T1 time separately. Repeat matched fresh latency. Replicate a frozen registered subset on A100 with its own compiled binaries and separately audited hardware/compiler assumptions, never assume RTX3090 bounds transfer automatically.

Primary qualification claims must be computed from locked confirmatory population; report source-level confidence interval and all violations/UNKNOWN. One-sided 95% exact bound for 300/300 independent successes is just above 99%, whereas the two-sided 95% lower bound is below 99%; correlations and fewer eligible successes reduce evidence. State which inferential definition was registered and never revise after results.

**R4 PASS** only if no correctness violation, source hashes/eligibility intact and predeclared coverage/latency/status gates met. In all cases preserve full outcome ledger and do not claim universal time-law proof or physical robot safety. No hidden GPU overlaps, no untracked external workloads during timing.

## H. Independent report, audit trail and research decision

Produce files under docs/reports/runs with distinct immutable manifest paths and SHA:
- Independent T1 review memo with proof-obligation matrix and detailed counterexamples/fixes or PASS, reviewer provenance and separate checker code/tests.
- Freeze declaration and machine-readable accepted policy/contract, source + binary SHA, compiler/GPU hashes, source-query manifests, uniqueness/leakage audit, and old-vs-amended population flag.
- Per-stage Dev/Holdout/Confirm tables: attempted/native succeeded/ineligible; mathematical K3 CERTIFIED/VIOLATED/UNKNOWN; analytic native-reported CERTIFIED/UNKNOWN; T1 audited-and-consistent/violated/failed-to-audit; time inversion max gap with per-path certified bounds, and zero false-certification checks.
- Exact witnesses, per-family constraint margins, initial Bernstein hull, subdivision depth/leaves, endpoint and positivity details. Genuine case studies shown in interactive offline HTML linked to original Proof Lab, with source SHA, scope labels, proof tree, native-samples-as-samples.
- Online fresh p50/p95/p99 and first-call and stage breakdown, 8-thread CPU overhead, paired AB/BA CIs; offline exact T1 audit p50/p95 wall time clearly separate. Explain run-to-run nondeterminism; do not average nonmatching outputs without flags.
- Separately preregistered 0.99 duration/feasibility results and an exploratory margin frontier analysis with paired source bootstrap CIs, no hindsight policy claims.
- Machine-readable JSON report and complete source data hashes, exact commands, hardware/settings, raw result checksums, no synthetic numerical measurements.
- docs/PROGRESS.md append method + outcome (do not rewrite historical evidence). Commit source fixes/tests and reports, push experimental branch; no PR, merge, force-push or main mutation.

Final verdict categories:
1. PAPER: GO/NO-GO for evidence-based RA-L/ICRA/T-RO contribution.
2. MODEL_K3 certification: QUALIFIED / PARTIAL / NO-GO.
3. Native reported analytic certificate: QUALIFIED (within assumptions) / PARTIAL / NO-GO.
4. Per-output exhaustive T1 float32-time auditing: AUDITED_CONDITIONAL / BLOCKED / FAILED, with its separate slow cost.
5. Arbitrary unseen trajectories, unsampled controller interpolation, physical-time derivative *of executed command*, dynamics/torque/collision, other GPU/ISA and real robot tracking: NOT_CLAIMED unless separately proven.

**Mandatory STOP:** review unavailable or unsound, unreliable interval enclosure or GPU coverage, 0.95 configuration not pinned, missing documented physical limits, broken split/source hashes, proof-verifier disagreement, unsafe CERTIFIED outcome, or any attempt to replace failed outcomes/retune after holdout exposure. Document stop and still commit/report valid findings.

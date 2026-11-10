# TimeOpt → adaptive certification: implementation and experiment plan

Status: PLANNED. No GPU latency or real-robot certification results have been collected by this planning deliverable. The HTML contains analytic illustrations only. Prepared 2026-11-10 for GPT-6 Luna.

## 1. Decision to answer

Can an unchanged TimeOpt solution be certified continuously by exact subdivision, and what additional complete-request latency does this cost? Measure coverage, rigorous violations, inconclusive outcomes, latency distributions and certificate size. Phase A never repairs, retimes, changes endpoints, changes planning margins, or optimizes new coefficients. A failed certificate is not evidence of infeasibility. Adaptive refinement changes the representation, not the motion.

Primary deliverables: reproducible JSONL measurements; immutable input/solution manifests; an offline interactive HTML report with all statistics and per-path evidence; a concise decision report. This repository initially contained no source or data. This document defines proposed modules and commands; they are implementation targets, not existing commands. The shipped report builder and demo are real, but are not a production certifier.

## 2. Verified upstream integration map

Inspection pin: NVlabs/curobo c301cb00a87012c2808b1a60649c4065c9b3c95a. Inspect and preserve its license/attribution before reuse. Use it as a pinned dependency; do not silently modify upstream. Confirm checkout and environment before implementing the adapter. Record a different SHA explicitly if the inspected version cannot be used; no automatic upgrade.

- `curobo/_src/motion/motion_trajectory_timeopt.py`: `TrajectoryTimeOpt`, CUDA FP32, graph-captured full timing call, sampled feasibility; internal effective geometry includes boundary handling.
- `curobo/_src/motion/motion_trajectory_timeopt_cfg.py`: 32 timing controls required; default 511 uniform intervals, 10 iterations, velocity/acceleration/torque planning scale 0.95 and jerk scale 0.95. These are inspected defaults, not measured recommendations.
- `curobo/_src/motion/motion_trajectory_timeopt_result.py`: `timing_controls` has shape [B,32]; outputs are borrowed. Call `result.clone()` when retaining across future solver calls; account for required copying in deployment latency.
- `curobo/_src/trajectory/timeopt/timeopt_basis.py`: clamped cubic basis with `breaks=np.linspace(0,1,count-2)` and three extra end knots on each side; CPU SciPy basis coefficients are converted to FP32 device polynomials. Compare exact intended spline and actual kernel interpretation.
- `curobo/_src/trajectory/timeopt/timeopt_solver.py`: `CudaTimeOptSolver`, fixed 32-control fused solver. Do not increase this count to certify.
- `curobo/tests/_src/trajectory/timeopt/reference_equations.py`: independent equation checks, not a rigorous interval oracle.

Adapter output must export timing controls, exact effective geometric spline (including start/goal boundary modifications and layout), knot vectors, polynomial convention, dtype, model/payload identity, original and effective scaled limits, native constraint locations and knot-side samples, boundary states, output status, and native duration. Reconstructing a spline by fitting sampled output is prohibited. If effective geometry cannot be extracted faithfully, write BLOCKED_ADAPTER with the missing item.

Source links:
- https://github.com/NVlabs/curobo/tree/c301cb00a87012c2808b1a60649c4065c9b3c95a
- https://arxiv.org/abs/2611.06913
- https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node18.html
- https://www.mdpi.com/1424-8220/22/5/1869
- https://arxiv.org/abs/2602.07507

## 3. Certification contract: approve mathematically before benchmarking

The primary certificate is for the exact mathematical trajectory represented by the serialized coefficients and declared interpretation. It is not a guarantee of physical tracking, model accuracy, or the controller's interpolation. Report the evaluation discrepancy between this interpretation and the native FP32 kernels. Never silently treat quantized, separately stored derivative polynomials as derivatives of a different spline.

Let x(s)=s_dot^2, with path q(s). On each common path/timing knot interval:

- v = q' sqrt(x)
- a = q'' x + 0.5 q' x'
- ell = q''' x + 1.5 q'' x' + 0.5 q' x''
- j = sqrt(x) ell

Primary K3 residuals, each required <= 0:

- speed positivity: -x (strict positivity in the moving interior must be established; handle stationary geometry separately)
- velocity: q'_i^2 x / vlim_i^2 - 1
- acceleration: a_i / alim_i - 1 and -a_i / alim_i - 1
- jerk: x ell_i^2 / jlim_i^2 - 1

For cubic q and x on a common span, residual degrees are at most 7, 4, and 9 respectively. Derivatives are with respect to global s: conversion to local u must include interval-length factors. A certificate over pieces includes appropriate one-sided knot values and checks continuity required by the contract. Bounded piecewise jerk does not imply jerk continuity. Jumps in lower derivatives cannot be ignored.

Convert each residual to Bernstein coefficients on its interval. All coefficient upper endpoints <= 0 establish that residual everywhere. For an inconclusive interval, midpoint de Casteljau subdivision produces child coefficients with no change to q or x. Use separate certificate leaves; leave the 32 optimized coefficients immutable. At a split, mathematical monotonic enclosure improvement is expected, but interval roundoff may obscure strict improvement.

Numerics: implement interval arithmetic with outward rounding through extraction, derivative operations, products, division by positive limits, and subdivision. A final `nextafter` alone is not a proof if earlier operations were unbounded. A CPU interval reference is the first accepted backend; GPU bounds need justified rounding/error propagation and adversarial parity. FP64 with an arbitrary epsilon is NUMERIC_BOUND_ONLY, not CERTIFIED. Keep validation tolerance distinct from a changed physical constraint. Certify original and planning-scaled limits as distinct contracts; the 0.95 planning margin may make physical-limit certification much easier.

Outcomes per family: CERTIFIED; VIOLATED (rigorous witness interval/point enclosure strictly above zero); UNKNOWN_DEPTH; UNKNOWN_NODES; UNKNOWN_TIME; UNSUPPORTED; NUMERICAL_ERROR; INVALID_INPUT. Dense sampled overshoot without an error bound is NUMERICAL_VIOLATION, not a rigorous witness. A positive upper hull bound alone is inconclusive. Overall status is CERTIFIED only if every requested family plus positivity/boundary/regularity requirements passes. Unsupported torque means PARTIAL_K3, never full certification. Preserve native solver failures in coverage accounting.

Interior zeros of x may invalidate finite traversal. Native stationary paths need their own zero-duration contract. Do not impose artificial endpoint speed floors. For zero-speed endpoints, verify finite integrability and endpoint interpretation separately. No uniform positive-coefficient duration bound may be applied to zero coefficients. Duration integration/estimation is outside the primary feasibility certificate; report native duration and independent quadrature diagnostic separately. Phase A expects unchanged duration.

Torque is not generally polynomial along q. Full K4 (add torque) and K5 (add torque rate) require interval dynamics or a polynomial approximation with rigorous remainder bounds (and derivative remainder for torque rate). Sampled RNEA/JVP checks remain diagnostics. Do not delay K3 evidence to pretend this is solved. Before K4, publish the exact enclosure equations and arithmetic assumptions, then test a one-joint analytic gravity case. Extending merely fitted torque samples is forbidden. K5 is later scope: TimeOpt does not optimize torque rate at the inspected API.

## 4. Algorithm and memory policy

Maintain leaf records `(path_id, family, joint, span_id, depth, s_left, s_right, coefficient_intervals, status)`. Cache path-only span data and common knot extraction matrices by full topology/model key. Report fresh-path preprocessing separately from warm same-path reuse. Whole residual polynomials must remain correlated: subdividing residual coefficients is usually tighter than independently bounding x, x', x'' and multiplying scalar ranges.

For adaptive breadth-first refinement: test all leaves; retire certified leaves; check selected rigorous witness points; split inconclusive leaves subject to a deterministic budget; stop with explicit reasons. Keep the last proof tree. On GPU use compacted or masked frontiers with bounded allocation and overflow detection. First implement a simple CPU version. Compare GPU uniform/compact-frontier implementations only after reference correctness.

Depth is relative to the common-knot partition. Max-node budget is the number of active and retained certificate records per path, not an ambiguous per-joint number; record both processed and peak-active counts. Budget exhaustion never changes limits. Runtime deadline checks occur at documented batch boundaries and may overshoot by one batch; report observed overshoot. Avoid a false hard-real-time claim.

## 5. Sequential work packages for GPT-6 Luna

Work on `exp/timeopt-adaptive-certification-20261110`. One work package per commit; do not create a PR, merge, or change upstream. Before each package read the preceding evidence and current repo status. Keep `docs/PROGRESS.md` updated with commit, command, actual output locations, passed gate, blockers and the next action. Do not implement the later optimizer or jerk objective during this campaign.

### P0 — environment and source contract

Create `pyproject.toml`, `src/curobo_certify/adapters/timeopt.py`, and `scripts/capture_environment.py`. Pin Python, PyTorch/CUDA, cuRobo, NumPy/SciPy and interval dependency versions once a compatible GPU environment is found. Record GPU/CPU/driver, clocks, power policy, CPU affinity, graph settings and compilation status. Check upstream tests and a real 1-path smoke. Export one immutable solution. GPU absence is a blocker for GPU evidence, not permission to invent timings.

Acceptance: exact upstream SHA recorded; output lifetime handled; effective geometry identified; re-evaluation matches native position, v/a/j at 1,001 random coordinates and both sides of knots within a declared dtype-based tolerance; any discrepancies explained. This tolerance verifies adapter consistency, not feasibility. No bulk campaign before this passes.

### P1 — reference polynomial arithmetic

Create `src/curobo_certify/bernstein.py`, `interval.py`, `contracts.py`; tests in `tests/test_bernstein.py` and `test_intervals.py`. Implement exact topology extraction, derivative scaling, Bernstein product using binomial-weight convolution, degree elevation only when needed, de Casteljau splitting and interval coefficient enclosure. Test reference dyadic/rational inputs against exact Fraction arithmetic; high precision is an independent diagnostic, not alone a proof.

Acceptance: identity before/after subdivisions for q,x and derivatives; exact polynomial enclosed; strictly safe, hidden violating, tangent, endpoint, repeated-knot, tiny-span, zero-curvature, large dynamic range, NaN/Inf and exhausted-budget cases classified correctly. Include g=[-1,0.2,-1] whose midpoint children have all negative coefficients, and tangency at a non-dyadic coordinate where finite certification can remain inconclusive. No false CERTIFIED outcomes in tests.

### P2 — K3 reference certifier

Create `residuals.py`, `certify_cpu.py`, `trace.py`, `tests/test_kinematic_certificate.py`, `scripts/certify_frozen.py`. Compile residuals at the union of timing/path knots; implement status semantics and family aggregation; attach coefficient enclosures and witness records. Use the same final certifier for every comparator. Preserve hashes of timing and geometry arrays before/after.

Acceptance: 200 analytic/property cases with exact or validated extrema where possible; native-vs-independent q/v/a/j diagnostic plots; seeded numerical cases tested; positivity/continuity/boundaries assessed; a tiny proof checker independently replays saved leaves. Full K3 certificate must never be inferred from dense samples alone.

### P3 — frozen corpus and benchmark harness

Create `scripts/freeze_corpus.py`, `benchmarks/run_certification.py`, `benchmarks/run_end_to_end.py`, and JSON configuration files. Defaults and sampling splits in sections 6–7 are the initial preregistration; changes must be dated before holdout execution. Freeze every TimeOpt output including failures. Build JSONL writers with atomic checkpoints; store large arrays separately with SHA256. Keep trace collection outside timed runs.

Acceptance: A/B reruns replay identical frozen inputs; result-clone/export time is separately identified; no plotting/disk IO in timers; timed benchmark agrees with an external wall-time spot check; empty/failed paths retained. The harness must refuse mixed hardware/contracts and incompatible hashes.

### P4 — Dev40 and cost attribution

Run all reference arms on Dev40. Profile extraction, residual construction, bounds, splitting, queue/compaction, transfers and decision synchronization. Produce a first measured report even if latency is poor. Choose one adaptive policy and one fixed policy from Dev40 only. Avoid optimizing kernels whose measured contribution is negligible.

Acceptance: report complete-request and isolated frozen-output costs, numerical status distribution, per-family costs and real failure examples. Show K3 and full requested-contract status independently. No claim about GPU certifier performance from CPU reference timings.

### P5 — GPU acceleration only after reference gate

Create `certify_gpu.py` and narrowly scoped CUDA/Triton kernels. Reuse fixed extraction matrices. Implement a bounded frontier and a fixed uniform-refinement arm; document proof arithmetic. Label any faster non-rigorous screen as such. A screen may dispatch to the rigorous checker but may not authorize CERTIFIED by itself.

Acceptance: all accepted proof leaves checked by the CPU proof checker on pilot; random/adversarial disagreements investigated; no input mutation; measure request latency inclusive of mandatory CPU proof checking if it is still required at runtime. Separate optional offline audit from runtime authorization honestly.

### P6 — Holdout100, Confirm300 and second GPU

Lock code/configs before Holdout100. Run frozen chosen variants; evaluate thresholds below. If correctness fails, stop, diagnose, version the correction, and obtain a new untouched holdout or label reused evidence. If promising, run Confirm300 and one second GPU without tuning. Record cold process startup and first request separately. Include all failure/unknown cases and their costs. No requirement to achieve a positive result.

Acceptance: reproducible summary with paired bootstrap CIs, denominator ledger, tails, per-path traces and actual GPU metadata. Complete-request p99 uses enough requests and is labeled imprecise when based on only 100 distinct paths.

### P7 — interactive report and decision

Extend the supplied offline report. Implement measured path selector, robot/contract/family/method/status filters, paired baseline overlay, synchronized s/time cursor, speed and joint-constraint plots, native sample markers, knot markers, rigorous envelope bands, one-sided knot values, adaptive-tree depth replay, worst-interval zoom, and SHA/config badges. Clicking a tail/outlier row opens that exact path. Include a timing waterfall (stages measured separately), ECDF, certificate-coverage vs budget curve and log-scaled overhead-vs-leaf-count plot. Separate signed quantities from squared normalized residuals.

Acceptance: open without network, no console errors, all filters tested, empty data shown as NOT RUN, all aggregate points trace to raw records, analytical demos visually distinct from measured data, keyboard usable, desktop/mobile readable. Plot sampling is explicitly visualization only. Never call browser-side demo arithmetic a formal certificate. Export filtered CSV/JSON. Ship compact per-path data and link larger archives by checksums.

### P8 — optional escalation, gated by results

First diagnose dominant UNKNOWN and VIOLATED cases. If unchanged-output certification meets coverage/overhead gates, conclude baseline success; no certificate-constrained optimizer is justified yet. If real violations are common, specify a separate repair campaign. If torque is the blocker, implement rigorous K4 enclosures as a separate work package, preserve K3 conclusions, and rerun matched comparisons. Add optimization-in-the-loop only in a later explicitly scoped study.

## 6. Cohorts and variant matrix

Availability gate: inspect accessible assets, record provenance and redistribution rights. Do not copy private AORRTC/VAMP inputs from old projects into this public repository. Prefer public robot configs/benchmark sources; if unavailable, generate and label a new distribution. Choose robot models actually supported by the pinned revision.

Initial plan: Panda and UR5, each 220 independent paths = 440 total. Split per robot into Dev20, Holdout50, Confirm150 (pooled Dev40/Holdout100/Confirm300). Split by source scene/query before generating related variants; related seeds/geometries stay in one split. Use fixed seed 20261110 and save explicit IDs. Stratify easy/medium/cluttered geometries and path-span counts; freeze inclusion/exclusion criteria. Payload/endpoint variations remain paired descendants, not independent statistical samples. Stationary/degenerate cases go in a separately reported correctness cohort.

Primary population: unmodified TimeOpt defaults, original physical limits for certification. Secondary contract: same outputs, planning-scaled limits. This avoids attributing easy certification from default 5% margin to method superiority. Exploratory margin arm: rerun TimeOpt with scales 1.00 only as a separately frozen solver condition; do not combine its latency/coverage with default outputs. Tight jerk and synthetic hidden peaks are labeled stress cohorts and never replace the natural population. Keep native non-success queries in denominator ledger.

Arms on identical frozen outputs:

- N0: native TimeOpt only (sampled acceptance; no continuous certificate).
- D1: dense 8,193 sample diagnostic; D2: event-aware 32,768 sample diagnostic. Samples are not certificates.
- H0: common-span Bernstein hull without subdivision.
- U1/U2/U4/U6: uniform binary subdivisions through depths 1/2/4/6; evaluate complete partition. Cap allocation and record exhaustion.
- A: adaptive midpoint breadth-first, split only inconclusive leaves, retire certified leaves, optional rigorous witness evaluation.
- A-no-witness: ablation on Dev40 to identify witness overhead and violation-discovery cost.
- A-priority: optional Dev40 CPU worst-bound-first queue; only port if its measured savings justify implementation complexity.

Adaptive depth sweep on Dev40: 0/2/4/8/12/16; record leaf budgets 2,048/8,192/32,768 and certification wall budgets 0.25/0.5/1/2/5 ms as feasibility targets, not promised performance. First map deterministic depth/node budgets; runtime deadlines are a separate experiment because machine load changes termination. Avoid full Cartesian sweep: default nodes 8,192 and no wall deadline for depth sweep; choose two depths, then vary nodes; run deadline curves for the selected policy. Keep proof and trace work budgets explicit.

## 7. Timing and statistics protocol

Primary metric: added complete-request wall latency on a fresh path, batch size one, from device-ready geometric input to an available certification decision. Report other application input locations separately. Include path-specific extraction/preparation, residual construction, transfers, synchronization and any required output retention. Compilation, common topology setup and graph capture are startup costs, measured and published separately. No per-path work may disappear into a warm cache in the primary measurement.

Two complementary measurements:

1. Frozen-output certification: input is an immutable saved native solution. Measure certification stages and pure subdivision overhead relative to H0 on the same solution. CPU and GPU input location/transfer costs are explicit. This isolates the question 'what do subdivisions cost?'.
2. End-to-end paired requests: native-only versus native+certificate, alternating randomized AB/BA order within path/repeat. Check native outputs match frozen hashes or declared numeric identity checks. Differences must be investigated, not averaged away. Wall timing begins before native work and ends after the certificate decision is synchronized.

Use perf_counter_ns with synchronization at request boundaries for wall time. CUDA events measure device work separately; GPU events alone omit CPU, transfer orchestration and decision overhead. Instrumented stage timings run separately so synchronization changes do not contaminate the primary fast path. Summed stage medians need not equal median total latency; do not label them an exact decomposition. No traces, dense audit, disk writes or HTML building in timing unless intentionally part of the deployed request.

Hardware preference: RTX 3090 as primary and A100 as second GPU if available and compatible; otherwise declare a substitute in P0 before freezing the campaign. Never compare one method on one GPU against another method on a different GPU.

Warmup: 50 untimed requests per implementation/topology after compilation, then 10 paired repetitions per path; randomize path order and policy order deterministically. Report first-call/startup separately. Reuse fixed native outputs across methods. The primary fresh-path experiment must rebuild path-dependent data each repetition. Secondary cache-reuse and throughput batches 8/32 are separately labeled. Record CPU affinity, GPU clocks/power, temperature and load; stop concurrent GPU jobs where authorized. Do not tune clocks on one arm only.

Statistics: per-path median over repetitions, then population p50/p90/p95/p99/max; empirical request distribution separately. Paired differences computed per matching path/repetition then summarized; never subtract unrelated percentile summaries. Bootstrap 10,000 resamples of source query/scene clusters (seed 20261110), keeping variants/replicates together; 95% intervals on median overhead and certification fraction. If scene count is low, report it and interpret intervals cautiously. Also give absolute millisecond overhead and ratio to matched native wall latency. Ratios are unstable for tiny baselines: show both.

Denominators: all source queries; native successes; valid adapter outputs; K3 eligible; each certificate status; full requested-contract coverage. Show aggregate latency for all attempted outputs and separately for CERTIFIED/VIOLATED/UNKNOWN. Native failures and timeouts remain visible. Never use certificate-success-only latency as the headline.

Per-family outputs: max rigorous residual upper bound, worst leaf/joint/s interval, witness, depth histogram, processed/split/active counts, rounds, memory and termination reason. Compare counts with interpolation complexity and native utilization. Export p95/p99 outliers and all disagreements. Dense audit at both grids plus random probes is a discrepancy detector; any contradiction to CERTIFIED blocks the campaign until resolved, but lack of contradiction is not a proof.

## 8. Preregistered decisions (engineering targets, not results)

Hard correctness gate: zero observed false certificates, no input change, full provenance, replayable interval proofs and no unsupported family hidden in an overall certificate. Finite testing does not replace the arithmetic justification.

Initial usefulness target for K3 on physical limits: >=99% of eligible native-success outputs certified within default policy budget, added fresh-path p50 <=1 ms and p95 <=3 ms on the declared primary GPU. These are adjustable engineering targets only before holdout; publish actual values even if they fail. Report planning-scaled coverage separately. Achieving this is a K3 result, not a full torque certificate.

- PASS_BASELINE: correctness and usefulness targets met on holdout/confirmation. Recommendation: retain post-certification.
- CORRECT_BUT_SLOW: proof valid, latency target missed. Profile and optimize proven bottleneck; no new timing optimizer yet.
- INCONCLUSIVE_DOMINATED: bounds/budgets or endpoint handling dominate. Improve enclosure/refinement first.
- VIOLATION_DOMINATED: rigorous witnesses show native outputs really violate the chosen contract. Quantify prevalence and repair scope; propose a follow-up.
- BLOCKED_DYNAMICS: K3 supported, full torque/rate certificate missing. Explicitly partial; never aggregate as full success.

Boundary reasons are recorded independently: a modest number of violations or unknowns may still yield useful data, even when no pass label applies.

## 9. Report contract

Ship `reports/timeopt_certification_campaign.html` as a self-contained HTML. Current version includes the plan, task handoff, an analytic subdivision demo and an empty measured-data dashboard with JSON import. Build with `python tools/build_report.py`. View locally by opening the file. Its demo uses JavaScript floating point and is educational only.

Measured summary input is defined in `schemas/report.schema.json`: metadata plus per-request records. This compact reader initially provides method/contract filters, pooled per-path medians, paired baseline overhead, status counts, ECDF and CSV export. Production statistical CIs, source-cluster analysis, real path traces and linked explorers remain P7 tasks; do not claim the initial dashboard already implements them. Full trace format is specified in section 10.

## 10. Required persisted records

`environment.json`: git SHAs, model/config/input-manifest hashes, runtime versions, GPU/CPU identity, compile/capture modes and costs, seed, host-affinity and clock policy.

`paths.jsonl`: path/source/scene/split/robot IDs, parent variant ID, exact q/x representation metadata, native statuses, limits original/scaled, endpoint contract, stationary flag, all content SHA256s, provenance and storage URI. A path manifest entry must identify actual immutable arrays, not just a seed.

`requests.jsonl`: run/path/source ID, method/backend/precision, phase, proof level, family scope, certification contract, cache/input-location mode, repeat/order/block ID, native/cert/total milliseconds, stage timings where instrumented, status/reason, depth/node counts, memory, duration before/after, input hashes before/after.

`traces/<path_id>/<method>.json`: native sample and knot locations; plotting samples of s,t,q,x,v,a,j,tau,taudot with units and masks; leaf binary-tree parent IDs, intervals, coefficient enclosures, family/joint, status, witness, depth and round; proof hash. Do not interpolate across a derivative discontinuity without marking both sides. Plot curves are sampled diagnostics; envelope bands come from certificate records. Unsupported fields must be null with a reason.

`summary.json`: schema version, source-query and scene counts, eligibility/status denominators, hardware/contract-stratified statistics, bootstrap method/seed, outlier IDs, coverage-by-budget and report build SHA. Avoid Infinity/NaN JSON; use null plus reason.

## 11. Initial tests and commands

Available now: `python tools/build_report.py` builds the planning report; `node --test tests/*.test.cjs` checks demonstration mathematics and imported-data aggregation. These do not test a production certifier.

Commands Luna must implement (not available yet):

- `python scripts/capture_environment.py --out runs/environment.json`
- `python scripts/freeze_corpus.py --config configs/dev.json --out runs/frozen_dev`
- `python scripts/certify_frozen.py --manifest runs/frozen_dev/manifest.json --method adaptive --contract physical --out runs/dev_cert`
- `python benchmarks/run_certification.py --config configs/dev.json --out runs/dev_bench`
- `python benchmarks/run_end_to_end.py --config configs/holdout.json --out runs/holdout_bench`
- `python tools/build_report.py --results runs/holdout_bench/report.json`

Exact CLI flags may change during implementation; update the plan, README and prompt together so no documented command silently becomes stale.

# Margin, duration, certifiability, time law, and bound latency campaign

Prepared 2026-11-17. STATUS: IMPLEMENTATION HANDOFF; ALL NEW CODE REQUIRES FULL TEST AND GPU VERIFICATION.

**Execution status (2026-11-17, see docs/PROGRESS.md):** G0 PASS after fixes (commit ca68a16); G1 PASS (180/180 requests, 145 native successes, constructed config audited); G2 PASS (zero unsound certificates; 3,045 benchmark rows); G3 delivered (`runs/margins/analysis.json`, `reports/margin_certifiability_results.html`, research report tabs); T1 PASS for a scoped per-output exhaustive float32 audit contract (docs/NATIVE_TIME_LAW_20261117.md), not a universal proof; L1: fresh added p50 12.32 → 2.04 ms, p95 2.06 ms (≤3 ms met, ≤1 ms p50 not met); G4 (Dev40/Holdout100/Confirm300) NOT RUN. Base: 5c7c0fcf2c52d35cbeece32dbcff0e80f30df661. Work branch: exp/margin-duration-native-time-20261117. No PR, merge, forced update, historical output rewrite, or main change.

## Goals and precise scientific contracts

H1: Relaxing TimeOpt's internal limit_scale and jerk_limit_scale from the original 0.95 toward 1.00 may improve duration on identical source queries, at the cost of additional genuine continuous-time constraint violations or failure to certify. No assumed monotonic trend or a priori win.

H2: Certified adaptive Bernstein subdivision may preserve feasibility guarantees at lower cost than uniform subdivision near active limits, and give mathematically exact violation witnesses.

H3: Precomputing reusable native-time arithmetic bounds and optimizing compiled loops may substantially reduce the reported ~9.7 ms native-bound stage, while preserving rigorous inequality enclosures. Previous full fresh added p50 ~13.1 ms, not ≤1 ms target.

H4: It is possible to prove a useful bounded native time-progress error and physical-time derivative relationship for a precisely specified command semantics. This is NOT established, and exact equality of discrete FP32 inversion with a smooth inverse at every real time is not a suitable requirement.

Always distinguish:
A. EXACT_MODEL_K3: continuous-time inequalities of an exact mathematical spline representation.
B. NATIVE_REPORTED_K3: source/ISA-assumption-conditional inequalities of native sample kernel's reported v/a/j for every admitted progress s.
C. NATIVE_PHYSICAL_TIME_K3: actual derivatives of a clearly defined continuous-time commanded q(t). Currently BLOCKED_NATIVE_TIME_INVERSION. Neither A nor B alone implies C.
Torque, torque-rate, collision and physical robot tracking are outside this K3 paper's guarantees.

Previous evidence: 79 exploratory real trajectories. 48 default 0.95 successes against physical mathematical limits, 1 needed subdivision. Among 31 separate 1.00-margin outputs, 19 exact-model certificates and 12 genuine mathematical witnesses; worst observed jerk ~1.0036× physical limit. Compiled producer+independent verifier p50 ~3.3 ms frozen; native arithmetic bound ~9.7 ms; fresh added p50 ~13.1 ms. Previous experiments NOT recreated by this branch. Formal Dev40/Holdout100/Confirm300 NOT RUN.

## Implemented now, pending independent tests

- src/curobo_certify/margin_campaign.py: canonical 3-decimal margins; SHA-verified matched query designs.
- scripts/freeze_margin_queries.py: default 0.90/0.95/0.975/0.99/1.00 × (4 scenes ×3 magnitudes ×3 source queries) = 180 planned attempts, 36 pair IDs, no actual GPU jobs.
- scripts/generate_hardcase_population.py: runtime-only scale overrides; preserves 0.95 pinned default and hashes source query prior to override. Audit effective native solver configuration!
- src/curobo_certify/margin_analysis.py and scripts/analyze_margin_campaign.py: all submitted, successful, failed and NOT_RUN denominators; source SHA joins; paired duration bootstrap; separate model/native/time-law status.
- scripts/bench_certification_methods.py: source SHA-backed planning scales (old default 0.95 when no manifest). Audit for nondefault margin correctness.
- src/curobo_certify/time_law_enclosures.py: Fraction-based rational bounds for 1/sqrt(x) and integral/inverse of exact serialized positive-x models, with zero-speed endpoints explicitly UNKNOWN. NOT native time inversion proof.
- src/curobo_certify/compiled_k3.py: pinned float32 timing table cached as read-only array. Benefit unmeasured.
- scripts/bench_native_bound_pieces.py: frozen path box-budget/division error sensitivity; full producer + independent verifier and raw repeated cost.
- web/margin_dashboard_template.html, scripts/build_margin_report.py, reports/margin_certifiability_dashboard.html: offline HTML viewer with import; EMPTY data by default. No synthetic results.
- tests/test_margin_campaign.py, tests/test_margin_benchmark.py, tests/test_time_law_enclosures.py, tests/test_margin_dashboard.py: regression starting point, unverified in full project runtime.

## Gate G0 — reconcile repository, run and fix ALL tests

Preserve local git branches, ignored runs, source NPZs and user artifacts. Identify pinned cuRobo SHA c301cb00a87012c2808b1a60649c4065c9b3c95a, compiler, CUDA flags, GPUs, CPU affinity and source hashes. Run:

  python -m unittest discover -s tests -p 'test_*.py' -q
  python -m compileall -q src scripts tests
  node --test tests/*.test.cjs
  python scripts/build_proof_playground.py
  python scripts/build_margin_report.py
  python scripts/freeze_margin_queries.py --seed 20261117 --per-cell 3 --out runs/margins/queries.json

Open offline margin HTML in Firefox and Chromium, desktop/mobile. Verify import/CSV/scatter filtering, empty state, data integrity and console. Verify HTML against committed template, JSON script injection escaping, and independently validate generated source hash sidecar. Prior inherited GitHub workflow tests Python 3.13 but project pyproject declares Python <3.13; fix CI to 3.11/3.12 and include the new branch.

Specifically audit source-runner summary: old hardcase generator reports only legacy 0.95/1.00 margins. Fix to include all five registered conditions, preserving failures. Run tests proving actual constructed native config differs only in limit_scale and jerk_limit_scale, and 0.95 is original constructor without monkeypatch. Watch CUDA graph/compiled config reuse and state leakage. Check output-file overwrite and the exact scale used by the benchmark. Fix discovered bugs, write regressions, commit.

G0 PASS: all tests and browser smoke pass, exact source hashes and 0.95 baseline verified, full 180-query frozen manifest unchanged, no source data loss.

## Gate G1 — registered matched native TimeOpt pilot

Use frozen source manifest and source-runner on RTX3090:
  python scripts/generate_hardcase_population.py --queries runs/margins/queries.json --out-dir runs/margins/native

Check solver scales inside constructed TimeOpt instance, not only a wrapper argument. Process arm order randomized or process-isolated; log order and environment, optionally separate repeatability runs. Retain all 180 request outcomes, including explicit FAILED_RETAINED and any NOT_RUN. Keep original frozen geometry, timing controls, physical limits and durations unchanged, with SHA256. Do not select easier paths or successful restarts post hoc. Preserve query-16 nondeterminism or similar issues as a separate report.

G1 PASS: complete 180-row denominator, source pairing and actual config proven, no implicit limit changes.

## Gate G2 — physical/planning/native-reported certification

On each frozen successful output, run compiled full K3 producer AND independent verifier against:
(1) original physical limits for exact mathematical model;
(2) planning limits with each record's source-hash-backed actual planning scale;
(3) original physical limits tightened by the sound native FP32 arithmetic error enclosure (native reported values).
Use existing scripts/bench_certification_methods.py with --margin-queries and --margin-records-dir for new scales; test 0.975/0.99/1.00 explicitly, preventing old 0.95 hardcoding. Run H0, U1/U2/U4/U6, adaptive and no-witness methods on identical NPZs. Exact-model positive witnesses against physical limits are mathematical violations; witnesses only against tightened limits are UNKNOWN_TIGHTENED_WITNESS, NEVER native-limit violations. Reject incomplete proof, verifier disagreements, nonfinite enclosures and unknown arithmetic assumptions.

G2 PASS: exact input hashes unchanged, all families/knot/endpoints verified, zero unsound certificates, full denominators and separate claim status.

## Gate G3 — stats, offline report, publication comparisons

For each margin present full source denominator, native success/failure rate, exact-model certified/violated/unknown, native-reported certified/unknown, time-law still BLOCKED, duration distributions, leaf count, split depth, witness location and max overshoot, complete certificate/verification/bound latency.

Matched duration gain for source pair p where both arms succeeded:
  G(p,m) = 100 * (T(p,0.95)-T(p,m))/T(p,0.95).
Report pair-cluster bootstrap median + 95% CI (>=4000 resamples), scene/magnitude strata, missing pairs: only baseline succeeded, only candidate succeeded, neither succeeded. No hindsight-best-of margins as real online policy. Explicitly analyze selection bias due to failures.

Commands:
  python scripts/analyze_margin_campaign.py --queries runs/margins/queries.json --records-dir runs/margins/native --certification-benchmark runs/margins/cert_bench.json --verify-arrays --out runs/margins/analysis.json
  python scripts/build_margin_report.py --analysis runs/margins/analysis.json --out runs/margins/report.html

Build a richer real-path HTML case explorer linked to original Proof Lab: original physical/planning limits, exact residual and peak brackets, hull conservatism, native sampled diagnostics, violation witnesses, proof trees, per-stage latency; all scope labels explicit.

G3 PASS: each number generated from raw source rows, no invented results, complete QA of downloadable report.

## Gate T1 — solve the native time-law problem, not merely same-progress error

Existing native-reported conditional proof already bounds the reported v/a/j at any admitted float32 progress. To certify derivatives of an actually defined time-parametrized trajectory, analyze native 511-cell time table construction, time query domain, lookup, Newton iteration, floating-point rounding/FMA, clamping and knot-side behavior, and interpolation between requested samples if any.

First extend exact-model rational integration to strict-rest endpoint behavior. Use root multiplicity and certified substitutions (e.g. x~c(s-a) at endpoint -> s=a+u^2) for finite traversal time. Detect higher-order nonintegrable zeros and internal zeros; never clip x away from zero to hide failure. Develop rigorous time-prefix and inverse-progress interval enclosures with independent replay. Cross-check against numerical 32/64-cell quadrature as DIAGNOSTIC ONLY.

Second bound native s_native(t) vs exact mathematical inverse s*(t) for every admitted time, using actual native arithmetic. Do not demand impossible bitwise equality of quantized FP32 s with smooth s*. Audit PTX/SASS instructions and documented error assumptions. Distinguish representable queried times, continuous command interpolation semantics, finite differences and actual derivatives. If physical-time derivative implication is not proven, leave BLOCKED_NATIVE_TIME_INVERSION and prohibit deployment/qualification claims.

T1 PASS: precise mathematical execution contract and independently checkable all-domain time-law error bounds, otherwise BLOCKED.

## Gate L1 — native-bound profiling/optimization

Reported old native-bound step ~9.7 ms, producer ~2.2, verifier ~1.3; total added p50 ~13.1 ms. Profile cached pinned timing table before/after, ctypes and NPZ packing, shared timing x vs repeated per-joint work, C++ interval boxes and loop overhead. Begin with:

  python scripts/bench_native_bound_pieces.py --npz <same frozen paths> --pieces 4 8 16 32 --div-ulps 2 4 --sqrt-ulps 4 --warmups 50 --repeats 20 --out runs/margins/native_bounds.json

Compare only fully verified producer+verifier on identical frozen original arrays. Use paired randomized A/B, 50 warmups, >=10 repeats, p50/p95/p99 and source-cluster bootstrap. At each change compare bound enclosure, statuses, leaves and original input hash. Optimize CPU shared timing coefficients and precomputation before porting to GPU, unless profiling justifies GPU. Finite random falsification never proves soundness. STOP on underenclosure, invalid proof, changed model, evidence corruption or missing soundness assumptions. Targets <=1ms p50 and <=3ms p95 added remain aspirational, not observed.

## Gate G4 — final qualifying cohorts ONLY after T1 correctness

Once native physical-time correctness contract and binary/ISA assumptions are independently defended, freeze source/compiler, production code, certificates and thresholds. Run untouched Dev40, then Holdout100, Confirm300. RTX3090 baseline, A100 replica after freeze. If correctness gate remains blocked, do not run or claim qualified cohorts; retain useful exploratory paper results.

## Deliverables and decisions

Commit source, tests, raw artifact summaries and full docs/PROGRESS.md with exact commands, dates, source SHA, GPU, seeds, denominators, CI and errors. Make offline interactive HTML report with direct real-path explorer. Give comparison table for each margin (attempted/solved/model-certified/model-violated/native-reported-certified/time-derivative-certified/unknown), path durations, paired gain CIs, latency breakdown, proof complexity and verifier outcome. Preserve failures/nondeterministic cases. Publication GO can be justified for verified exploratory math plus numerical scope; physical-time deployment NO-GO until native inversion is soundly linked.

No PR, merge, force push, main changes, output mutations, or misrepresentation of not-run experiments.

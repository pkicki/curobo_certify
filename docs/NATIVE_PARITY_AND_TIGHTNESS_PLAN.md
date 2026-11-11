# Device-side jerk parity and certificate tightness: 2026-11-11

**Status: independent recovery implementation; NOT a native GPU experiment.** The user's latest local integration is exp/timeopt-cert-integration-20261110 at 018fa07c935fa40be52890d929619aa520ed5b3b. That commit is absent from the remote. The independent branch is based on published f40d82a. Preserve the local unpublished branch and cherry-pick these commits onto a *new child of the local branch*, resolving conflicts; never reset or replace it.

## Science questions and evidence scope

1. Find the source of native CUDA FP32 jerk differences, without changing the registered scaled-relative parity threshold of **4e-4**.
2. Measure exploratory frozen-output CPU certification costs on the **eight existing real Panda serialized models**, while retaining the distinction from native-linked certification and official Dev40 complete-request latency.
3. Explain how constraint margin and Bernstein overestimation predict subdivision cost.
4. Pursue UR5e model validation independently, without blocking Panda-only exploratory studies.

The latest *reported*, not independently reproduced, pilot had 16 queries: eight Panda native successes, eight unavailable UR5e models. All eight Panda serialized-model K3 proofs passed; zero passed native jerk parity. Worst reported example: Panda query06 joint0 at t=0.113864064s, s=0.379210323, native jerk -96.82724, adapter FP64 -96.822096, manual FP32 replay -96.825195. Maximum residual among FP32 replay points reached 0.00452. Torque/rate remain unsupported.

## Gate D0 — Safe environment and native tracing

Record local commit/status, native upstream revision, build flags, driver/GPU, CUDA graph mode and input hashes. Preserve cuRobo c301cb00a87012c2808b1a60649c4065c9b3c95a unchanged. Build a *separate disposable pinned checkout*.

The new make_timeopt_cuda_trace_patch.py requires exact source HEAD and a Git blob checksum (714d99c58d2e0f083a8c9fafdc68bef0b0a63f8f) for the CUDA sample kernel. It generates a unified diff without touching upstream. Apply that patch **only** inside the disposable clone, rebuild CUDA code, select a flattened native sample index and joint, and parse the emitted device-side float32 operand bits with parse_timeopt_cuda_trace.py.

For each selected point (begin with query06 joint0), log native requested time, internal path progress s, segment/local coordinates, q' q'' q''', x x' x'', sqrt(x), native jerk and reconstructed diagnostic subterms. Each record uses exact float32 bit patterns. Compare pristine and instrumented native outputs bitwise for the same inputs: printf and extra arithmetic can affect compiler codegen. The independently recomputed diagnostic subterms are **not** evidence of original FMA grouping. Collect at least 20 points and repeat across eight Panda paths.

Diagnose in two branches: (A) *same native s* in both implementations (disables time inversion), (B) same physical t using native inversion. Compare FP32 bit/ULP and abs/scaled errors at each stage. Check centered geometry controls, B-spline derivative evaluation, FP32 coefficient construction, Horner order, FMA, knot-side selection and CUDA graph replay. Do not change original output, timing controls, duration, endpoints, limits or 4e-4 parity tolerance. If native errors cannot be bounded rigorously, leave BLOCKED_NATIVE_LINKAGE.

## Gate D1 — Frozen-model CPU exploration (does not depend on D0)

Use ignored local runs/native_pilot_resolved/ real Panda evidence. Export each exact common-span spline as JSON schema_version=1, basis=local_power_ascending_u, spans of local-power cubic q and x, exact target endpoints and original physical velocity/acceleration/jerk limits. No fitting or conversion by sample interpolation. Link native source hashes to the derived export SHA256 in a provenance manifest.

Run for every model:

    python scripts/explore_k3_models.py MODEL.json --out /tmp/model_tightness.json --tolerance 1/1000000
    python scripts/bench_frozen_model_cpu.py MODEL.json --warmups 50 --repeats 10 --out /tmp/model_cpu.json
    python scripts/render_tightness_html.py /tmp/model_tightness.json --out /tmp/tightness.html

The exact Fraction tightness analyser bounds each polynomial maximum by rational L*<=max r<=U*, initial Bernstein hull upper U0, residual margin interval [-U*,-L*], overestimation interval [max(0,U0-U*), max(0,U0-L*)]. Even budget-UNKNOWN outputs have valid mathematical brackets. The HTML is offline; its plots use approximate doubles while underlying JSON retains rational strings. These are mathematical-model findings, not native FP32 certificates.

The CPU harness measures frozen-model hook execution, including parsing and replay for default exact_reference. It excludes native planning/TimeOpt, GPU transfer/synchronization, native parity, and complete-request latency. Use --hook name=module:function to integrate the *local outward-rounded CPU certifier and independent proof replay*, with parsing/extraction inside the timed hook. Do not mistake Fraction performance for the deployment certifier. Preserve raw repeats, statuses, path IDs and source SHA; run paired balanced AB/BA repeats and record CPU affinity. No campaign p50/p95/p99 is authorized.

## Gate D2 — Probe real adaptive refinement

The reported model pilot had 47 spans * 7 joints = 329 span/joint records, with positivity, velocity, +/-acceleration and jerk checked within each record; no refinement. Audit per-family completeness and shared-helper replay limitations. Plot margin vs exact hull overestimation, proof depth and runtime; separately label sampled diagnostic utilization. Include exact analytic counterexamples where initial hull is inconclusive and subdivision proves safety.

Search for naturally harder real native TimeOpt outputs; keep the source-query ledger, failures and model-only/native-linked labels. If predeclared planning scales are varied before solving, freeze those outputs in a separate stress cohort. Never modify solution coefficients after TimeOpt.

## Gate D3 — UR5e, formal cohort and reporting

Pin official Universal Robots ROS2 Description source at 78c5afd8bfee78a459c0265304340a85f1d44792. Generate UR5e Xacro/URDF in isolated env; verify 6-DOF joint order, limits, collision geometry and FK against an independent parser. Acceleration and jerk limits must have defensible provenance; do not invent them or substitute UR10e. UR5e blockers do not prevent Panda-only *exploration*.

Only after native-linking correctness and verified multi-robot data gates pass: freeze Dev40 by source scene/query, then perform registered H0/U1/U2/U4/U6/adaptive comparisons. Complete-request latency on RTX3090 must include the native solve, cloning, extraction, residuals, transfers, proof replay and CPU-visible decision. Use 50 warmups, >=10 paired balanced AB/BA repeats and 10,000 source-cluster bootstrap replicates. Freeze method before Holdout100/Confirm300, report full denominators and original engineering targets (K3 >=99%, added fresh p50 <=1ms/p95 <=3ms). No torque/rate full certificate until proved.

## Deliverables and acceptance

- Pinned CUDA diff, instrumented-vs-pristine bit parity and stage-wise numerical attribution (or explicit unresolved blocker).
- Immutable native→serialized model manifests and per-family exact margin/hull/depth data.
- Actual outward-rounded CPU frozen-model exploratory timings separately from exact-rational oracle timing; no fake complete-request latency.
- Interactive offline plots with provenance and categorical evidence scope.
- UR5e source/FK/limit status and full query-denominator ledger.
- Passing Python, CUDA/native, Node/browser tests as applicable, clean working tree and logical commits; no PR/merge.

**Section 8: NOT REACHED** until formal native correctness, corpus and latency requirements pass.

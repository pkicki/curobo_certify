# Coding agent: implement and run the TimeOpt jerk-parity + CPU-cost campaign

You are on the user's GPU machine, continuing pkicki/curobo_certify. Latest user-reported **local** branch exp/timeopt-cert-integration-20261110 HEAD 018fa07c935fa40be52890d929619aa520ed5b3b. GitHub remote does NOT contain it; preserve the local commit and all artifacts. The independently published branch exp/timeopt-jerk-and-tightness-20261111 contains source/test/plan changes based on f40d82a, not the local integration branch.

**Do not create a PR, merge to main, reset, or force-push.** Start with git branch -avv, git status, git log --graph --all --oneline -25. Create a *new* child branch from the local 018fa07c head and cherry-pick the new recovery commits after comparing ancestry; resolve conflicts without losing any local implementation or evidence. Read AGENTS.md, docs/CAMPAIGN_PLAN.md, docs/RECOVERY_PLAN.md, docs/NATIVE_PARITY_AND_TIGHTNESS_PLAN.md, docs/PROGRESS.md and local J1–J7 reports.

## A. Instrument actual CUDA jerk evaluation — no threshold changes

- Preserve pinned cuRobo revision c301cb00a87012c2808b1a60649c4065c9b3c95a untouched.
- In a **disposable separate native checkout**, generate a kernel patch with scripts/make_timeopt_cuda_trace_patch.py. Validate the original source HEAD and Git blob. Apply only to the disposable checkout, rebuild CUDA extension, verify native trace prints device-side IEEE FP32 operands at the selected sample index/joint. Parse with scripts/parse_timeopt_cuda_trace.py.
- Capture pristine vs instrumented output bits for *identical* queries and sample times; a changed debug output invalidates inference about pristine FMA arithmetic. Also record the actual native time-table duration, q/x coefficient hashes, interpolation convention, CUDA flags and source archive. Start with Panda query06 joint0, t=0.113864064s; then at least 20 diverse points and all eight Panda paths.
- Compare native t→s, geometry segment/local parameter, centered q',q'',q''', x,x',x'', three separately computed jerk terms, final native jerk and original adapter. Run separate **same internal s** experiments to isolate s(t) inversion. Compare exact float32 bit patterns, ULP differences, absolute and scaled error; investigate FMA, derivative scaling, coefficient construction, Horner order and knots.
- The diagnostic recomputed terms may not match the native fused expression. Do not claim a rigorous native error bound from observed max differences. Retain registered scaled-relative tolerance 4e-4 unchanged. If it still fails, native-linked certificate status stays BLOCKED_NATIVE_LINKAGE; do not change mathematical spline, duration, 32 timing controls or original limits.

## B. Frozen-model analysis without waiting for CUDA parity

- Preserve all eight real Panda native solutions under ignored runs/native_pilot_resolved/ and their original hashes. Write a versioned, tested export bridge to JSON local_power_ascending_u from real effective polynomial coefficients (do **not** fit sampled outputs). Record source hashes, original robot limits, strict-rest and knot contracts. Keep private data out of public Git.
- Run scripts/explore_k3_models.py for per-family exact rational residual peak brackets, margins and Bernstein-hull overestimation. Build a single interactive offline report using scripts/render_tightness_html.py with all eight models and filters.
- Run scripts/bench_frozen_model_cpu.py with 50 warmups and 10 repeats/path, retaining raw ns and statuses. Its default exact_reference is only a slow reference oracle; implement an explicit hook for the *actual local outward-rounded CPU certifier*, including correct extraction/preprocessing and independent replay inside the timed hook. Document cache/fresh-path assumptions, CPU affinity and environment. No official Dev40 or end-to-end p50/p95/p99 claims from this exploration.
- Cross-check per-family residual formula and proof accounting separately from the shared-helper proof replay. For each span/joint report positivity, velocity, +/-acceleration and jerk; distinguish 329 span/joint records from individual family checks. Test hidden peaks, tangencies, zero-speed interior, negative polynomial speed and budget UNKNOWNs.

## C. Adaptive subdivision hypothesis

- Plot exact residual margin versus exact Bernstein-hull overestimation, initial hull-certification status, depth/nodes/latency and numeric utilization (separate sampled diagnostic).
- The eight existing real paths reportedly certify with zero subdivision. Search a *separately labeled* real native stress cohort to find naturally tighter residuals; if solver planning margin is varied, predeclare it before generating paths and label the resulting population distinctly. Never repair/modify solved coefficients.
- Compare no subdivision, fixed uniform and adaptive refinement on matching frozen outputs. Report first nontrivial genuine subdivision proof with per-leaf replay and hashes; if no real path needs subdivision, preserve that negative evidence.

## D. UR5e and eventual formal cohort

- Generate the genuine UR5e URDF from pinned official Universal Robots ROS2 Description source 78c5afd8bfee78a459c0265304340a85f1d44792. Validate independent FK and six-joint order, collision geometry, robot limits. Provide a defensible source for acceleration and jerk limits, or explicitly keep UR5e BLOCKED; do not substitute UR10e or invent physical data.
- Keep Panda exploratory analysis moving even if UR5e remains blocked.
- Only after the original native parity/error-bounding gate and validated two-robot corpus are complete, freeze Dev40, then run registered fresh-path and frozen-output benchmarks; lock before Holdout100/Confirm300. Follow original 50 warmups, paired AB/BA repeats, clustered uncertainty protocol, complete request timing including all required proof and device/host costs.

## E. Reporting and validation

Implement changes rather than stopping at a plan. Execute meaningful tests, inspect failures, fix bugs, retest, commit each package and update docs/PROGRESS.md with commands, source/config hashes, actual numerical results, missing environment dependencies and explicit next gate. Run Python regression suite, compileall, Node report tests, offline browser tests and pinned native smoke tests; source syntax tests for diagnostic patch and a real instrumented CUDA build when available. Preserve original/diagnostic CUDA builds and compare output bits.

Finish with: exact local branch + commits + clean status; native/patched GPU versions; per-stage jerk attribution and unchanged threshold result; eight Panda model-only and native-linked denominators; rigorous exact margin/overestimation distributions and depth; CPU exploratory timing only with raw repeats and matched hook scopes; UR5e FK/limits outcome; interactive HTML path; full provenance; Section 8 NOT REACHED unless formal gates genuinely pass. No fabricated or inferred timing, no PR, no merge.

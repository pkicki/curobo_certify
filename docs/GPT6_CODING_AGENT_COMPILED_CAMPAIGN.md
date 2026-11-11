# GPT-6 coding agent — compiled K3 hull and native jerk parity continuation

You are working on the user's GPU machine in pkicki/curobo_certify. IMPLEMENT, RUN TESTS/EXPERIMENTS, DEBUG, COMMIT and REPORT actual results. No PR, merge, reset or force-push.

## 0. Reconcile branches
Latest reported unpublished local: exp/timeopt-jerk-tightness-integration-20261111, HEAD 54e3905ec058f7f060eb3ea5b908e81e63f895e9. Published independent new branch: exp/timeopt-compiled-k3-hull-linkage-20261111, descended from e99c3d50ef914b174a12c570818023e498135883, not from local 54e3905.

Inspect git status, git branch -avv, git log --all --graph --oneline -30 and ignored runs/. Make a NEW child from local 54e3905 and cherry-pick new recovery commits with ancestry/conflict checks. Preserve unpublished code, NPZs and pinned native source. Read AGENTS.md, docs/CAMPAIGN_PLAN.md, docs/NATIVE_PARITY_AND_TIGHTNESS_PLAN.md, docs/COMPILED_HULL_AND_NATIVE_LINKAGE_PLAN_20261111.md, docs/PROGRESS.md.

## 1. Compiled mathematical hull correctness
Build the new C++ shared library:
    python scripts/build_compiled_hull.py --out /tmp/libk3hull.so --manifest /tmp/libk3build.json
Run the full Python suite and compileall. This is a PARTIAL initial Bernstein K3 hull checker, not full certification; never convert a returned negative hull into CERTIFIED_K3 without endpoint, knot, strict positivity/integrability, independently implemented arithmetic proof replay and native evaluation linkage.

The wrapper encloses exact rational input coefficients in finite binary64 endpoints. C++ outward arithmetic uses nextafter, no fast-math/FMA contraction. Test at least 10,000 seeded exact Fraction polynomial cases (all positivity/velocity/+/-acceleration/jerk coefficients), near-active cases, tiny h, non-dyadic rationals, cancellation, signed zero, overflow and violated families. Add independent MPFR comparison where available and rounding-mode/optimization checks. Compare the same padded Bernstein polynomial degrees: x:3, velocity:7, acceleration:4, jerk:9. Degree elevation changes hull looseness. Any under-enclosure blocks usage; fix and retest. Record binary/compiler/environment/input hashes.

## 2. Real eight-Panda CPU stage attribution
Existing pilot: eight real Panda serialized spline models K3-certified conditionally; 47 spans by 7 joints; no splits. Native-linked parity still FAILED. Outward Python + proof-replay p50 was about 1.34–1.62 SECONDS. Use verified existing runs/native_pilot_resolved/ NPZs and runs/pilot_exact_models/, not synthetic replacements.

Run scripts/bench_compiled_hull.py against the actual exact exported models, 50 warmups and >=10 repetitions per path; preserve raw timing and source SHA. Render interactive exploratory timing with scripts/render_compiled_profile_html.py. Implement MATCHED-SCOPE Python hull-only baseline plus separate COMPLETE outward+proof-replay baseline. In a complete compiled+reference pipeline, include boundary/knot/strict-positive/proof-replay work in timed totals. Compare initial-hull to initial-hull, full proof to full proof; never present the hull-only result as native-linked latency. Attribute JSON parse, exact interval packing, residual construction, Bernstein conversion, hull checks, proof capture/replay, serialization and decision. Preserve CPU affinity, compiler flags, cold/warm labels and all repetitions. Do not compare synthetic 47x7 timings with real full-proof seconds as equivalent.

## 3. Native jerk operands
Keep pinned cuRobo c301cb00a87012c2808b1a60649c4065c9b3c95a and disposable diagnostic checkout. Prior 1,176 device trace points match final libm.fmaf replay bitwise using captured operands, but adapter/native parity still fails unchanged scaled-relative tolerance 4e-4. Do not modify tolerance. Export exact matching native trace plus adapter at same captured native INTERNAL PROGRESS s with idx,joint,s,q1,q2,q3,x0,x1,x2,jerk and source SHA/time. Run scripts/attribute_native_jerk.py for float32 ULP and empirical geometric-vs-timing Shapley decomposition. Analyze time inversion separately. No causal claim when progress differs. Verify pristine/instrumented output bits identical; instrumentation-sensitive forced progress remains inconclusive. Attempt rigorously justified continuous error bound for native coefficient creation, spline evaluation, time inversion and jerk arithmetic, not an empirical maximum.

## 4. Subdivision exploration
On eight existing Panda models and a separate native Panda exploratory population, measure exact-rational residual peak/margin brackets, hull overestimation, depth/leaf counts, CPU cost and all query denominators. Compare unchanged TimeOpt output under no-split, fixed uniform and adaptive subdivision. Analytic fixture does not count as native path. Negative no-split evidence is important. Separate physical and planning-scaled limits and preserve model/control/duration hashes.

## 5. UR5e and formal qualification
Use official UR5e description at 78c5afd8bfee78a459c0265304340a85f1d44792, validated independent FK. Acceleration/jerk limits and cuRobo collision config remain unresolved: do not invent limits or substitute UR10e. Panda exploratory work should proceed independently. Only after native-linkage correctness and two-robot source cohorts pass should formal Dev40/Holdout100/Confirm300 and complete-request timings run. Include native solve, copying, transfers, certification/replay and synchronization; no holdout tuning.

## 6. Workflow and handback
For each package inspect -> implement -> test -> diagnose/fix -> retest -> commit -> update docs/PROGRESS.md. Run Python/C++/Node/offline browser/native CUDA tests where applicable. Record SHA and provenance, preserve private NPZs ignored, no PR/merge.

Report branch/HEAD/clean state; compiler flags and source/binary hashes; full 10k+ exact-vs-compiled case counts and failures; native operand error decomposition; real Panda matched-scope per-stage p50/p95/p99 and raw repetitions or NOT RUN; K3 mathematical/native/full scopes; subdivision data; UR5e state; interactive HTML and blockers. Section 8 NOT REACHED unless the formal gates genuinely pass. Never fabricate measurements or call the hull-only checker a full certificate.

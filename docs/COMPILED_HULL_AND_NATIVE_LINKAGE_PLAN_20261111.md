# Compiled K3 Bernstein hull + native operand linkage — experiment plan
Date: 2026-11-11. State: CODE IMPLEMENTED LOCALLY, FORMAL EXPERIMENTS NOT RUN.
See the existing docs/CAMPAIGN_PLAN.md for registered scientific acceptance.

## Provenance and branches
Latest user-reported local unpublished branch: exp/timeopt-jerk-tightness-integration-20261111, HEAD 54e3905ec058f7f060eb3ea5b908e81e63f895e9.
This independent work is descended from published exp/timeopt-jerk-and-tightness-20261111 at e99c3d50ef914b174a12c570818023e498135883. **Do not overwrite the unpublished local branch or ignored runs/**. On GPU machine, create a new child from local 54e3905, then carefully cherry-pick the independent work-package commits. No PR, merge, reset or force-push.

Reported prior facts (not replicated here): eight real Panda models each had 47 spans/7 joints, no subdivisions and successful mathematical K3 proof replay. Native jerk parity still fails the registered 4e-4 scaled-relative threshold despite 1,176 bitwise-consistent native device traces and captured-operand fmaf replay. Exploratory outward Python + proof replay p50 was ~1.34–1.62 s on the eight real paths, not production or formal campaign latency. UR5e URDF FK passed; acceleration/jerk limits and collision model unvalidated. No Dev40/holdout/confirmation.

## New source modules
- src/curobo_certify/native/kinematic_hull.cpp: C ABI initial K3 Bernstein coefficient interval hull screening, q/x cubic local power, degree padding, global-s derivative scaling, 5 families (positive x, velocity, ±acceleration, squared jerk), interval input endpoints and conservative nextafter arithmetic.
- src/curobo_certify/compiled_hull.py: exact Fraction -> enclosing binary64 input arrays, guarded ctypes, mandatory partial-only semantics; positivity counted once per span.
- scripts/build_compiled_hull.py: reproducible GCC shared object with -O3 -fno-fast-math -ffp-contract=off -fexcess-precision=standard and hashes.
- scripts/bench_compiled_hull.py and scripts/render_compiled_profile_html.py: stage timing and offline interactive chart, explicitly hull-only.
- scripts/attribute_native_jerk.py: empirical bitwise-operand ULP analysis and geometric/timing Shapley decomposition of jerk, NOT a continuous-time error bound.
- tests/test_compiled_hull.py, tests/test_jerk_operand_attribution.py, tests/test_compiled_profile_report.py.

All compiled hull outcomes are initial per-family inequalities **only**. Never promote them to CERTIFIED_K3 without separately audited endpoint, knot continuity, strict interior positivity/integrability, independent proof replay, and native linkage. Positive hull upper is UNKNOWN rather than a rigorous violation witness.

## C0. Local integration and build gate
Review local branch ancestry and preserve all current evidence and pinned cuRobo c301cb00a87012c2808b1a60649c4065c9b3c95a.
Build:
  python scripts/build_compiled_hull.py --out /tmp/libk3hull.so --manifest /tmp/libk3build.json
  python -m unittest discover -s tests -p 'test_*.py' -q
  python -m compileall -q src scripts tests
  python scripts/bench_compiled_hull.py fixtures/boundary_multispan_analytic.json --lib /tmp/libk3hull.so --out /tmp/hull.json
  python scripts/render_compiled_profile_html.py /tmp/hull.json --out /tmp/hull.html
Record source/binary SHAs, compiler flags, CPU ISA, IEEE binary64/FLT_EVAL_METHOD, rounding mode, affinity and platform. Binary64 nextafter enclosure assumes correctly-rounded primitives under documented compiler flags; cross-platform formal conformance remains unproven. Treat numerical overflow/nonfinite as invalid/unknown.

## C1. Correctness and arithmetic audit
Cross-check every compiled residual Bernstein coefficient against an independently implemented exact Fraction and where available MPFR directed-rounding oracle. IMPORTANT: q,x are padded to degree 3; resulting residual Bernstein degrees are positivity 3, velocity 7, acceleration ± 4, jerk 9 even when actual polynomial degree is smaller. Degree elevation may change the *hull maximum*, although not polynomial values; compare at equal degree. Include at least 10k seeded multijoint cases, tiny spans, non-dyadic rationals, ±0/NaN/Inf, cancellation, tight/violated limits, zero-speed endpoints/interior zeros, overflow, varied optimization flags, GCC/Clang and fenv probes. Zero observed under-enclosures; any counterexample blocks certification usage. Compare against existing outward Python interval coefficients and exact oracle. No validation from finite tests alone justifies a universal platform proof.

## C2. Eight real Panda frozen-output exploratory performance
Use the existing eight hash-verified original NPZs and exported exact effective cubic geometry and timing controls. No fitting sampled output. On *same CPU*, benchmark:
(a) current outward Python full certificate+independent proof replay;
(b) matching Python hull-only residual construction and hull check;
(c) compiled hull-only component, including exact input packing;
(d) a compiled+reference fully verified pipeline if integration works.
Account separately for parse/cloning, coefficient extraction, residual polynomial arithmetic, Bernstein conversion, hull check, boundary/strict-positivity, proof capture/replay and serialization. Use 50 warmups and at least 10 paired repeats/path, paired AB/BA order where scopes match, raw repetitions, CPU affinity and binary hashes. **Never compare 3 ms compiled hull-only with 1.4 seconds full Python proof as equivalent workloads.** Report frozen-output exploratory distributions; complete native TimeOpt request and Dev40 remain NOT RUN.

## C3. Device-side jerk operand error attribution
Reuse the 1,176 captured native traces and reconstruct adapter operands at precisely the same native internal s, with matching source SHA and idx/joint/t. Run scripts/attribute_native_jerk.py on parsed native trace JSON and adapter operand JSON (q1 q2 q3 x0 x1 x2 s jerk). Report per-operand FP32 ULP and absolute errors, symmetric Shapley geometric/timing effects, final FP32 arithmetic residual and progress differences. Algebraic decomposition is *diagnostic*, not proof of causal source when s differs; a mismatch in progress must be investigated separately. Pristine-vs-instrumented arrays must remain bitwise identical. Never relax 4e-4 and never call empirical worst error a rigorous bound. Establish a continuous-time error enclosure of coefficient construction, spline evaluation, time inversion and jerk arithmetic before native-linked certification.

## C4. Real-path refinement and UR5e
For the eight existing Panda models, record original physical-limit residual margin, exact polynomial maximum bracket, Bernstein overestimation, zero-split coverage and per-family counts. Expand a separately labeled Panda-only exploratory cohort of genuine, unchanged native outputs stratified by geometry and utilization. Analyze why natural paths do/don't require splitting. Analytic hidden-peak fixtures are not real paths. UR5e description at official pin 78c5afd8bfee78a459c0265304340a85f1d44792 still needs defensible acceleration/jerk physical limits and validated collision/curobo configuration; no substitution of UR10e or invented limits.

## C5. Formal gates and final report
Only after native arithmetic/error proof and validated Panda+UR5e cohorts, freeze Dev40 and proceed to official matched frozen/complete-request comparisons (H0, uniform, adaptive); subsequently lock code then Holdout100/Confirm300. Keep published engineering targets K3 >=99% eligible native successes, fresh added p50 <=1 ms and p95 <=3 ms on RTX3090. All requested torque/torque-rate status remains UNSUPPORTED_DYNAMICS. Confidence intervals by source scene/query with complete denominators. No qualification decision until gates met.

Each phase: inspect -> implement -> meaningful tests -> diagnose/fix -> rerun -> commit -> update docs/PROGRESS.md with exact commands, failures, hashes, metrics and output artifacts. Final HTML needs real per-path stage bars, proof hull/margin, operand ULP plots, subdivision trees and provenance. Section 8 NOT REACHED until genuine eligible campaign evidence exists.

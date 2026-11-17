# Coding agent: complete K4 continuous-time torque certification research

Repository: pkicki/curobo_certify
Branch: exp/k4-certified-torque-reference-20261117
Parent/frozen K3 branch state at creation: 2c9b9b72158c2be62aba1890a168b79d4f96a64b
Read the binding detailed plan: docs/K4_CERTIFIED_TORQUE_RESEARCH_PLAN_20261117.md
Other required context: docs/CAMPAIGN_PLAN.md, docs/PROGRESS.md, docs/NATIVE_LINKAGE_20261116.md, docs/T1_V2_CONTRACT_20261117.md, docs/COMPILED_K3_CERTIFIER_20261116.md, src/curobo_certify/k4_reference.py, src/curobo_certify/k4_affine.py, scripts/run_k4_reference_pilot.py, tests/test_k4_reference.py, tests/test_k4_affine.py.

The user authorizes autonomous implementations and experimental execution. Implement and test, fix bugs, commit coherent packages, run experiments only after their gates, report complete results. No PR, merge, force push, reset, overwrite of historical results or main changes. Do not alter or inspect ongoing K3 Panda-Confirm300 as a virgin K4 population; isolate all K4 development on this branch.

## 1. First: run and audit the committed analytical reference

Use Python >=3.11,<3.13 and the pinned repository environment. The upstream cuRobo pin is c301cb00a87012c2808b1a60649c4065c9b3c95a.

Run:
  python -m compileall -q src scripts tests
  PYTHONPATH=src python -m unittest tests.test_k4_reference tests.test_k4_affine -v
  python -m unittest discover -s tests -p 'test_*.py' -q
  node --test tests/*.test.cjs
  PYTHONPATH=src python scripts/run_k4_reference_pilot.py --out runs/k4/synthetic_reference.json --depths 0 2 4 8 12 --warmups 5 --repeats 10

If a command has environment obstacles, document precisely and fix real code/config issues rather than claiming PASS. Run Chromium/Firefox browser QA where available. Check original K3 baseline/regression tests; do not change frozen K3 algorithm. Rebuild and smoke-test the new K4 CI branch workflow. Test new Python under both 3.11/3.12.

Independently scrutinize the math in k4_reference.py and k4_affine.py: exact rational interval arithmetic, signed multiplication, sin/cos Taylor remainder and interval radius, negative and zero x, C2/C1 knot obligations, two-link mass matrix and Coriolis sign, asymmetric limits, point-witness disjointness, leaf replay, cache hash collisions/model parameter mutation and failed budget handling. Source fixtures use analytic 1-DOF/2-DOF planar models and DO NOT certify Panda. The replay currently reuses model dynamics formulas: distinguish structural replay from INDEPENDENT dynamics verification. Fix bugs, add failing regression tests first, rerun. Ask an isolated reviewer/independent oracle to derive and check equations; include signed proof obligations and evidence.

## 2. Freeze and independently validate the real robot dynamics

Complete K4 P0 as described in the plan. Locate EXACT pinned cuRobo dynamics source and franka.yml, available URDF or robot description, physical torque limits and the fixed-frame attachment of payload inertias. Trace source of every mass/CoM/inertia, gravity vector, joint name/order/sign, torque scale and model output conventions. Do not invent missing physical hardware limits or payload inertial transforms.

Implement immutable source-hashed robot/dynamics/model/payload contract and independent F(q,v,a) numerical parity against Pinocchio (or equivalent) on at least 100 frozen seeded states, gravity-only, full velocity/acceleration, nonzero q2-type couplings. Check whether the cuRobo coefficient identity
  G=F(q,0,0)
  B=0.5*(F(q,0,q')-G)
  A=F(q,q',q'')-G
holds under actual model conditions; friction, damping, rotor, external wrenches or floating-base assumptions may invalidate it. Record both native and independent values with per-joint errors, input hashes and source details.

If no authoritative robot dynamics and asymmetric effort limits are available, issue BLOCKED_MODEL and STOP real K4 certification, but complete synthetic and independently checked development. Never silently rebrand a nonmatching model as the real Panda.

## 3. Implement general, rigorous interval spatial inverse dynamics

Build a source-backed interval RNEA over joint positions and path derivatives as EXACT/OUTWARD enclosures. Propagate joint configuration interval sin/cos, spatial transforms, velocity, acceleration, inertial forces and wrench recursion including validated payload. Use Bernstein bounds for q,q',q'' on the common q/x spline knot partition; recover intervals A(s), B(s), G(s). Do not obtain proof from min/max of 511 sampled inverse dynamics evaluations.

Add an independent verifier deriving model coefficients from a different algorithm or independently checked interval formulation (CRBA+bias versus RNEA). MPFR/arb exact enclosures are valuable; ordinary Pinocchio/dense double evaluations are DIAGNOSTICS only. Test analytic pendulum/two-link and a general 7-DOF fixed-base Panda model; random states, knots and near-torque-limit conditions. UNKNOWN is a valid outcome; no false CERTIFIED. If the interval backend is too conservative, report node budget and enclosure widths rather than inventing a feasible witness.

## 4. Certified geometry-once, retime-many approximations

Build on k4_affine.py, which is only a correctness reference. Replace per-call recomputation with prepared, independently verified, model/source/geometry-hash-keyed coefficient enclosures for A(s), B(s), G(s).

Compare:
- direct interval RNEA over every interval;
- uniform/subdivided geometry boxes;
- rigorous polynomial Taylor/Chebyshev approximants WITH independently certified remainders;
- Bernstein residual polynomials enlarged by remainder bounds;
- adaptive versus uniform subdivision.

Prove that A/B/G boxes remain valid for all s and all new x(s) on the same geometry, but reject changed geometry, model, gravity, payload, frame or precision. Signed x'(s) must be handled correctly. Separate one-time geometry preparation, repeated timing certification, production memory bandwidth and verifier latency.

A polynomial fit to sampled torques, with no valid remainder, is NOT a certificate. A trajectory with positive Bernstein hull coefficient is UNKNOWN unless a rigorous evaluated-point torque interval proves a violation. Continue to recheck K3 positivity/integrability and kinematic constraints before declaring K3+K4.

## 5. Frozen Panda exploratory pilot on actual cuRobo GPU

Only if model and interval RNEA gates pass, freeze a NEW K4 source-query manifest. Use 0.95 untouched native TimeOpt as primary, separate 0.99 paired arm and 1.00 stress arm. Verify 0/3/5 kg payload inertias, collision and attachment mapping before using them, otherwise skip and label BLOCKED_PAYLOAD. Do not substitute unmodeled payload mass or fabricated limits.

Begin with 40 preregistered source-query groups per supported payload condition (adjust only BEFORE running and document). Keep native failures, output SHA, actual constructed TimeOpt config and planner nondeterminism, eligible selection rule and source-query cluster grouping. Repeat native planning on frozen seeds for output variability, but certify the precise frozen arrays from each result.

For all successful native trajectories report separately:
- mathematical K3/model torque K4/combined K3+K4;
- native sampled RNEA at default 511, 8193 and event-aware 32768 points as diagnostics only;
- rigorous K4 exact witnesses and per-joint error bands vs native sampled peaks;
- source-level paired duration vs 0.95, torque utilization and safety margin by scene and payload;
- CERTIFIED, VIOLATED_TORQUE, UNKNOWN and failed native attempt denominators, never just accepted outputs.

Do not preempt an independent K4 holdout. If exploratory Dev passes and independent reviewer approves, freeze new and *untouched* K4 Holdout100/Confirm300 with distinct source groups. Keep K3 formal campaign completely separate.

## 6. Fast compiled certification and CUDA linkage

Implement a fast compiled K4 proof producer and independent dynamics-aware verifier when correctness gates allow. Treat replayed leaf partition and raw model/payload/input SHA as mandatory; test alternate compiler flags and outward rounding. Refuse unsafe FAST_MATH, FTZ/DAZ or unsupported ISA cases unless rigorously accounted for.

To claim NATIVE_REPORTED_TORQUE, independently analyze actual pinned CUDA inverse-dynamics instructions/FP32 error and compare native torque against exact model over each progress span. The K3 T1-v2 time-law audit does not automatically cover torque. Model error, tracking and actuator saturation are separate. Mark unsupported portions clearly.

Measure frozen-output costs AND full fresh request latency in randomized A/B paired order (50 warmups, >=10 repeats, 8 CPU threads, GPU profile), with producer, verifier, RNEA, copying, geometry prep, SHA/model validation and timing-only reuse all separate. Report p50/p95/p99/max plus cold/first call and CI. Existing K3 ~2ms overhead is baseline only, not an achieved K4 result.

## 7. Optional: torque-aware certificate-guided retiming

If the coefficient enclosures are sound and cheap, enforce conservative Bernstein affine inequalities on timing spline coefficients r_k (x=sum r_k B_k), including signed x' and certified remainder margins. The goal is to reduce duration while passing *all* frozen MODEL_K3 and MODEL_K4 obligations. Jerk/positivity/duration are not automatically linear or globally convex; use verified repair or constrained SQP with post-solve proof. Never accept UNKNOWN.

Compare native CuRoboV2-Fast 0.95/0.99, posthoc K4 rejection, conservative fixed-geometry retiming repair and optional geometry optimization as a DIFFERENT study. Distinguish gravity hold infeasibility from path timing infeasibility (dynamic terms may compensate gravity). If duration gains are absent or overhead dominates, report NO-GO and preserve negative experiments.

## 8. Deliverable and STOP protocol

Append docs/PROGRESS.md; add docs/K4_*_RESULTS_20261117.md (or dated equivalents), raw/hash-backed manifests, external numerical models/provenance table, C++/Python/Node/test logs, independent reviewer matrix, exact benchmark commands and environment snapshots. Produce a truly offline interactive HTML report with q/x, torque-limit bands, sampled vs rigorous-enclosed torque, certificate leaf tree/witnesses, geometry prep and reuse latency, margin/payload comparisons and source-level confidence intervals. No fabricated values.

Hard STOP on missing model/limits, unsound trig/RNEA/interval, false certified/false witness, verifier disagreement, altered NPZ, wrong payload or collision model, native CUDA mismatch or contamination of K3 qualification. Commit a partial STOP report with findings if any gate fails. Otherwise implement, fix bugs, commit/push in packages, and report all valid outcomes.

Final answer: branch and SHAs, clean worktree, tests and review, P0-P7 gate status, models/assets/hardware, strict claim scope, attempted/successful/eligible counts and confidence intervals, torque feasibility and violation witnesses, prep/reuse/full fresh latency, duration gains or negative results, interactive report paths, blocked/open items. Never claim a torque-certified Panda trajectory until the independent MODEL_K4 verifier and real model gate pass.

# K4: certified inverse-dynamics torque — detailed implementation and experimentation plan

Date: 2026-11-17
Branch: exp/k4-certified-torque-reference-20261117
Parent commit: 2c9b9b72158c2be62aba1890a168b79d4f96a64b
STATUS: Analytic P1 and cached-geometry P2 reference components implemented. Full Panda RNEA, CUDA linkage, real payload experiments and K4 formal cohorts NOT RUN. Never change the ongoing K3 Panda-Confirm300 freeze, its data, or main. No PR, no merge, no force push.

## A. Objectives, mathematics and claim boundaries

Goal: independently verifiable continuous-time inverse-dynamics torque feasibility on actual cuRobo TimeOpt outputs, preserving kinematic K3, including torque-active heavy payloads, then test certification-aware retiming. A numerical sampled torque maximum or a fitted torque polynomial alone cannot certify.

For a fixed geometric path q(s), squared path speed x(s)=(ds/dt)^2, and fixed frictionless rigid-body model F(q,v,a)=M(q)a+C(q,v)v+g(q),

    qdot=q'(s)*sqrt(x(s)),
    qddot=q''(s)*x(s)+0.5*q'(s)*x'(s),
    tau(s)=A(s)*x(s)+B(s)*x'(s)+G(s),
    A=M(q)q''+C(q,q')q',
    B=0.5*M(q)q',
    G=g(q).

Independent numerical coefficient identity check with F:
    G=F(q,0,0);
    B=0.5*(F(q,0,q')-G);
    A=F(q,q',q'')-G.
These evaluations at samples are NOT proofs across intervals. This is the established TOPP affine-in-x,x' reduction, not a novel derivation. Inertia/payload and actuator limits must be sourced and fixed. Friction, contacts, torque-speed derating and motor dynamics require independent contracts.

K4_MODEL_TORQUE refers exclusively to inverse-dynamics model torque for exact serialized q/x over every s. K3_PLUS_K4_MODEL requires BOTH K3 and K4 proofs for the IDENTICAL bytes/model/physical limits and endpoint/positivity contract. NATIVE_REPORTED_TORQUE requires separate CUDA FP32 error analysis and must not inherit the T1 v2 q/v/a/j proof by assertion. ROBUST_ACTUATOR_TORQUE (friction, model/payload uncertainty, motor limits, tracking) remains UNSUPPORTED.

## B. Current implemented, committed P1/P2 components

- src/curobo_certify/k4_reference.py: exact-rational interval arithmetic; rigorous sin/cos enclosure via polynomial Taylor remainder + Lipschitz radius, conservative [-1,1] fallback for |angle midpoint|>16; analytic pendulum and planar two-link dynamics; exact q,x Bernstein bounds; C2 q/C1 x continuity checking; asymmetric torque limits; adaptive dyadic subdivision, rigorous torque point witnesses and explicit UNKNOWN_DEPTH/NODES; strict SCOPE="ANALYTIC_PLANAR_MODEL_TORQUE_ONLY_REQUIRES_SEPARATE_K3_AND_DYNAMICS_VALIDATION".
- Reference proof replay recomputes subinterval power polynomials via a different exact affine substitution and verifies full dyadic partition/no overlaps/leaf bounds. It shares the same dynamics formulas and IS NOT an independent dynamics implementation.
- src/curobo_certify/k4_affine.py: frozen geometric path + model-hash keyed interval boxes for A(s), B(s), G(s), verified by recomputation, and sound combination with alternative x(s),x'(s) profiles. Cache is NOT YET fast since it rechecks all boxes on every call.
- tests/test_k4_reference.py and tests/test_k4_affine.py: 18 focused analytic tests passed in a local isolated Python workspace; tests for exact inertial 2-link, gravity, true torque witness, depth UNKNOWN, structural invalidity, negative x, tampered leaves, geometry/model cache mismatches and affine timing dependence.
- scripts/run_k4_reference_pilot.py: deterministic synthetic correctness pilot, recorded fixture/module hashes, repeat wall-times. Local synthetic depth 0/4/8 outcome totals: 8 CERTIFIED, 6 VIOLATED_TORQUE, 1 UNKNOWN_DEPTH across 15 case/depth cells. These timings are Python exact-rational reference timings, NOT Panda/GPU/full-request latency.

Known assumptions/limitations: only 1/2-link planar analytical models; q and x interpreted as exact rational polynomials; gravity convention fixed by fixture; sine/cosine proof depends on exact rational Taylor/Lagrange bound (independently audit it). Zero speed at an internal s can pass a torque-only test, but must fail K3 traversal integrability before a combined certificate. Model errors, friction, actuator derating, torque rate, actual hardware tracking and native CUDA numerical operations remain out of scope. No claim of a real Panda torque certifier has been made.

## C. P0 — dynamics/actuator model freeze (MANDATORY BEFORE PANDA CERTIFICATION)

Inspect pinned NVlabs/curobo c301cb00a87012c2808b1a60649c4065c9b3c95a. Identify exactly how cuRobo computes torque (joint order, spatial RNEA, coordinates, frames, gravity, effective payload transform, motor inertia, floating base/fixed base, link inertias and constraints), and where its TimeOpt solver evaluates torque. Do not assume the robot dynamics model matches a generic FRANKA URDF.

Create immutable k4_model_contract:
- robot/URDF or cuRobo dynamics source SHA, exact masses/CoMs/inertia tensors, joint axes/frames, gravity, base frame, payload mass/CoM/inertia and attachment frame, and inverse-dynamics sign conventions;
- source-backed physical asymmetric lower/upper torque effort bounds; any torque-speed/current/thermal constraints explicitly supported or excluded;
- native TimeOpt coefficient/dynamics evaluation parity and independently configured Pinocchio/RNEA comparator for diagnostics;
- unique input SHA of geometric controls, timing controls, model/payload and torque limits, with no implicit reuse across changed payloads.

Test 100+ random configurations and v/a states including gravity-only and inverse-dynamics coefficient identity (0,3,5 kg payloads only if physically validated). Independent Pinocchio agrees numerically to stated tolerance; this is a DIAGNOSTIC, not continuous-time proof. Stop BLOCKED_MODEL if model mismatch, missing inertia, unsourced torque limits, or unsupported payload transformation. Never fabricate limits.

## D. P1 — analytically proven correctness and independent review

Review the committed exact interval reference independently. Audit trig enclosure and its radius bound, interval multiplication sign cases, exact power-to-Bernstein conversion, subspan global-s derivative scaling, dynamics formulas including gravity/coriolis signs, velocity-independent torque and positivity/integrability obligations, asymmetric torque limits, endpoint sides. Compare against an independently implemented rigid-body/closed-form model (not only two calls to the same equations). Add tests for q2≠0 and nonzero gravity on two-link, changing x', interior true witnesses, near-boundary unknowns, negative x, x zero interior, angles >16, nonlinear interval dependency and model parameter uncertainty.

Proof replay must detect omitted or overlapping leaves, altered model/payload parameters, fake witnesses and too-narrow interval boxes. A true witness needs a rigorously DISJOINT point torque enclosure outside torque limits; a Bernstein hull with positive upper bound is only UNKNOWN. Do not promote reference replay sharing dynamics code to independent dynamics verification.

P1 PASS: reviewer signs off mathematical analytic formulation, independent oracle tests, 0 unsound certificates. Fail => fix, commit regression and re-review before progressing.

## E. P2 — rigorous spatial interval RNEA for Panda

Implement spatial vector interval rigid-body inverse dynamics over Bernstein-enclosed joint configuration/path derivatives, including trigonometric transform intervals and payload inertia. Use the frozen model P0. Ensure outward arithmetic correctness, subnormal/overflow handling, angle reduction or safe [-1,1] fallback, and consistent frame/sign conventions.

Produce geometry-dependent interval functions [A_i],[B_i],[G_i] for each common q/x knot interval, and per-joint continuous-time torque enclosures. The direct interval RNEA backend is the sound conservative reference; many UNKNOWNs are acceptable before refinement. An independent verifier using an alternate spatial formulation (e.g. interval CRBA plus bias) should recompute model dynamics; an ordinary numeric Pinocchio comparator alone cannot verify enclosures.

P2 PASS: interval bounds contain all independently evaluated exact/rigorous analytic test cases, no false CERTIFIED, real Panda model parity when physically sourced. Stop if unresolved frame or payload mismatches.

## F. P3 — reusable certified polynomial dynamics approximations

For each frozen geometry/model span, develop a tight interval Taylor model, rational Chebyshev approximation or rigorous polynomial enclosure of A(s),B(s),G(s). Approximation must include an independently verified REMAINDER bound over the ENTIRE span: sampled fits are not sufficient.

If A=Atilde+eA, B=Btilde+eB, G=Gtilde+eG with bounds eps_A/B/G, then for a new timing profile

  tau_approx(s)=Gtilde(s)+Atilde(s)x(s)+Btilde(s)x'(s)
  eps_tau <= eps_G+eps_A*max|x|+eps_B*max|x'|.

Use exact/outward Bernstein coefficients of tau_approx, then enlarge by eps_tau. Signed x' and signed coefficient intervals are essential. This yields continuous-time sufficient bounds for each asymmetric torque limit and permits reuse of geometry-heavy calculations across timing candidates.

Compare reference interval RNEA, uniform leaf boxes, adaptive subdivision and certified polynomial remainders. Depths 0/1/2/4/8 and node budgets 2048/8192/32768; retain leaves and UNKNOWN explicitly. Freeze geometry SHA, robot/payload dynamics SHA, backend/precision/source hash and limits. Factor actual precompute cost vs cheap reuse cost, and audit the stored boxes before using them.

P3 PASS: independently checkable coefficient enclosures, substantially reduced UNKNOWN/overconservatism without unsoundness. The current k4_affine.py is only an analytic prototype with recomputation and must not be presented as low-latency Panda K4.

## G. P4 — exploratory frozen Panda/payload campaign (GPU required)

Do not open, alter, retune or otherwise contaminate K3 Confirm300. Already seen K3 outputs may be used as K4 EXPLORATORY cases only. Freeze entirely new K4 source-query groups/splits before inspecting torque results.

Primary: native unchanged cuRoboV2-Fast 0.95 TimeOpt margin, original physical torque limits. Secondary: separately registered 0.99 margin, paired on identical source start/goal/scene; 1.00 as stress, separately reported. Payload 0/3/5 kg ONLY if P0 verified masses, CoM, inertia, collision and attachment transforms. Stratify empty/table/simple/clutter and torque utilization; include gravity-active states, severe deratings, near-limit trajectories, and source-native planning failures.

Initial Dev proposal: 40 source-query groups per validated payload, 0.95 vs 0.99 paired; 3 repeated native plans per source to diagnose output nondeterminism. Predeclare first-K eligible selection, never discard native failures or retroactively pick favorable solutions. Record source/geometry/timing/robot/limits/payload hashes, binary and GPU identity, true attempted denominator, eligible, source-level paired duration.

For each immutable output: MODEL_K3, MODEL_K4, conjunction, optional NATIVE_REPORTED_TORQUE only when P5 passes; sampled native torque checks at 511/8193/32768/event-aware as DIAGNOSTICS, exact interval torque witnesses by joint/constraint, proof depth/nodes, cumulative CPU/GPU costs. Compare native sampled-feasible but continuously violated cases; reported torque bounds/unknown versus actual utilization. Stop on disagreement/unsoundness. After Dev, freeze new untouched Holdout100 and Confirm300 ONLY when independent dynamics review and model contracts pass.

## H. P5 — complete compiled K4 producer and independent verifier; native linkage

Once P2/P3 work:
- Implement fast C++/CUDA producer with rigorously outward rounding, interval products, trigonometric enclosures, certified model and timing coefficients, adaptive certificates and exact point witnesses; test O0/O3/-march=native and FP rounding modes. Use existing K3 compiled infrastructure but NEVER mutate frozen K3 production code.
- Write independently developed proof verifier that reconstructs model/path/coefficients from frozen data, checks all dyadic partition coverage, bounds, fingerprints and original physical torque limits; quantify witness replay.
- Audit cuRobo native inverse-dynamics FP32 differences using actual source, compiled ISA, model-match, time-law assumptions; native same-progress bound separate from progress inversion. T1 v2 does NOT automatically certify torque. If unable to prove native output enclosure, mark BLOCKED_NATIVE_TORQUE.
- Benchmark geometry prep, coefficient enclosure, host/gpu transfers, compiled producer, verifier and full fresh-request latency with randomized 50 warmups/10 paired repeats, CPU affinity and 8-thread overhead. Compare to native-only and current K3 (~2ms added), but do not declare K4 low latency before measurement. Include cold startup, p50/p95/p99/max and 95% source-cluster CIs.

P5 PASS: no false certificates or verifier disagreements, candidate model and native scope precisely documented, measured coverage-vs-latency frontier.

## I. P6 — certification-aware retiming (optional after K4 success)

Because x(s)=sum_k r_k B_k(s),

  tau_i(s)=G_i(s)+sum_k r_k[A_i(s)B_k(s)+B_i(s)B'_k(s)].

With rigorous coefficient polynomial/remainder bounds, Bernstein coefficient inequalities provide sufficient, conservative linear/convex constraints on the timing controls under properly bounded remainders. They are not generally necessary. The duration objective integral ds/sqrt(x) and jerk constraints remain nonlinear; full K3+K4 proof required after every candidate. No general convexity guarantee of the full problem.

Compare native 0.95 and 0.99, native + posthoc reject, certificate-guided repair of fixed geometry timing, and optional geometry changes as a separate experiment. Report timing-only gains, violations avoided, latency and feasible return rate; use frozen source pairs, no hindsight best-of-margin as deployable policy. Gravity > actuator limits at a static hold does NOT imply all dynamic traversals of that configuration infeasible; acceleration may counteract gravity. Only label geometry-dependent infeasibility if a separately verified necessary-condition/lower-bound proves it.

## J. Gates, reproducibility and publication

Immediate STOP on missing physical dynamics or torque limit sources, sample-based proof substituted for interval bound, invalid proof leaf/box, false CERTIFIED/witness, numerical underenclosure, inadequate payload model, source/hash mismatch, FP32 native assumption without proof, or frozen K3 qualification contamination.

For each experiment report attempted/native successful/K3/K4/K3+K4, VIOLATED and UNKNOWN separately, per joint, scene, payload, margin and source. Preserve all failures. Use source-query-level bootstrap (>=10,000 resamples), one-sided exact binomial confidence bounds for certification coverage, full raw manifests/hashes, p50/p95/p99 and worst-case end-to-end latency; never claim that dense checks are mathematical proof.

Deliver source, tests, independent-review memo, source- and payload-hash frozen datasets, plots and a standalone offline HTML explorer (trajectory, torque utilization, model enclosures, witnesses, Bernstein leaf tree, geometry-prep vs reuse and latency). Commit and push work incrementally, append docs/PROGRESS.md, no PRs, merges, resets or force pushes.

Publication contribution should emphasize: continuous-time independently replayable torque certificates for real GPU-optimized motions; numerical and payload model fidelity; certified geometry-once, retime-many constraints; and actual torque-aware timing gains. Affine fixed-path inverse dynamics is established in TOPPRA and cuRoboV2-Fast. Negative and unknown results are publishable research evidence and must not be hidden.

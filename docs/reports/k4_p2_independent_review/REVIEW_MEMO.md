# K4 P2 independent review: Panda interval-RNEA torque certifier, plus the G1/G3 fixes

Date: 2026-11-17. Branch `exp/k4-certified-torque-reference-20261117`.

The review covers commit **`6117b6ba1e16d7a436d600a6be49c786cffa2685`**. All evidence was
produced against a read-only `git archive 6117b6b` snapshot in /tmp, because the working tree
was being edited during the review. HEAD then advanced to `f41e2da` (P3). Between 6117b6b and
f41e2da the only change to a reviewed file is in `k4_interval.py`:

- the new `DIA` dual class (P3, NOT reviewed here);
- `NotImplemented` guards for dual operands in `IA.__add__/__sub__/__rsub__/__mul__`;
- dual dispatch in `isin/icos`.

The arithmetic for IA operands is byte-for-byte unchanged, so the IA verdicts below also hold
at f41e2da. The new test file was run against the f41e2da working tree.

## Reviewer identity

An isolated Claude subagent, the same reviewer as P1. It wrote none of the reviewed code. It
made no commits or git state changes, used no GPU, and modified no existing file.

## Overall P2 verdict

**P2 soundness: PASS. No false CERTIFIED, no false witness, no under-enclosure was found.**

**P2 gate as a whole: PASS WITH FINDINGS, not clean.**

- Three medium findings are provenance/binding gaps (F-P2-1, F-P2-2, F-P2-3). They mean the
  verifier's `pass` is a correct statement about *whatever inputs it was handed*. The
  certificate object does not bind itself to:
  - the frozen contract (the hash is not checked),
  - the full [0,1] span tiling,
  - the limits.

  These should be fixed before P4 cohorts publish verifier passes as evidence.
- The other findings are low or cosmetic.

What was checked:

- **Exact states:**
  - 60 random exact states × 5 quantities × 7 joints = 2,100 rigorous comparisons.
  - Producer τ, A, B and G, and verifier τ, all against a rigorous (mpmath.iv) Euler-Lagrange oracle.
  - Result: 1,988 CONTAINED and 112 EXACT_ZERO_CONTAINED; 0 DISJOINT, 0 AMBIGUOUS.
- **Sub-boxes:** 16,800 / 16,800 CONTAINED. These are A/B/G and torque enclosures on 40 random
  sub-boxes, radius 1e-6 to 5e-2, with signed x′ and x ≥ 0, against 12 dense 160-bit samples
  each, corners included.
- **Paths:**
  - 1,778 / 1,778 leaf-point checks CONTAINED, on the real fixture plus 30 synthetic paths.
  - 14 / 14 VIOLATED_TORQUE witnesses confirmed rigorously.
  - All 36 decided paths were independently verified.
- **Boundary:**
  - Bisection resolves the fixture's VIOLATED/CERTIFIED threshold to 3e-6 relative.
  - The true joint-4 peak torque from the oracle, 29.151767 N·m, lies strictly between:
    - the witnessed limit, 29.151691 (VIOLATED);
    - the certified limit, 29.151956 (CERTIFIED).
  - The band in between returns UNKNOWN_DEPTH. See `boundary_bisection.json`.

## Independent oracle

`independent_checks/k4p2/oracle.py` imports nothing from `src/`.

- **Model.** The reviewer parses the pinned URDF (SHA `8f4d6768…`) with their own XML code and
  applies the franka.yml settings:
  - lock_joints 0.04 m;
  - `attached_object` at `panda_hand`, identity transform;
  - default inertial 0.01 kg / 1e-4;
  - `ee_link` and `right_gripper` not loaded.

  The test `test_reviewer_urdf_parse_equals_contract_v2` shows this model equals contract v2
  field by field: names, parents, joint indices, xyz, rpy, mass, CoM and inertia, all as exact
  rationals.
- **Dynamics: energy method, different from both RNEA and the world-frame Jacobian-transpose
  verifier.**
  - Forward kinematics by transform composition.
  - Kinetic energy `T = ½Σ m|ċ|² + ½ΩᵀIΩ`, with `Ω^ = RᵀṘ` and time derivatives by forward-mode AD.
  - `M(q)w` by polarization of the quadratic form.
  - `c(q,q̇) = Ṁq̇ − ∇_q T` and `G = ∇_q V` by a second, nested AD layer.
  - Path torque uses the physical substitution `q̇ = q′√x`, `q̈ = q″x + q′x′/2`; the affine
    identity is not used.
  - Scalars are `mpmath.iv` at 160 bits (rigorous, widths about 1e-45), or `mpmath.mp` at
    160 bits for dense sampling, with ±1e-30 slack.
- **Sensitivity.** The oracle detects deliberately wrong models: flipped gravity sign,
  link-origin instead of CoM inertia, and the wrong rpy order (Rx·Ry·Rz). All give DISJOINT.
  The comparison therefore has teeth.

## Obligation matrix

Line references are to the 6117b6b versions:
- `k4_interval.py` (KI)
- `k4_panda_model.py` (PM)
- `k4_path.py` (KP)
- `k4_verify.py` (KV)
- `k4_reference.py` (KR)

Test names refer to `tests/test_independent_k4_p2.py`.

| # | Obligation | Verdict | Evidence |
|---|---|---|---|
| P2-1 | Outward rounding on every IA op | PASS | KI:76-111. add/sub/mul/sqr/scale widen with `nextafter` after each round-to-nearest result. 3,000 random pairs over magnitudes 1e-310 (subnormal) to 1e150 enclose the exact Fraction result (`test_basic_ops_enclose_exact_results`). Negation is exact. |
| P2-2 | mul/scale with inf/nan; overflow | PASS | KI:92-98: any nan product (0·inf) gives [−inf, inf]. Overflow gives inf. `finite()` (KI:114) rejects nan, inf and inverted intervals. KP:121 and KV:227 require finite torque. `scale(0.0)·inf` gives nan, which fails `finite`. Tested in `test_nonfinite_fails_closed`. |
| P2-3 | IA.const for Fractions | PASS (hazard N-IA1) | KI:58-65 uses correctly rounded `float(Fraction)`, widened by one ulp when inexact. 2,000 random Fractions over 1e-300 to 1e250 are all enclosed. **Hazard:** `IA.const(int)`, `IA.of(int)` and `scale(Fraction)` round to nearest and do not enclose (`IA.const(2**53+1)` gives a point at 2**53). Every current call site passes Fractions, exact float64 values or `scale(0.5)` (grep-audited), so nothing is affected today. |
| P2-4 | sin/cos Taylor (order 30) and Lagrange remainder | PASS | KI:151-167. The polynomial has degree 61 (sin) or 60 (cos), Horner in x², with outward coefficient intervals. The remainder `\|x\|^(d+1)/(d+1)!` uses the exact rational bound at the array's max \|mid\|, rounded up. At \|x\| = 8 the bound is below 1e-24 and the interval-Horner cancellation width is 1.26e-12 (`test_taylor_remainder_and_cancellation_at_8`). |
| P2-5 | Lipschitz widening and fallback | PASS | KI:170-183. `rad_up` is computed outward around the (inexact) float mid. This also covers subnormal inputs where 0.5·lo underflows (`test_subnormal_midpoint_lipschitz`). \|mid\| > 8, including nan, gives [−1,1]. Tests ran on 308 intervals (mid in [−9,9]; ±8 and nextafter(8); π/2, π, 3π/2; radii 0 to 3) in one vectorized call, so the remainder is shared across the array. The exact range, via the hull of endpoints and stationary points with rigorous π, was CONTAINED in every case (`test_trig_encloses_exact_range`). |
| P2-6 | FP-environment check | PASS (note N-IA2) | KI:32-41 detects FTZ/DAZ and non-nearest rounding, but only at import. A later library load that sets FTZ would go undetected. Recommend re-checking it in `certify_path`/`verify_k4`. |
| P2-7 | URDF rpy convention | PASS | PM:43-50 builds R = Rz(y)Ry(p)Rx(r). The verifier (KV:56-62) multiplies the elementary rotations explicitly. Both contain the oracle's rotation for 10 random rpy triples. The wrong order is DISJOINT (`test_rpy_convention_and_sensitivity`). For the Panda model this is moot: every origin has at most one nonzero rpy angle. |
| P2-8 | Link ordering and parent map | PASS | Contract v2 order and parents equal the reviewer's own URDF tree walk (`test_reviewer_urdf_parse_equals_contract_v2`). PM:92-98 enforces topological order and joint coverage 0..6. The v1 float32 link-7 mass equals float32(0.35973), not 0.36973, so cuRobo did NOT lump `right_gripper`; the 13-link set is confirmed (`test_contract_hash_and_v1_shows_right_gripper_not_lumped`). |
| P2-9 | Locked-finger translation | PASS (latent N-C1) | Contract xyz = (0, ±0.04, 0.0584), which is origin + 0.04·axis. This is correct because the finger joint rpy is 0. `scripts/k4_model_contract_v2.py:72-74` adds lock·axis WITHOUT rotating by the origin rpy, so it would be wrong for a URDF with a rotated prismatic origin. The reviewer's parser raises in that case. |
| P2-10 | Gravity sign | PASS | PM:106 sets base acceleration to −g; KV:137-138 uses F = m(a − g); the oracle uses V = −Σ m g·c. G is CONTAINED in all tests, and the flipped-sign mutant is DISJOINT. |
| P2-11 | Inertia about CoM | PASS | PM:90 builds Ic from (ixx iyy izz ixy ixz iyz), with N = Ic ω̇ + ω × Ic ω about the CoM. The contract script asserts inertial rpy = 0 (`k4_model_contract_v2.py:80-81`). The parallel-axis (link-origin) mutant is DISJOINT for A. |
| P2-12 | Producer A, B, G and τ at exact states | PASS | 60 rigorous states (`campaign_results.json` `exact_states`): 1,988 CONTAINED. The 112 EXACT_ZERO_CONTAINED are B with v = 0 exactly, where B = 0 lies in a ±1e-321 box. Zero DISJOINT or AMBIGUOUS. |
| P2-13 | Producer and verifier on random sub-boxes vs dense samples | PASS | 40 boxes × 12 samples × 5 quantities × 7 joints = 16,800 / 16,800 CONTAINED (`sub_boxes`). |
| P2-14 | Path leaves contain true torque (real fixture and synthetic) | PASS | The fixture with physical limits is CERTIFIED, 47 leaves; 560 / 560 oracle points CONTAINED. 9 scaled-limit variants and 30 synthetic paths (positive x, x(0) = 0, a negative-x pocket, x ≡ 0) add 1,778 / 1,778 CONTAINED in total, with no point outside the limits. |
| P2-15 | No false CERTIFIED near the boundary | PASS | Bisection is in `boundary_bisection.json`. The oracle's maximized peak is consistent with both statuses (see Verdict). Every CERTIFIED in the band is independently verified. |
| P2-16 | No false witnesses | PASS | KP:149-158 requires a strict, finite point-enclosure disjointness at exact rational s. All 14 / 14 campaign witnesses and 2 test witnesses are rigorously confirmed by the oracle. `verify_k4_witness` rejects forged u (1/2, 2, −1/3, "abc"), span (999, −1), joint (7, other), a CERTIFIED result, and limits other than those the witness violates (`test_witness_forgeries_rejected`). |
| P2-17 | Partition holes, overlaps, foreign spans (within the given spans) | PASS | KV:197-207: span set equals range(len(spans)), Kraft sum = 1, no adjacent prefixes in sorted order (sufficient). The verifier rejects a missing span, a duplicate, a parent+child overlap and a foreign span. A different but complete partition is accepted and re-proven, which is sound (`test_verifier_partition_attacks`). A malformed path symbol raises ValueError instead of returning `pass: False` (N-V1, fail-closed). |
| P2-18 | Holes at the span level ([0,1] tiling) | **FAIL (F-P2-2, medium)** | Neither `certify_path` nor `verify_k4` checks that the spans tile [0,1]. `spans[:20]` (covering [0, 8/19]) is CERTIFIED and verified, although KP:7 claims "every s in [0,1]". This is safe only when the spans come from `extract_timeopt_spans` (which does tile [0,1]). The verifier should rebuild the spans from frozen geometry/timing bytes, or at least check left₀ = 0, contiguity and right_last = 1. Test: `test_FINDING_span_coverage_of_unit_interval` (expectedFailure). |
| P2-19 | Leaf boxes too narrow | PASS for the verdict; **FINDING F-P2-4 (low)** | The verifier recomputes everything and ignores stored leaf boxes (KV:198). Zeroed leaf boxes with `worst_utilization = 0` still pass. The verdict is sound, but reported per-leaf enclosures and utilization numbers are unverified data. Either check stored ⊇ recomputed, or label them informational (`test_verifier_ignores_stored_leaf_boxes`). |
| P2-20 | Non-physical limits accepted as physical | PASS (verifier) / **FINDING F-P2-5 (low)** | `verify_k4(expect_physical=True)` requires exact equality with the contract's ±87/±12. It rejects 0.95-scaled limits and 2× limits (`test_limits_physical_gate`). The producer rejects nan and lower ≥ upper, but accepts ±inf limits and returns a vacuous CERTIFIED (`test_FINDING_nonfinite_limits_should_be_rejected`, expectedFailure). |
| P2-21 | Model contract binding in the verifier | **FAIL (F-P2-1, medium)** | The KV docstring (lines 4-5) says it checks the "model contract hash", but `verify_k4` never checks `contract_sha256`. A contract dict with link-7 mass set to 0.0001, or with zero gravity, and the hash unchanged, verifies the fixture certificate. Only `load_contract` checks the hash (PM:194-200). Test: `test_FINDING_verifier_should_check_contract_hash` (expectedFailure). |
| P2-22 | Result binds inputs (G3 for Panda) | **FAIL (F-P2-3, medium)** | `K4PathResult` (KP:42-52) carries no digest of spans, contract, limits or source. The G3 fix was applied to `k4_reference` only (`test_FINDING_path_result_binds_inputs`, expectedFailure). |
| P2-23 | Budgets fail closed | PASS | KP:110-112 checks the node budget before each BFS level. max_nodes 0 and 46 (below the 47 needed) give UNKNOWN_NODES. max_depth −1 gives UNKNOWN_DEPTH or a true witness. A dimension mismatch gives INVALID_INPUT. batch = 0 raises ValueError (cosmetic). Test: `test_budget_fail_closed`. |
| P2-24 | x < 0 handling | PASS | A leaf needs `x.lo ≥ 0` (KP:122, KV:227). `_f_lo` turns a tiny negative rational into a negative float, so it fails closed. A point x < 0 gives VIOLATED_X with exact evaluation (KP:142). A 1e-9 negative pocket at the non-dyadic s = 1/3 is never CERTIFIED: the result is UNKNOWN_DEPTH, or a true torque witness elsewhere (`test_x_cases`, synthetic neg_pocket cases). |
| P2-25 | Treatment of x = 0 | PASS (scope note N-X) | x(0) = 0 (start at rest) certifies and is oracle-contained. x ≡ 0 is CERTIFIED as a pure static-gravity claim for a path that is never traversed. Traversability and positivity must come from K3; KP:13-14 states this. |
| P2-26 | G1 fix: `replay_witness` | PASS | KR:446-465. It recomputes the exact point enclosure with strict disjointness, and checks range, type and x ≥ 0. It accepts a true witness and rejects forged u, joint and span, other limits, and another model (`test_replay_witness_accepts_true_and_rejects_forged`). Note it shares `_point_torque` with the producer; the P1 oracle is the independent check. |
| P2-27 | G3 fix: `input_digest` | PASS (finding F-G3, low) | KR:334-349 hashes the spans, model class and fields, and limits; changes to any of them alter the digest (`test_input_digest_sensitivity`). However, replay checks the digest only `if result.input_digest` (KR:409, 453), so stripping it silently disables the binding. Soundness is unaffected because replay re-proves (`test_FINDING_stripped_digest_should_not_bypass_binding`, expectedFailure). |
| P2-28 | P0 parity evidence | PASS (diagnostic) | `parity.json` and `test_k4_interval_rnea` check Pinocchio against the interval RNEA on the v1 parameterization with 1e-12 relative slack. This is a numeric diagnostic, not rigorous. Contract v2 had only producer-vs-verifier intersection evidence, which shares the interval layer and the span/Bernstein code. This review's v2 oracle closes that gap. |
| P2-29 | Physical fidelity | NOT CLAIMED (correctly) | Contract status `PHYSICALLY_INCONSISTENT_PARAMETERS`; link3 inertia is not positive definite. Scope `CUROBO_PINNED_PANDA_RIGID_BODY_MODEL`. P2 checks the code against the contract model, not the robot. |

## Findings summary

| ID | Severity | Finding | Recommended fix |
|---|---|---|---|
| F-P2-1 | Medium (fail-open on provenance; docstring false) | `verify_k4` does not check `contract_sha256`. A tampered contract passes. | Recompute the canonical model hash inside `WorldFrameDynamics`/`verify_k4` and require it to equal a frozen expected SHA passed by the caller. |
| F-P2-2 | Medium (fail-open on scope "every s in [0,1]") | The span tiling of [0,1] is not checked by the producer or the verifier. | Verifier: rebuild spans from the frozen geometry/timing bytes, or check that left₀ = 0, the spans are contiguous, and right_last = 1. |
| F-P2-3 | Medium (provenance) | `K4PathResult` has no input digest (spans, contract, limits, source). | Port the G3 digest to `k4_path`; have the verifier require it. |
| F-P2-4 | Low | Stored leaf boxes and `worst_utilization` are unverified. | Check stored ⊇ recomputed, or mark them informational. |
| F-P2-5 | Low | ±inf limits give a vacuous CERTIFIED. | Reject non-finite limits. |
| F-G3 | Low | The digest check is skipped when the digest is empty. | Require the digest to be present. |
| N-IA1 | Low (API hazard) | `IA.const`/`IA.of` on non-Fractions and `scale(Fraction)` round to nearest instead of enclosing. | Reject non-float/non-Fraction input; enclose ints exactly; accept only exactly representable floats in `scale`. |
| N-IA2 | Low | The FTZ/DAZ check runs only at import. | Re-check at each certify/verify call. |
| N-C1 | Latent | The contract script's prismatic lock ignores origin rpy. | Rotate lock·axis by R(rpy). |
| N-V1 | Cosmetic | A bad path symbol raises ValueError out of `verify_k4`. | Catch it and return `pass: False`. |
| N-X | Scope | x ≡ 0 or x = 0 at interior points certifies torque only. | Keep the K3 conjunction mandatory. |

No counterexample to soundness was found: no false CERTIFIED, no false witness, no
under-enclosure.

## SHA-256 of reviewed files

At 6117b6b (snapshot used for all evidence):

```
24e1d75fb3c9b55631585e1a6935f1128f5a33c103e372cb59f92623c7d58738  src/curobo_certify/k4_interval.py
29c274114f795eec6b5bc7bb957d8e92fcb424225fec89807af40b091473c46f  src/curobo_certify/k4_panda_model.py
5670197f048e61e3e7cd8a3c6ea09150e1750b2210aa79131da6902efb6da5d8  src/curobo_certify/k4_path.py
7b799a2dfef734bc8eb008cf8b17db88291fa2e460c768ea9a8102cf46ddca64  src/curobo_certify/k4_verify.py
a74a02bde75ecb30ba068eedb00d36673924fc311d64734a4d20c9d1b3232471  src/curobo_certify/k4_reference.py
9af0b96b6008dbb8ce58f7129b7ac5e1eed7c900649a7a5db728293f10aa95da  src/curobo_certify/spline_polynomials.py
5ba5ec3e24b41c48b5ba0db8387187f84fbbf7bf6456480778fa251d2a274152  src/curobo_certify/boundary_reference.py
851459c341974b7be32a3bb6ff93bb2d6f9561493d31833de71c3e0a4c5e6b36  docs/reports/k4/p0/panda_model_contract_v2.json  (contract_sha256 9caa38bf...)
13747cca75892a838d20a5f71952976199bf4e91bcf2ce78c2fc53fb248a017f  docs/reports/k4/p0/panda_model_contract.json
7499d4907f819852559dcec28bda70ae41906483a3339e7779e08fa62988b547  docs/reports/k4/p0/parity.json
81e0892b224b633b6a5fa91ca19efbf05cc0d3c32657f82c0a1c987239f06069  docs/K4_P0_MODEL_CONTRACT_20261117.md
d149bf74428735f16706196bee4bbc7467dcd241a055493e406e6634ec41f5e8  scripts/k4_model_contract_v2.py
da2a47d393334a5b13ce273fc4b577af030fe6b900eee9d38ab950d6c557cbac  scripts/k4_p0_parity.py
7d66645834e96ac9635e9e4b77c6442b5470d0c3b73405e9f9a3107aa0c3d0d8  tests/test_k4_interval_rnea.py
03662057427d8eedc1ed305890cc1fa805587a56c17c182c1d8e5cac6aaa6d28  tests/test_k4_path.py
a1112126d056a6c15cc97ffd091f83bf7c3a8573a3cd2d19dda13e858bee4e2f  fixtures/k4_real_path_margin0000.json
8afc8c4e88ceb79b0a45e651ed8bdba6209eab0332723a1cca3287fbc7a7fda1  fixtures/k4_pinocchio_parity_states.json
8f4d6768e7587ca6bbc1356007af48f26b91faa82025d21d2cadc9b22a1c4802  (pinned) franka_panda.urdf
91d48371b9cd2358b0b5594bb5e93d7abbd05af436b2c28d05c41b5667e65e55  (pinned) franka.yml
```

At f41e2da, only `src/curobo_certify/k4_interval.py` differs:
`531dda38e74ae4e024d72c720aba28dbd6dbec4941bb1137f80270dcd1379b54`. The difference is the
additive DIA code and dual guards described above.

Reviewer-created files:

```
b889e64a9ea66092765d024dffbd7ba9b84ffc3219f0dddd90d1d63fea6dd852  independent_checks/k4p2/oracle.py
2c169cfde94c66049d63cea05b66fe9f1482025c9a02c500812f295aa6ce5f53  independent_checks/k4p2/campaign.py
eb88438519b1337e7004d02c98ceef03accb8531ee45dc873fa8704e2ee1a05c  independent_checks/k4p2/run_review.py
4b1793756e5255ac484fe4d0f57236742aecd7154ce695c2bc0a73c75342a615  independent_checks/k4p2/__init__.py
43aba252f5740325c83d4ef7f4c9890e07f72ea3bea4c42e39026ccac56173bb  docs/reports/k4_p2_independent_review/campaign_results.json
f2d77684f0576bba4a2eb78c11233bd7a902c3ef28c912015dd78e3b5b4e3791  docs/reports/k4_p2_independent_review/boundary_bisection.json
2ff2d5f4ddde26498826ebfd3cebb1e49ff9429674e0a35b3ab39ba5da432355  tests/test_independent_k4_p2.py
```

## Reproduce

```
PYTHONPATH=src:. python -m unittest tests.test_independent_k4_p2 -v      # 30 tests, ~55 s, 5 expected failures = findings
git archive 6117b6b src docs/reports/k4/p0 | tar -x -C /tmp/snap
PYTHONPATH=/tmp/snap/src:. python independent_checks/k4p2/run_review.py \
    --contract /tmp/snap/docs/reports/k4/p0/panda_model_contract_v2.json --out /tmp/k4p2.json   # ~15 min CPU
```

## Results at review time (working tree at f41e2da plus the reviewer files)

- `tests.test_independent_k4_p2`: 30 tests OK, including 5 expected failures, which are findings F-P2-1, -2, -3, -5 and F-G3 (about 55 s).
- Existing reviewed tests (`test_k4_path`, `test_k4_interval_rnea`, `test_k4_reference`, `test_k4_affine`): 37 OK.
- Full suite `PYTHONPATH=src:. python -m unittest discover -s tests -p 'test_*.py'`: 353 tests OK (skipped=6, expected failures=5), 207 s.
- Campaign (6117b6b snapshot): 888 s CPU, 0 counterexamples, 0 ambiguous.

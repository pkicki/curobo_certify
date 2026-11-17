# K4 P1 independent review: analytic continuous-time torque reference

Date: 2026-11-17
Gate: P1 (plan section D, docs/K4_CERTIFIED_TORQUE_RESEARCH_PLAN_20261117.md)
Repository HEAD reviewed: `d04a9ceee74cafae886765d66bfe0b3c780fee65` on branch
`exp/k4-certified-torque-reference-20261117`.

## Reviewer identity

An isolated Claude subagent (Claude Code, Opus-class model) that did not write the
reviewed code. It was given only the repository and the review brief. It made no commits
or git state changes and used no GPU. It modified no existing file. The only files it
created are listed at the end of this memo.

## Verdict

**P1 analytic soundness: PASS. The P1 obligation list as written: NOT FULLY MET (3 gaps,
none unsound).**

- No unsound CERTIFIED result and no false VIOLATED witness was found. The checks were:
  1,500 randomized rational path/model/limit cases plus 45 targeted adversarial tests.
  Every claim was compared against an independently derived rigorous oracle (numbers in O19).
- The trigonometric enclosure, interval arithmetic, Bernstein conversion and split,
  derivative scaling, two-link M/C/g and the pendulum formulas are all correct. So are
  the knot-regularity checks, the closed asymmetric limits, the strict witness
  disjointness, the replay partition check and the k4_affine recomputation.
- Gaps, none of them unsound:
  - G1: there is no witness replay. A forged VIOLATED result cannot be detected by
    repository tooling.
  - G2: model-parameter uncertainty cannot be represented.
  - G3: certificates carry no input/model/limit hashes.
- Several low-severity scope and documentation findings are listed below.

Scope boundary, which is unchanged by this review: ANALYTIC_PLANAR_MODEL_TORQUE_ONLY.
The review covers 1-DOF and 2-DOF planar fixtures only. It does not cover Panda, the
interval RNEA, native FP32, K3+K4, friction or motor limits. The reviewer's dense oracle
sampling is a falsification diagnostic, NOT a proof. The soundness argument rests on the
line-by-line audit (O1-O15). The sampling only failed to refute it.

## Independent oracle (what "independent" means here)

`independent_checks/k4/oracle.py` imports nothing from `src/`. The oracle:

- evaluates the path itself in exact Fractions. It uses a plain power sum, not Horner,
  and its own derivative formula `sum c_n n!/(n-k)! u^(n-k) / h^k`.
- computes torque with a **planar projected Newton-Euler / Kane recursion**:
  `tau_j = sum_{i>=j} perp(pc_i - p_j) . m_i (a_ci - g_vec) + J_i alpha_i`, with recursive
  point accelerations. This is not the reviewed closed-form M/C/g.
- uses `mpmath.iv` for rigorous outward-rounded sin/cos and arithmetic at 256, 640 and
  1600 bits.
- converts endpoints back to exact Fractions for every comparison.
- substitutes the path velocity as `qdot = q' w`, `w^2 = x`, `qddot = q'' x + q' x'/2`.
  No sqrt is needed.
- has an exact-rational branch for trig-free cases (single link with g = 0; two-link
  with g = 0 and q2 = 0). It resolves edge-equal comparisons exactly.

`independent_checks/k4/lagrange_sympy.py` is a second derivation. It writes the
Euler-Lagrange equations from the CoM positions and energies with sympy, and the
reviewer's own expression walker evaluates them into `mpmath.iv`. It agrees with the
Kane oracle at 45 random states (3 models), with both enclosures narrower than 1e-50.
It also proves symbolically that tau is affine in qddot and homogeneous quadratic in qdot.
The sympy test is skipped if sympy is missing (CI does not install it). The Kane oracle
needs only mpmath, which CI installs.

Verdict semantics:
- **DISJOINT**: the oracle enclosure lies strictly outside the reviewed box. This is a
  proven counterexample.
- **CONTAINED**: the oracle enclosure lies inside the reviewed box. This proves the true
  value is inside the box.
- **AMBIGUOUS**: the enclosures overlap without either relation. Recorded; it occurred
  0 times.
- **POINT_CONSISTENT / EDGE_CONSISTENT / EXACT_CONTAINED**: special cases for point
  boxes, tight endpoints, and the exact trig-free branch respectively. These are counted
  separately.

## Obligation matrix

Line numbers refer to `src/curobo_certify/k4_reference.py` (R) and
`src/curobo_certify/k4_affine.py` (AF) at HEAD. Test names refer to
`tests/test_independent_k4_oracle.py`.

| # | Obligation | Verdict | Evidence |
|---|---|---|---|
| O1 | sin/cos Taylor + Lagrange remainder | PASS | R:85-95. The sum is over degree d = 2·28 (cos) or 2·28+1 (sin); the remainder is \|c\|^(d+1)/(d+1)!, valid because every derivative of sin/cos has \|·\| <= 1. For cos the bound is one order looser than necessary (degree 57 is also an even polynomial), which is valid but conservative. At \|c\| = 16 the remainder is about 1.06e-8. All arithmetic is exact `Fraction`, so alternating terms up to about 1e6 cause no cancellation error. The clamp to [-1,1] cannot invert the interval, because p-r <= true value <= 1. Tests: `TrigAudit.test_taylor_lagrange_remainder_at_threshold` (c = ±16, 16-1e-20, 15.5); `test_sin_cos_enclose_exact_range_random_including_large` (500 intervals × sin/cos, center in [-24,24], radius 0 to 7): 0 DISJOINT, 0 AMBIGUOUS. |
| O2 | Lipschitz radius widening | PASS | R:98-106: f([m-r, m+r]) ⊆ [f(m)-r, f(m)+r] ∩ [-1,1] by \|f'\| <= 1. The oracle checks the exact range: the hull of the endpoint values and ±1 at every stationary point whose rigorous pi-enclosure touches the interval. Test: `test_lipschitz_widening_across_stationary_points`, plus the 500-interval test. |
| O3 | Large-angle fallback (\|mid\| > 16 gives [-1,1]) | PASS (sound) / note | R:102-103. mid = 16 still uses Taylor; mid = 16+1e-30 falls back (`test_fallback_boundary_and_huge_angles`). Paths over [15,17], [-40,40] and [199.5,200.5] certify and are oracle-contained (`test_large_angles_and_fallback_region`). Coverage note N4: a constant angle of 20 rad can be neither certified nor witnessed. Result is UNKNOWN_DEPTH, which is fail-closed. |
| O4 | Interval multiplication sign cases | PASS | R:63-67 takes the min/max of the four endpoint products, which is exact for real multiplication. All 49 sign-class pairs (negative, positive, straddling, zero-touching, zero) were tested. The endpoints are attained, and 200 sampled products per pair lie inside. Test: `test_multiplication_all_sign_cases_exact_range`. Dependency note N6: `d2*d2` is a product rather than a square, which over-estimates but stays sound. |
| O5 | Interval division | PASS | R:71-75 raises on any divisor with lo <= 0 <= hi, including zero-touching and [0,0]. Otherwise it multiplies by [1/hi, 1/lo]. Test: `test_division_and_zero_cases`. Division is not used by the dynamics. |
| O6 | Power-to-Bernstein conversion | PASS | R:131-134, b_k = Σ_{i<=k} C(k,i)/C(n,i) p_i. The oracle checks the exact polynomial identity Σ b_k C(n,k) u^k (1-u)^(n-k) ≡ p for degrees 0-6, 30 random polynomials each. Test: `test_power_to_bernstein_identity`. |
| O7 | de Casteljau midpoint split | PASS | R:137-144: left = levels[i][0], right = levels[n-i][i]. Identity checks against p(u/2) and p((1+u)/2) for degrees 0-6. Test: `test_de_casteljau_split_identity`. compose_subinterval (R:151-157) equals the oracle's own shift (`test_compose_subinterval_matches_own_shift`). |
| O8 | dpoly derivative scaling by h | PASS | R:124-128 divides by h per order, so derivatives are global in s. Compared with the oracle's closed form for orders 0-3, 200 random h. Test: `test_dpoly_scaling_by_span_length`. Also exercised end to end with unequal knots (h = 1/16 to 15/16) in the campaign. |
| O9 | Two-link mass matrix | PASS | R:236-238 matches the oracle (absolute angles from +x; J_i are CoM inertias). Test: `test_two_link_ABG_vs_kane_oracle_random`: 150 random models with nonzero gravity, q2 != 0 and \|q\| up to about 15.5. All 900 A/B/G components are CONTAINED or POINT_CONSISTENT; none DISJOINT or AMBIGUOUS. |
| O10 | Coriolis/centrifugal sign | PASS | R:239-241: h = -m2 l1 c2 sin q2; A1 += h(2 q1'q2' + q2'^2); A2 -= h q1'^2. Hand value: tau2 = +m2 l1 c2 sin(1) when q1' = 1 (`test_coriolis_and_gravity_hand_values`), plus O9. |
| O11 | Gravity convention | PASS (convention documented by fixture) | Two-link (R:244-245): gravity vector (0,-g), angles from horizontal. Pendulum (R:208): tau = m c g sin q, i.e. angle from the downward vertical, and `inertia` is the pivot inertia. Both are confirmed by the oracle (`test_pendulum_ABG_vs_kane_oracle_random`, hand value for the outstretched arm). See N2 for the doc gap. |
| O12 | A/B/G affine decomposition and path substitution | PASS | Oracle identities G = F(q,0,0), B = (F(q,0,q') - G)/2, A = F(q,q',q'') - G, with F the independent Kane recursion (O9). Path-level torque uses qddot = q''x + q'x'/2 and qdot² = q'²x, compared at 251,730 oracle leaf-point checks (O19). |
| O13 | C2(q)/C1(x) regularity at knots | PASS | R:278-293 compares exact global derivatives at knots, using each span's own h. Test `test_regularity_at_knots_uses_global_derivatives` checks five cases: <br>• equal local but unequal global slope (h = 1/3 vs 2/3) is rejected;<br>• a genuinely C2 path with unequal knots is accepted;<br>• a q'' jump is rejected;<br>• an x' jump is rejected;<br>• an x'' jump is accepted, as specified;<br>• a gap between spans is rejected. |
| O14 | Asymmetric limits, closed boundary, endpoint sides | PASS | R:296-302 and R:355 (closed subset). `test_asymmetric_limits_lower_side_witness`: limits [4,9] certify; limits [5,9] are violated on the LOWER side, with a witness hi of 4.79 < 5. `test_torque_exactly_at_limit_and_strict_witness`: tau ≡ 1 against [-1,1] is CERTIFIED; against 1-1e-30 it is VIOLATED. Knot endpoints are covered from both adjacent spans; under C2/C1 the two sides agree. |
| O15 | Witness disjointness | PASS | R:357-367: violation requires v.lo > hi or v.hi < lo (strict), on a point enclosure at an exact rational u. A Bernstein hull alone never produces VIOLATED. Every VIOLATED_TORQUE witness in the campaign and tests is re-proven by the oracle (`verify_violation`): 580/580 in the campaign. Every VIOLATED_X witness has exactly x(u) < 0: 144/144. `test_interior_true_violation_witness` shows the subdivision finds an interior witness (u = 3/8 class) when u = 0, 1/2, 1 are feasible. |
| O16 | Near-boundary cases must not certify | PASS | `test_near_boundary_nondyadic_peak_is_never_certified`: peak at u = 1/3 with limit = peak - 1e-30 is never CERTIFIED (gives UNKNOWN). With limit = peak + 1e-30 it is never VIOLATED. `test_near_boundary_irrational_gravity_extremum`: limits at sin(1) ± 1e-30 (rigorous 400-bit bracket) behave the same way. |
| O17 | Negative x, and x = 0 at interior points | PASS (scope note N5) | R:355 requires the Bernstein hull of x to have lo >= 0 per leaf; R:358-361 gives VIOLATED_X only for an exact x(u) < 0. A 1e-12 negative pocket at non-dyadic 1/3 is never certified. A pocket at a dyadic point gives VIOLATED_X with a true negative witness. x = (u-1/2)^2 is CERTIFIED with SCOPE "...REQUIRES_SEPARATE_K3..."; x = (u-1/3)^2 never gives a false VIOLATED_X (`test_negative_x_cases`, `test_x_zero_at_interior_points`). |
| O18 | Nonlinear interval dependency | PASS | Two-link with q2 ∈ [-3,3] and large q': all leaf boxes are oracle-contained (`test_nonlinear_dependency_two_link_wide_q2`). Over-estimation only (N6). |
| O19 | Randomized dense-sampling falsification | PASS (0 counterexamples) | `independent_checks/k4/run_review.py`, 6 seeds × 250 cases, max_depth 8 / max_nodes 1500. Results are in `campaign_results.json`. See the campaign table below for counts. Every CERTIFIED leaf is sampled at both endpoints, the midpoint and 3 random rationals. A 65-point dense grid per span is mapped to its covering leaves, with an independent partition check. The unit test `test_randomized_rational_families_no_counterexample` runs a smaller seeded version on every test run. |
| O20 | Budget / fail-closed | PASS (cosmetic N8) | R:340-341, 350-351, 368-369, 375-376. With max_nodes 1..4 the result is UNKNOWN and nodes <= max_nodes. Negative or float budgets give INVALID_INPUT. A dof mismatch through the public `dof` field gives INVALID_INPUT. Empty spans give INVALID_PATH (`test_budget_fail_closed`). A bool budget is accepted as an int (cosmetic). |
| O21 | No float mixing into exact rationals | PASS (doc note N7) | An AST scan of both modules finds no float literals, no `float()`/`round()` calls, and only `comb/factorial/isfinite` from math (`test_no_float_literals_or_float_calls_in_reviewed_modules`). All leaf endpoints are `Fraction`. Float INPUTS are accepted and converted exactly to their binary value (0.1 != 1/10, `test_float_inputs_are_converted_exactly_not_decimally`). The pilot JSON uses strings. |
| O22 | Replay: omitted / overlapping / duplicate / foreign / bad-path leaves | PASS | R:385-401. Prefix-free path sets with Kraft sum exactly 1 are exactly the complete dyadic partitions, so the check is sufficient. Test `test_omitted_duplicate_overlapping_and_foreign_leaves` covers the attacks. Each forged result is rejected:<br>• omitted leaf;<br>• duplicate leaf;<br>• leaf plus its parent;<br>• span index 2 or -1;<br>• 'X' in a path;<br>• siblings merged into a forged root. |
| O23 | Replay: misordered leaves | PASS | Shuffled leaves are accepted, and this is correct: the certificate is a set. The oracle audit of the shuffled result passes (`test_misordered_leaves_are_accepted_and_still_sound`). |
| O24 | Replay: boxes too narrow | PASS | Shrinking any single endpoint by 1e-30 is rejected (12 cases, `test_too_narrow_boxes_rejected`). |
| O25 | Replay: altered model parameters / limits | PASS (with G3) | All 24 parameter alterations (each of 8 parameters × {3/2, 1/2, 11/10}) of a 16-leaf tight-limit certificate are rejected. Any acceptance would be oracle-audited against the ALTERED model, because replay re-proves for whatever model it is given (`test_altered_model_parameters`). Tighter limits are rejected. Replay against a pendulum is rejected. G3: the result object binds no model/limit/path hash. |
| O26 | Replay: fake witnesses | **FAIL (gap G1, not unsound)** | The code has no witness replay. `replay_torque_leaves` returns False for any non-CERTIFIED result (R:385) and ignores the `witness` of a CERTIFIED result. A forged `VIOLATED_TORQUE` result cannot be distinguished from a genuine one with repository code. Only the reviewer oracle rejects it (`test_status_limits_and_fake_witness`). Plan section D explicitly requires fake-witness detection. |
| O27 | Model parameter uncertainty | **UNKNOWN / NOT IMPLEMENTED (gap G2)** | Parameters are exact points. Interval parameters raise TypeError at construction (`test_model_parameter_uncertainty_not_representable`). Plan D lists "model parameter uncertainty" tests. They cannot be written against the current API. |
| O28 | k4_affine prepared boxes contain the true A,B,G | PASS | AF:46-56. At depth-4 boxes, oracle A/B/G at 6 points per leaf are all CONTAINED or POINT_CONSISTENT (`test_prepared_ABG_boxes_contain_oracle`). |
| O29 | k4_affine signed x' and alternative timings | PASS | AF:116-118 uses a signed interval for x'. Four timing profiles were tested, including ones where x' changes sign and is negative throughout. Every CERTIFIED result is oracle-audited and replays (`test_signed_xprime_alternative_timings`). In the campaign, all affine CERTIFIED results were audited with 0 counterexamples and 0 replay rejections. A negative-x pocket is never CERTIFIED. |
| O30 | k4_affine hash collisions / model mutation / forged caches | PASS (note N9) | AF:73-95 recomputes every box, so the hashes are not trust anchors and a collision cannot cause unsoundness. The following are all rejected (`test_model_mutation_and_forged_prepared`):<br>• model mutated by `object.__setattr__`;<br>• forged cache with the correct geometry digest but boxes from a different geometry;<br>• duplicate-plus-missing leaf;<br>• wrong span index;<br>• bad path character;<br>• wrong depth;<br>• narrowed box.<br>The geometry digest excludes x by design and includes the knots. It is representation-sensitive: a trailing zero coefficient changes the hash, which fails closed (`test_digest_semantics`). |
| O31 | Reference replay is not claimed as independent dynamics | PASS | R:9-11, R:380-383 and plan B say this explicitly. Replay shares `_torque_bounds`, `bernstein`, `dpoly` and the trig code with the producer. This review supplies the independent dynamics comparison. |
| O32 | Claim boundaries / scope strings | PASS | SCOPE (R:330) and the pilot `not_claimed` list. The plan states no Panda claim. Pilot totals were reproduced: 8 CERTIFIED / 6 VIOLATED_TORQUE / 1 UNKNOWN_DEPTH over depths 0/4/8. The existing 18 K4 tests pass. |

Campaign summary (O19, `campaign_results.json`):

| Quantity | Value |
|---|---|
| Cases (6 seeds × 250) | 1,500 (pendulum and two-link, 1-3 C2 spans, random rational params, gravity 0 / 9.81 / random) |
| Two-link cases with q2 != 0 and gravity != 0 | 686 |
| Cases with negative x' somewhere | 1,081 |
| certify_torque status | 550 CERTIFIED, 580 VIOLATED_TORQUE, 144 VIOLATED_X, 226 UNKNOWN_DEPTH |
| Oracle leaf-point torque checks (incl. affine) | 251,730 (affine alone 40,596) |
| Resolved exactly by trig-free rational oracle | 251 |
| DISJOINT (counterexamples) | **0** |
| AMBIGUOUS | 0 |
| VIOLATED_TORQUE witnesses confirmed by oracle | 580 / 580 |
| VIOLATED_X witnesses with exact x(u) < 0 | 144 / 144 |
| CERTIFIED rejected by replay | 0 / 550 |
| k4_affine (alternative x scaled by a random factor) | 150 CERTIFIED (all oracle-audited, all replay True), 408 UNKNOWN_PREPARED_ENCLOSURE |
| Wall time | 272 s CPU |

## Counterexamples

**None.** No DISJOINT verdict occurred in any test or campaign case: there is no
CERTIFIED leaf box missing the oracle torque, no CERTIFIED result with oracle torque
outside the limits, no uncovered point, no partition gap, and no unconfirmed witness.
The AMBIGUOUS count is 0.

## Gaps and findings (severity and recommended fix)

| ID | Finding | Severity | Recommendation |
|---|---|---|---|
| G1 | No witness replay. A forged VIOLATED_TORQUE/VIOLATED_X result is undetectable by repository tooling (R:385). | Fail-open for NEGATIVE claims (an infeasibility claim cannot be audited); not unsound for CERTIFIED. Blocks the plan-D obligation. | Add `replay_witness(spans, model, limits, result)`. It should check that span is in range and u is an exact rational in [0,1], recompute `_point_torque`, and require strict disjointness on the stated joint (or x(u) < 0 exactly). Add tests for a forged u, joint or torque. |
| G2 | Model-parameter uncertainty is not representable (points only). | Scope. | Accept `I` parameters in `Pendulum`/`PlanarTwoLink`. The existing interval algebra already supports this once `rat()` is bypassed for `I`. Add tests whose boxes must contain the oracle for every parameter corner and for random interior points. |
| G3 | `K4Result` carries no SHA of spans/model/limits/source. Replay re-proves for whatever inputs it is given. | Provenance/doc. AGENTS.md requires exact hashes; not unsound. | Embed `geometry/x/model/limits/source` digests in the result and have replay compare them. |
| N1 | Stored leaf boxes are not checked against the limits; only the recomputed boxes are (R:414-417). A stored box of [-1000,1000] is accepted (`test_stored_boxes_wider_than_limits_are_accepted_harmlessly`). | Cosmetic. | Also require `leaf.torque[j] ⊆ lim[j]`, or document that stored boxes are informational. |
| N2 | `Pendulum.inertia` is the PIVOT inertia, which is undocumented. Values with inertia < m c² (physically impossible) are accepted, e.g. `Pendulum(0,1,1,1)`. | Doc/scope (fixture-only). | Document the convention, or parametrize by CoM inertia; enforce inertia >= m c². |
| N3 | The cos remainder uses degree d+1 although the next nonzero term is d+2. | None (conservative). | Optional tightening. |
| N4 | \|mid\| > 16 gives [-1,1] with no range reduction. Constant large angles cannot be certified or witnessed (coverage loss). | Coverage, fail-closed. | Rigorous range reduction with an interval pi enclosure (exact rational bounds of pi). |
| N5 | x = 0 at interior points, or x ≡ 0, certifies torque (= G-dominated) for a path that may not be traversable. | Scope (correctly flagged in SCOPE). | Keep the K3 integrability conjunction mandatory before any combined claim. |
| N6 | Interval dependency: `d*d` and repeated `cos(q2)` occurrences are treated as independent. | Tightness only. | Add an `sqr()` op and centered forms for later RNEA work. |
| N7 | Float inputs are silently accepted as exact binary values. | Doc. | Optionally reject floats in proof inputs, or document the behaviour. |
| N8 | `max_depth=True` is accepted. `replay_torque_leaves` calls `_regularity` outside its try (a generator for `spans` raises TypeError instead of returning False). k4_affine does not catch OverflowError. | Cosmetic, fail-closed. | Type-check budgets; move the regularity call inside the try. |
| N9 | The geometry digest is repr-based and representation-sensitive (trailing zeros). | Fail-closed spurious INVALID. | Canonicalize (strip trailing zeros) before hashing. |

## Out of scope / not reviewed

- Concurrent activity: during the review a separate actor committed `9f33208` ("K4 P0: frozen
  cuRobo Panda dynamics contract, interval RNEA, ..."), adding `k4_panda_model.py`,
  `test_k4_interval_rnea.py` and others. `git diff d04a9ce 9f33208` touches NONE of the
  reviewed files, so the hashes and verdicts above apply unchanged at 9f33208. Those Panda
  files, the uncommitted `k4_panda_model.py` modification, and the untracked
  `k4_path.py` / `k4_verify.py` / `k4_model_contract_v2.py` were NOT reviewed.

- `src/curobo_certify/k4_path.py` appeared as an UNTRACKED file in the working tree during
  this review (mtime 2026-11-17 22:03, SHA-256
  `5670197f048e61e3e7cd8a3c6ea09150e1750b2210aa79131da6902efb6da5d8`). It is not part of
  HEAD, was not created by the reviewer, and was NOT reviewed. The same holds for any Panda
  RNEA / `k4_panda_model` code. These require their own P2 review.
- Plan obligations for P0/P2-P5 (Panda model contract, interval RNEA, native FP32, latency).

## SHA-256 of reviewed files (at HEAD d04a9ceee74cafae886765d66bfe0b3c780fee65)

```
4c41d40612cbc2297930f1d5e8d71c41b081a69543988d21be087ac0a33c46ba  src/curobo_certify/k4_reference.py
a3448cd6f2fa7fe4d96b8d4ad0a21be95821f20db03ffd4ee4b1918c85f309f3  src/curobo_certify/k4_affine.py
19e91a642005e98b8756a05bada7c6da39a9cec1427979d63817dfa2ffa2d217  tests/test_k4_reference.py
d3352a2e06a2634f73708befd23400014b5d09a522f5eeeb9170c677f42e6c7e  tests/test_k4_affine.py
aa99cacee894708785db38f82004433c46d345ab95eb8b0ca01bb1aa8ba66846  scripts/run_k4_reference_pilot.py
f3fe64e1caa7ce6cfa6bd31e4fda174a707394e553a35438c83152ffe1c7b4d2  docs/K4_CERTIFIED_TORQUE_RESEARCH_PLAN_20261117.md
f214d0aef8adc483a7d973278c889371efc4d61923604ea186f492b27f4b0cf6  AGENTS.md
```

Reviewer-created files (SHA-256):

```
618edcf4fcdee71f4d9b33f30580b4e4c74f2a77c2c9bab722968fe28df524a4  independent_checks/k4/__init__.py
97a4bec9dfc04a652818088a6e9c63e3263b9cd284a4e7990fe754f14aa5cd53  independent_checks/k4/oracle.py
dfe2a1f531d64541a89fa82def33b9325442735f9cb3e8ea005713fd4dad0300  independent_checks/k4/lagrange_sympy.py
cef3ed32d4bc951a1de338e77b512ce9bfcc901bd1eb3990309a57effbf48be6  independent_checks/k4/campaign.py
7213367a9809bfcd9a3b7729b755f654e4d9b1a91a5215877f12922dc0b1cf06  independent_checks/k4/run_review.py
e7b27d7bdeaaedfed003455d4ab434b3a06194295c2480e5cf9bbc3959feb9f1  tests/test_independent_k4_oracle.py
ab44fa451ded3058241ca54cd8d5321f316afd8792559bd04ebe9078a5437354  docs/reports/k4_p1_independent_review/campaign_results.json
```

## Reproduce

```
PYTHONPATH=src:. python -m unittest tests.test_independent_k4_oracle -v
PYTHONPATH=src:. python independent_checks/k4/run_review.py --out /tmp/k4_p1_campaign.json \
    --seeds 101 202 303 404 505 606 --cases 250 --max-depth 8 --max-nodes 1500
python -m unittest discover -s tests -p 'test_*.py'
```

Environment: Python 3.12 (anaconda), mpmath 1.3.0, sympy 1.12 (optional), CPU only.

## Results at review time

- `tests.test_independent_k4_oracle`: 45 tests, OK (about 20 s; the sympy test runs when sympy is present).
- Full suite (`PYTHONPATH=src:. python -m unittest discover -s tests -p 'test_*.py'`): 310 tests, OK (skipped=6), 89 s.
- The existing K4 tests (`tests.test_k4_reference`, `tests.test_k4_affine`): 18 tests, OK.
- The memo hash covers itself only implicitly; the campaign JSON hash is listed above.

# Compiled complete K3 certifier and independent verifier

Date 2026-11-16. Frozen sources: producer `src/curobo_certify/native/k3_certifier.cpp`
SHA256 `0a762f9c16c79c189d4a6621fad311d72d0991e55468c9964a468a0531a68ae7`, verifier
`src/curobo_certify/native/k3_verifier.cpp` SHA256 `6757df8f1d37cc26975c7b02b5c83411bfcf4a6b7e7a5acc7afb32fb4e687140`,
export map `src/curobo_certify/native/exports.map`. Build: GCC 9.4.0
`-std=c++17 -O3 -shared -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror`
plus `-Wl,--version-script=exports.map` (see `compiled_k3.build`).

## Contract

Input: float32 effective geometry `[C][D]` (uniform cardinal cubic B-spline), 32 float32 timing
controls (clamped cubic squared-speed spline, knots k/29), requested start/goal, per-joint limits
(binary64), optional per-common-span tightening `[ns][D][3]` (native FP32 error bound).
CERTIFIED (producer) **and** PASS (verifier) means: for the exact serialized model, on every common
span, `-x<0` strictly in the trajectory interior (span ends that are interior knots included),
`q'^2 x <= v^2`, `|q''x + q'x'/2| <= a`, `x (q'''x + 1.5q''x' + 0.5q'x'')^2 <= j^2` (with limits reduced
by the tightening), plus exact endpoint position equality, strict rest (zero physical velocity and
acceleration) and local finite traversal time at both endpoints. Knot continuity (q C², x C²) is a
structural property of the B-spline construction used by both programs. Torque/torque rate:
UNSUPPORTED_DYNAMICS. With the native tightening, the claim transfers to native reported values
(see `docs/NATIVE_LINKAGE_20261116.md`).

Statuses: CERTIFIED, VIOLATED (rigorous interval witness: point enclosure strictly > 0, or ≥ 0 at an
interior point for positivity), UNKNOWN_DEPTH, UNKNOWN_NODES, UNKNOWN_UNIFORM, UNKNOWN_BOUNDARY,
VIOLATED_BOUNDARY, INVALID_INPUT (NaN/Inf, bad limits/dims), NUMERICAL_ERROR (overflow, rounding
mode not nearest, FTZ/DAZ). Budgets never yield CERTIFIED or VIOLATED.

## Producer vs verifier independence

| | producer | verifier |
| --- | --- | --- |
| residual construction | power basis, degree-exact Horner composition, power→Bernstein | B-spline blossoming (de Boor, distinct parameters) directly to Bernstein on each common span, Bernstein derivative/product/elevation |
| rounding | `nextafter`-style outward (inline bit successor) | relative widening `|r|·2^-52 + 2^-1074`, error-free exactness checks |
| partition | sort+unique of k/S ∪ m/29 | two-pointer integer merge |
| trusted from producer | — | leaf (span, family, joint, depth, path bits) only; status flags, uppers and coefficients ignored |
| also checks | | SHA-256 of canonical input bytes, prefix-free dyadic cover with exact measure 1, duplicates/overlaps/unknown ids, endpoint obligations with exact integers |

## Validation

- `tests/test_compiled_k3.py` (12 tests): tamper detection (dropped/duplicate/overlapping/moved
  leaves, bits beyond depth, changed limits/geometry vs digest), forged certificate for a violating
  path, boundary obligations, interior zero speed, NaN/Inf/overflow, budgets and ablations,
  rounding-mode fail-closed, inline `nextafter` vs libm on 40k adversarial doubles, exported-symbol
  table, compiled native bound ≥ Python bound, 150 seeded differential cases.
- `scripts/k3_differential_campaign.py`: 10,000 seeded adversarial cases each under `-O3`, `-O0`,
  `-O3 -march=native` versus the exact Fraction reference: **0 breaches** (false certificate, false
  witness, verifier rejection); 1,368 reference-certified, 1,229 compiled-certified-and-verified
  (the remainder UNKNOWN by the smaller budget depth 8 / nodes 4096 vs reference 14 / 200k).
- `tests/test_native_error_bound.py`, `tests/test_model_core_crosscheck.py`.

## Bugs found during validation (all fail-closed)

1. Composition padded a zero degree → producer hulls at degree 6/10/14; verifier (true degree)
   rejected 20/40 certificates. Fixed: degree-exact composition.
2. Outward primitives widened exact zeros (e.g. x(0)=0 with integrable start) → never certified.
   Fixed with error-free exact point operations in both programs.
3. **Symbol interposition:** helper functions accidentally given C linkage were exported as
   `add/mul/div`; inside Python, libc's `div(int,int)` was bound instead, so the native bound silently
   became ~2^-126 (unsound if used). Same library linked from C++ was correct. Fixed with hidden
   visibility + linker version script; regression test checks the dynamic symbol table.

## Remaining assumptions

IEEE binary64 round-to-nearest without FTZ/DAZ (checked at runtime), compiler honours
`-fno-fast-math -ffp-contract=off`, x86-64 little-endian. Finite tests are not a proof of the
compiled binary; no formal verification or MPFR oracle (headers unavailable) was used.

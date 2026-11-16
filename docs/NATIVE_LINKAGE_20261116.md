# Native FP32 linkage: rigorous same-progress bound and time-law diagnostic

Date: 2026-11-16. Branch `exp/interactive-bernstein-proof-lab-20261113`. Upstream pin
`NVlabs/curobo@c301cb00a87012c2808b1a60649c4065c9b3c95a`. Hardware: RTX 3090 (GPU harness),
AMD Ryzen 9 5900X (CPU bounds and exact certificates).

## Outcome

| Claim | Status |
| --- | --- |
| NATIVE_REPORTED_K3: for every float32 progress `s` in [0, 1+4u] (hence every finite requested time), the native kernel's reported velocity/acceleration/jerk lie within the original physical limits | **CERTIFIED 16/16** frozen Panda outputs, *conditional on the arithmetic assumptions A1–A5 below* |
| Same-progress native-vs-model error | rigorously bounded per common span and joint (forward error analysis) |
| Time-law consistency: native `s_n(t)` equals exact inverse `s*(t)` of `t(s)=∫ds/√x` | **not proven** → `BLOCKED_NATIVE_LINKAGE` for complete native equivalence; diagnostic max error 4 float32 ulps of `s` |
| Registered 4e-4 scaled-relative jerk parity gate | still FAILS (unchanged threshold); failure fully explained by the two effects above |

The native-reported certificate does **not** depend on the time-law claim: the native kernel computes
every reported value as a deterministic function `G(s)` of the final progress, and `progress_at_time`
clamps `s` to `[left,right] ⊂ [0, 1+4u]`. Bounding `G` over all `s` covers any `t`.
Not claimed: that the native position stream's time derivatives equal the reported v/a/j,
torque/torque rate, collision, controller tracking, failed or stationary requests (those report NaN/0).

## Native computation (pinned source)

`sample_with_progress_kernel` → `progress_at_time` (Newton on a 511-interval FP32 time table) →
`timing_state` (FP32 table `CubicTimingBasis(32).polynomial`, Horner in `local = s - span/29`) and
`sample_joint` (`BSplineBasisPolynomial<3>`, centered controls, division by `knot_dt^k`, `knot_dt = 1/spans`).
NVRTC options (pinned `kernel_config.py`/`kernel_cache.py`): `--ftz=true --fmad=true --prec-div=false --prec-sqrt=false -O3`.

Discrepancy sources identified:
1. **Approximate division** (`prec-div=false`): `knot_dt`, all `/6.0f` basis divisions, `span/29`.
2. **FMA contraction** anywhere `a*b+c` occurs, including `s*spans - segment` (so the geometry may be
   evaluated slightly outside its segment) — the 1,176 trace replays already showed FMA order matters.
3. **FP32 timing table** is a float32 rounding of SciPy binary64 basis coefficients: worst relative error
   3.4e-8 on nonzero entries, **and 19 used entries nonzero (≤ 4.3e-14) where the exact coefficient is 0**.
4. **Knot-side / local coordinate offsets**: geometry point `(k+t)/S` and timing point `m/29 + local`
   differ from `s` by ≤ 6u and ≤ 5.04u; `s` may exceed 1 by ≤ 4u (approximate `(cell+1)/intervals`).
5. **Cancellation**: centered-control differences and the three jerk terms (e.g. −40.96 + 22.35 − 3.83).
6. **Time-to-progress inversion**: ≤ 4 ulp progress differences from the exact time law (diagnostic).

## Arithmetic assumptions (A1–A5)

A1 float32 add/sub/mul/fma correctly rounded; flush-to-zero absolute error < 2^-126.
A2 FMA contraction optional anywhere; every product and every sum is charged a rounding (dominates both).
A3 approximate division ≤ 2 ulp (NVIDIA documented); A4 `sqrtf` ≤ 3 ulp (documented 1 ulp for cc≥5.2);
A5 binary implements source semantics otherwise; captured float32 inputs; no subnormal inputs.
Sensitivity: with 4-ulp division and 4-ulp sqrt the result is still 16/16.

## Bound construction (`src/curobo_certify/native_error_bound.py`)

Each native intermediate is an `ET` = (outward interval of the exact model value over a parameter box,
upper bound on |native − exact|), propagated through the *native expression tree* with binary64 upward
rounding. Geometry: exact reference is the same segment polynomial at the clamped native point plus a
Lipschitz term (`sup|Q_{i+1}|·δ`, δ=1e-6 ≥ 11.04u). Timing: exact table difference charged per entry;
reference shift from `s_t` to `s_g` by Lipschitz plus a C² cross-knot term `|ΔX'''|δ^{3-i}/(3-i)!`.
Square root: `min(√e, e/√x_min)`. Bounds are taken per common span and joint over 16 sub-pieces.
The exact model is then certified with exact Bernstein subdivision against `limit − E_span,joint`
(`certify_native_reported`), including all exact boundary/knot/strict-rest obligations.

Observed bound sizes over 16 paths (relative to original limits): velocity ≤ 1.25e-5,
acceleration ≤ 4.6e-4, jerk ≤ 6.6e-3 (largest near the start where x is small and the sqrt term dominates).

## Falsification evidence (not proof)

`scripts/native_same_s_falsify.py` compiles a harness (`src/curobo_certify/native/same_s_harness.cuh`)
appended to the *unmodified* upstream headers with upstream's own NVRTC path/options:
- production reproduction: harness at production requested times reproduces saved S8193 and
  event-aware S32768 outputs **bit for bit on all 16 paths** (0 mismatches, ~1.15 M values/path);
- 1,176/1,176 captured device jerk values reproduced bitwise at captured progress;
- same-progress sweep per path: 2,097,152 uniform + 1,048,576 random + ±64-ulp neighbourhoods of all
  geometry/timing knots + 8,192 endpoint-logspaced float32 `s`, ×7 joints ×3 families:
  **0 exceedances**, max |native−model| / bound = 0.121.
CPU emulations (`tests/test_native_error_bound.py`) of no-FMA / FMA / ±2-ulp-division variants also stay
within the bound.

## Identical requested physical times (`scripts/native_time_consistency.py`)

Exact time law `T(s)` by composite 8-point Gauss–Legendre (64 cells/span, 32-cell cross-check),
`s*(t)` by safeguarded Newton; native `s_n(t)` from the bitwise-reproducing harness at the 8,193 production
times. All 16 paths: |s_n − s*| ≤ 1.9e-7 (≤ 4 float32 ulps); native vs model duration within 9.4e-8 relative.
Max jerk |native(t) − model(s*(t))| 0.0034–0.0072, decomposed into same-progress part 0.0018–0.0039
(rigorously bounded) and inversion-induced part 0.0033–0.0086. This is a HIGH_PRECISION_NUMERICAL_DIAGNOSTIC.

## Commands

```bash
python scripts/certify_native_reported.py --npz runs/native_pilot_resolved/pilot-panda-0*.npz \
  runs/native_panda_exploratory_seed20261111/*.npz --out runs/native_linkage/native_reported_16paths.json
python scripts/certify_native_reported.py ... --div-ulps 4 --sqrt-ulps 4 --out runs/native_linkage/native_reported_16paths_4ulp_sensitivity.json
CUDA_VISIBLE_DEVICES=1 conda run -n curobo-timeopt-c301cb python scripts/native_same_s_falsify.py \
  --npz <same 16> --dense 2097152 --random 1048576 --out runs/native_linkage/same_s_falsify_16paths.json
CUDA_VISIBLE_DEVICES=1 conda run -n curobo-timeopt-c301cb python scripts/native_time_consistency.py \
  --npz <same 16> --out runs/native_linkage/time_consistency_16paths.json
```

Fixture: `fixtures/timeopt_timing_polynomial_c301cb.npy` (float32 bytes SHA256
`0e238103f51c51b9269dca507607b419965a685fb6150b1af027fbd36e32fc4a`), generated from the pinned class.

## Remaining blockers

- A1–A5 are vendor/compiler specifications, not verified hardware proofs; a SASS-level audit of the
  compiled kernel (FMA placement, division instruction) was not performed.
- Time-law consistency has no rigorous continuous bound (a posteriori interval quadrature is feasible
  future work); complete native equivalence remains `BLOCKED_NATIVE_LINKAGE`.
- Bounds are computed in Python (~1.9 s/path); see the compiled certifier for the port.

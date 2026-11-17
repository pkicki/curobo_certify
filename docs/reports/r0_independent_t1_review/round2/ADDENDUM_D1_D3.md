# Addendum: review of the D1–D3 working-tree diff (before freeze)

**Reviewer:** the same isolated Claude subagent as rounds 1 and 2.
**Scope:** the uncommitted `git diff` on HEAD c1797b4, which touches only the two files below.
**Verdict:** confirmed. The round-2 conclusions are unchanged, and **R0 round 2 PASS stands.**

## D3: window stop in `scripts/native_time_audit_v2.py`

The new code is `f = float(np.float32(T)); if f < T: f = nextafter_f32(f, +inf)`. The comparison is now between two Python floats, so it is exact in binary64.

**Argument.** `np.float32(T)` is the float32 nearest to T. If it is ≥ T, it is the smallest float32 ≥ T: any smaller float32 would be closer to T while still being ≥ T, which contradicts nearest rounding. If it is < T, then the next float32 up is the smallest float32 ≥ T, because T lies between those two adjacent float32 values.

This holds for every finite positive T ≤ FLT_MAX, including the subnormal range. For T > FLT_MAX the result is +Inf, which is the IEEE answer and cannot occur for durations.

**Check.** `independent_checks/t1/round2_d3_stop_check.py` compares the code against an exact bit-pattern bisection with Fraction comparisons, under numpy 2.1.3. It covered 260,004 values: log-uniform values over [1e-45, 3e38], exact float32 values, their binary64 neighbours on both sides, and edge cases. Result: **0 mismatches**.

The old code was already sound (round-2 memo §1). The new code now also matches its comment.

## D2: `REPORTED_NONFINITE` label

Integrity failure takes precedence (`NOT_ESTABLISHED`). Non-finite outputs are labelled before the within-limits check. The label can only become more specific; no acceptance path is widened.

## D1 and the window note in `docs/T1_V2_CONTRACT_20261117.md`

The new wording is accurate. It matches probe Q3, which found −0 bitwise identical to +0. The window note correctly limits the L·ε statement to the exact-model values. The verdict section reflects the round-1 and round-2 memos accurately.

## SHA-256 of the reviewed working-tree files

| File | SHA-256 |
|---|---|
| `scripts/native_time_audit_v2.py` | `ce380518921e0df7ed9689ced36ca71be886a6fd133069c912ade5b3f332c705` |
| `docs/T1_V2_CONTRACT_20261117.md` | `e761e1ec9d48e8e8cd0b4fc570bd206eaa9a668858d756cbc0042f3634bbdbc9` |

The freeze must use these exact hashes. The other reviewed hashes are in REVIEW_MEMO_ROUND2.md §5 and are unchanged.

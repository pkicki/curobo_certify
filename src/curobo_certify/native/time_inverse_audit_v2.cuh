// Exhaustive native time->progress audit, version 2 (curobo_certify).
// Compiled after the UNMODIFIED pinned headers timeopt_common.cuh and
// timeopt_path_kernel.cuh and same_s_harness.cuh with cuRobo's own NVRTC options.
//
// Changes from v1 (time_inverse_audit.cuh, kept unchanged for the historical result):
//  * fail closed: a non-finite / negative / >1 native progress, an empty bracket
//    (lo > hi) or a native-vs-reference progress bit mismatch is COUNTED and forces the
//    maximum error key to +Inf (v1 silently mapped NaN progress to zero error);
//  * float32 -> binary64 conversion of the requested time and of s is done from the
//    bit pattern (exact for subnormals, independent of any FTZ compile flag);
//  * per-thread work is counted (block __syncthreads_count) so the host can prove that
//    every enumerated bit pattern was processed exactly once;
//  * maxima are warp-reduced before the 64-bit atomicMax (same monotone key packing);
//  * SOUNDNESS FIX (R0 U1): the error bound uses the far bracket end max(s-lo, hi-s).
// The reported-value audit is NOT done here any more: the driver runs the production
// sample_with_progress_kernel itself (upstream launcher) over every float32 time.
#pragma once
namespace curobo_certify_harness_v2 {
using namespace curobo::trajectory::timeopt;

// Exact binary64 value of a nonnegative finite float32 bit pattern (b < 0x7f800000).
__device__ __forceinline__ double exact_f64(unsigned int b) {
  const unsigned int e = (b >> 23) & 0xffu, m = b & 0x7fffffu;
  if (e == 0u) return double(m) * __hiloint2double(0x36A00000, 0);  // m * 2^-149, exact
  const unsigned long long d = ((unsigned long long)(e + 896u) << 52) | ((unsigned long long)m << 29);
  return __longlong_as_double((long long)d);
}

__device__ __forceinline__ long long largest_le(const double *a, long long n1, double t) {
  long long lo = -1, hi = n1 - 1;  // largest idx in [0, n1-1] with a[idx] <= t, or -1
  while (lo < hi) {
    const long long mid = (lo + hi + 1) >> 1;
    if (a[mid] <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}
__device__ __forceinline__ long long largest_lt(const double *a, long long n1, double t) {
  long long lo = -1, hi = n1 - 1;  // largest idx with a[idx] < t, or -1
  while (lo < hi) {
    const long long mid = (lo + hi + 1) >> 1;
    if (a[mid] < t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

// Monotone 64-bit key: upward-rounded float32 bits of a nonnegative double, then the time bits.
// The FTZ compile flag could flush a positive result below 2^-126 to +0 (an under-report);
// such a result is replaced by the smallest normal float32 (an over-report, sound).
__device__ __forceinline__ unsigned long long key_ru(double e, unsigned int bits) {
  unsigned int f = __float_as_uint(__double2float_ru(e));
  if (e > 0.0 && f < 0x00800000u) f = 0x00800000u;
  if (!(e >= 0.0)) f = 0x7f800000u;  // NaN/negative cannot occur here; fail closed anyway
  return (static_cast<unsigned long long>(f) << 32) | bits;
}

__device__ __forceinline__ unsigned long long warp_max(unsigned long long v) {
  for (int o = 16; o > 0; o >>= 1) {
    const unsigned long long w = __shfl_xor_sync(0xffffffffu, v, o);
    v = w > v ? w : v;
  }
  return v;
}

// counters[0] processed threads, [1] bad progress (non-finite, <0 or >1),
// [2] empty bracket, [3] progress bit mismatch vs s_ref, [4] bracket lookup failure.
__global__ void inverse_audit_v2(const float *polynomial, const float *timing, const float *nodes,
                                 const float *weights, const float *time, int intervals,
                                 unsigned int bits_start, unsigned int count, const float *s_ref,
                                 const double *T_lo, const double *T_hi, const double *fmin_tab,
                                 const double *fmax_tab, int log2n,
                                 unsigned long long *max_any, unsigned long long *max_ahead,
                                 unsigned long long *max_behind, unsigned long long *counters) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = i < count;
  const unsigned long long INFKEY = 0x7f800000ULL << 32;
  unsigned long long k_any = 0, k_ahead = 0, k_behind = 0;
  int bad_s = 0, bad_br = 0, mism = 0, bad_lookup = 0;
  if (valid) {
    const unsigned int bits = bits_start + i;
    const float t = __uint_as_float(bits);
    float x[3];
    const float s = progress_at_time(polynomial, timing, nodes, weights, time, 0, intervals, t, x);
    unsigned int sb = __float_as_uint(s);
    if (s_ref != nullptr && sb != __float_as_uint(s_ref[i])) mism = 1;
    if (sb == 0x80000000u) sb = 0u;  // -0 == +0
    if ((sb >> 31) != 0u || sb > 0x3f800000u) bad_s = 1;  // negative, >1, Inf or NaN
    const long long N = 1LL << log2n;
    const double inv = 1.0 / double(N);  // exact power of two
    const double td = exact_f64(bits);
    double lo = 0.0, hi = 0.0;
    if (td > 0.0) {
      const long long k = largest_le(T_hi, N + 1, td);
      const long long j = largest_lt(T_lo, N + 1, td);
      if (k < 0 || j < 0) bad_lookup = 1;  // impossible when T_hi[0] = T_lo[0] = 0 < td
      if (k >= N) {
        lo = 1.0;
      } else if (k >= 0) {
        const double sk = double(k) * inv, sk1 = double(k + 1) * inv;
        double v = __dadd_rd(sk, __ddiv_rd(__dsub_rd(td, T_hi[k]), fmax_tab[k]));
        v = v > sk ? v : sk;
        lo = v < sk1 ? v : sk1;
      }
      if (j >= N) {
        hi = 1.0;
      } else if (j >= 0) {
        const double sj = double(j) * inv, sj1 = double(j + 1) * inv;
        double v = __dadd_ru(sj, __ddiv_ru(__dsub_ru(td, T_lo[j]), fmin_tab[j]));
        hi = v < sj1 ? v : sj1;
        if (hi > 1.0) hi = 1.0;
      } else {
        hi = 1.0;  // fail-safe widest upper end (also counted as bad_lookup)
      }
    }
    if (!(lo <= hi)) bad_br = 1;
    if (bad_s || bad_br || bad_lookup || mism) {
      k_any = k_ahead = k_behind = INFKEY | bits;
    } else {
      // s* lies somewhere in [lo, hi], so the SOUND upper bounds use the FAR bracket end:
      //   s_native - s* <= s_native - lo,   s* - s_native <= hi - s_native.
      // (v1 and the first v2 draft used the near end, max(s-hi, lo-s, 0): a LOWER bound on
      //  |s_native - s*| -- independent review R0 finding U1.)
      const double sd = exact_f64(sb);
      const double ahead = __dsub_ru(sd, lo);   // upper bound on s_native - s*
      const double behind = __dsub_ru(hi, sd);  // upper bound on s* - s_native
      const double e_ahead = ahead > 0 ? ahead : 0.0, e_behind = behind > 0 ? behind : 0.0;
      const double e = e_ahead > e_behind ? e_ahead : e_behind;
      k_any = key_ru(e, bits);
      k_ahead = key_ru(e_ahead, bits);
      k_behind = key_ru(e_behind, bits);
    }
  }
  // Every thread of the block reaches the barriers (no early return).
  const int n_valid = __syncthreads_count(valid);
  const int n_bad_s = __syncthreads_count(bad_s);
  const int n_bad_br = __syncthreads_count(bad_br);
  const int n_mism = __syncthreads_count(mism);
  const int n_lookup = __syncthreads_count(bad_lookup);
  if (threadIdx.x == 0) {
    atomicAdd(&counters[0], (unsigned long long)n_valid);
    if (n_bad_s) atomicAdd(&counters[1], (unsigned long long)n_bad_s);
    if (n_bad_br) atomicAdd(&counters[2], (unsigned long long)n_bad_br);
    if (n_mism) atomicAdd(&counters[3], (unsigned long long)n_mism);
    if (n_lookup) atomicAdd(&counters[4], (unsigned long long)n_lookup);
  }
  k_any = warp_max(k_any);
  k_ahead = warp_max(k_ahead);
  k_behind = warp_max(k_behind);
  if ((threadIdx.x & 31) == 0) {
    atomicMax(max_any, k_any);
    atomicMax(max_ahead, k_ahead);
    atomicMax(max_behind, k_behind);
  }
}

// Harness progress + mirrored tail at a chunk of consecutive float32 times (bits_start + i),
// written in the production output layout [4][count][dof] (stride count*dof) so it can be
// compared bitwise with the production sample_with_progress_kernel on the same times.
__global__ void progress_chunk_v2(const float *controls, const float *start, const float *goal,
                                  const float *timing, const float *polynomial, const float *nodes,
                                  const float *weights, const float *time, unsigned int bits_start,
                                  unsigned int count, float *times_out, float *out, float *progress,
                                  int dof, int knots, int intervals, int start_offset) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float t = __uint_as_float(bits_start + i);
  times_out[i] = t;
  float x[3];
  const float s = progress_at_time(polynomial, timing, nodes, weights, time, 0, intervals, t, x);
  progress[i] = s;
  curobo_certify_harness::tail_values(controls, start, goal, s, x, dof, knots, start_offset, out,
                                      (int)count * dof, (int)i);
}
}  // namespace curobo_certify_harness_v2

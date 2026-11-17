// R0 reviewer's corrected variant of curobo_certify_harness::inverse_audit (review artifact only).
// Differences from src/curobo_certify/native/time_inverse_audit.cuh:
//   * the certified error is the distance to the FAR end of the bracket,
//       e = max(s - lo, hi - s)  (upward rounded),
//     which is an upper bound of |s - s*| for every s* in [lo, hi]; the reviewed kernel uses
//     max(s - hi, lo - s, 0), a LOWER bound of the error when s is outside the bracket and 0
//     when it is inside;
//   * the legacy (reviewed) quantity is reduced as well, for a like-for-like comparison;
//   * fail-closed counters: processed threads, empty brackets (lo > hi), non-finite s or e.
// Bracket arithmetic is otherwise byte-for-byte the reviewed one.
#pragma once
namespace curobo_certify_review {
using namespace curobo::trajectory::timeopt;

__device__ __forceinline__ long long r_largest_le(const double *a, long long n1, double t) {
  long long lo = -1, hi = n1 - 1;
  while (lo < hi) { const long long mid = (lo + hi + 1) >> 1; if (a[mid] <= t) lo = mid; else hi = mid - 1; }
  return lo;
}
__device__ __forceinline__ long long r_largest_lt(const double *a, long long n1, double t) {
  long long lo = -1, hi = n1 - 1;
  while (lo < hi) { const long long mid = (lo + hi + 1) >> 1; if (a[mid] < t) lo = mid; else hi = mid - 1; }
  return lo;
}

__global__ void corrected_inverse_audit(const float *polynomial, const float *timing, const float *nodes,
                                        const float *weights, const float *time, int intervals,
                                        unsigned int bits_start, unsigned int count,
                                        const double *T_lo, const double *T_hi, const double *fmin_tab,
                                        const double *fmax_tab, int log2n,
                                        unsigned long long *max_far, unsigned long long *max_legacy,
                                        unsigned long long *max_width,
                                        unsigned long long *counters /*[3]: processed, empty, nonfinite*/) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const unsigned int bits = bits_start + i;
  const float t = __uint_as_float(bits);
  float x[3];
  const float s = progress_at_time(polynomial, timing, nodes, weights, time, 0, intervals, t, x);
  const long long N = 1LL << log2n;
  const double inv = 1.0 / double(N);
  const double td = double(t);
  double lo, hi;
  if (!(td > 0.0)) {
    lo = 0.0; hi = 0.0;
  } else {
    const long long k = r_largest_le(T_hi, N + 1, td);
    const long long j = r_largest_lt(T_lo, N + 1, td);
    if (k >= N || k < 0) {
      lo = k >= N ? 1.0 : 0.0;
    } else {
      const double sk = double(k) * inv, sk1 = double(k + 1) * inv;
      double v = __dadd_rd(sk, __ddiv_rd(__dsub_rd(td, T_hi[k]), fmax_tab[k]));
      v = v > sk ? v : sk;
      lo = v < sk1 ? v : sk1;
    }
    if (j >= N) {
      hi = 1.0;
    } else if (j < 0) {
      hi = 0.0;
    } else {
      const double sj = double(j) * inv, sj1 = double(j + 1) * inv;
      double v = __dadd_ru(sj, __ddiv_ru(__dsub_ru(td, T_lo[j]), fmin_tab[j]));
      hi = v < sj1 ? v : sj1;
      if (hi > 1.0) hi = 1.0;
    }
  }
  const double sd = double(s);
  const double far_up = __dsub_ru(sd, lo), far_dn = __dsub_ru(hi, sd);
  double e = far_up > far_dn ? far_up : far_dn;
  if (!(e >= 0.0)) e = 0.0;  // both negative is impossible unless lo > hi; counted below
  const double ahead = __dsub_ru(sd, hi), behind = __dsub_ru(lo, sd);
  const double ea = ahead > 0 ? ahead : 0.0, eb = behind > 0 ? behind : 0.0;
  const double legacy = ea > eb ? ea : eb;
  const double width = __dsub_ru(hi, lo);
  atomicMax(max_far, (static_cast<unsigned long long>(__float_as_uint(__double2float_ru(e))) << 32) | bits);
  atomicMax(max_legacy, (static_cast<unsigned long long>(__float_as_uint(__double2float_ru(legacy))) << 32) | bits);
  atomicMax(max_width, (static_cast<unsigned long long>(__float_as_uint(__double2float_ru(width > 0 ? width : 0.0))) << 32) | bits);
  atomicAdd(&counters[0], 1ULL);
  if (lo > hi) atomicAdd(&counters[1], 1ULL);
  if (!isfinite(s) || !isfinite(e)) atomicAdd(&counters[2], 1ULL);
}
}  // namespace curobo_certify_review

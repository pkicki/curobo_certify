// Exhaustive native time->progress audit (curobo_certify).
// Compiled after the UNMODIFIED pinned headers timeopt_common.cuh and
// timeopt_path_kernel.cuh with cuRobo's own NVRTC options. For every float32
// requested time t = bits_start + i it calls the upstream progress_at_time (the
// exact function used by sample_with_progress_kernel) and compares the native s
// with a rigorous bracket [lo, hi] of the exact mathematical inverse s*(t)
// computed from CPU-side interval tables (T_lo, T_hi, fmin, fmax on s_k = k/N).
// Bracket arithmetic uses explicitly directed-rounding double intrinsics.
#pragma once
namespace curobo_certify_harness {
using namespace curobo::trajectory::timeopt;

__device__ __forceinline__ long long largest_le(const double *a, long long n1, double t) {
  // largest index in [0, n1-1] with a[idx] <= t, or -1
  long long lo = -1, hi = n1 - 1;
  while (lo < hi) {
    const long long mid = (lo + hi + 1) >> 1;
    if (a[mid] <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}
__device__ __forceinline__ long long largest_lt(const double *a, long long n1, double t) {
  long long lo = -1, hi = n1 - 1;
  while (lo < hi) {
    const long long mid = (lo + hi + 1) >> 1;
    if (a[mid] < t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

__global__ void inverse_audit(const float *polynomial, const float *timing, const float *nodes,
                              const float *weights, const float *time, int intervals,
                              unsigned int bits_start, unsigned int count,
                              const double *T_lo, const double *T_hi, const double *fmin_tab,
                              const double *fmax_tab, int log2n,
                              unsigned long long *max_any, unsigned long long *max_ahead,
                              unsigned long long *max_behind) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const unsigned int bits = bits_start + i;
  const float t = __uint_as_float(bits);
  float x[3];
  const float s = progress_at_time(polynomial, timing, nodes, weights, time, 0, intervals, t, x);
  const long long N = 1LL << log2n;
  const double inv = 1.0 / double(N);  // exact power of two
  const double td = double(t);
  double lo, hi;
  if (!(td > 0.0)) {
    lo = 0.0; hi = 0.0;
  } else {
    const long long k = largest_le(T_hi, N + 1, td);
    const long long j = largest_lt(T_lo, N + 1, td);
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
  const double ahead = __dsub_ru(sd, hi);   // native beyond the exact inverse
  const double behind = __dsub_ru(lo, sd);  // native before the exact inverse
  const double e_ahead = ahead > 0 ? ahead : 0.0, e_behind = behind > 0 ? behind : 0.0;
  const double e = e_ahead > e_behind ? e_ahead : e_behind;
  const unsigned long long key = (static_cast<unsigned long long>(__float_as_uint(__double2float_ru(e))) << 32) | bits;
  atomicMax(max_any, key);
  atomicMax(max_ahead, (static_cast<unsigned long long>(__float_as_uint(__double2float_ru(e_ahead))) << 32) | bits);
  atomicMax(max_behind, (static_cast<unsigned long long>(__float_as_uint(__double2float_ru(e_behind))) << 32) | bits);
}
}  // namespace curobo_certify_harness

namespace curobo_certify_harness {
// Exhaustive reported-value audit: for every float32 requested time, compute the
// production outputs (progress_at_time + sample tail, same expressions as
// sample_with_progress_kernel) and record max |v|/vlim, |a|/alim, |j|/jlim over
// joints (ratios in double, rounded up) and the count of non-finite outputs.
__global__ void reported_audit(const float *controls, const float *start, const float *goal,
                               const float *timing, const float *polynomial, const float *nodes,
                               const float *weights, const float *time, int intervals, int dof,
                               int knots, int start_offset, unsigned int bits_start, unsigned int count,
                               const double *vlim, const double *alim, const double *jlim,
                               unsigned long long *max_ratio /*[3]*/, unsigned long long *nonfinite) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const unsigned int bits = bits_start + i;
  const float t = __uint_as_float(bits);
  float x[3];
  const float s = progress_at_time(polynomial, timing, nodes, weights, time, 0, intervals, t, x);
  float out[4 * 16];
  tail_values(controls, start, goal, s, x, dof, knots, start_offset, out, dof, 0);
  double r[3] = {0, 0, 0};
  bool bad = false;
  for (int j = 0; j < dof; ++j) {
    const float v = out[dof + j], a = out[2 * dof + j], jk = out[3 * dof + j];
    if (!isfinite(v) || !isfinite(a) || !isfinite(jk)) bad = true;
    const double rv = __ddiv_ru(fabs(double(v)), vlim[j]), ra = __ddiv_ru(fabs(double(a)), alim[j]),
                 rj = __ddiv_ru(fabs(double(jk)), jlim[j]);
    r[0] = rv > r[0] ? rv : r[0]; r[1] = ra > r[1] ? ra : r[1]; r[2] = rj > r[2] ? rj : r[2];
  }
  for (int f = 0; f < 3; ++f)
    atomicMax(&max_ratio[f], (static_cast<unsigned long long>(__float_as_uint(__double2float_ru(r[f]))) << 32) | bits);
  if (bad) atomicAdd(nonfinite, 1ULL);
}
}  // namespace curobo_certify_harness

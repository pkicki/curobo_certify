// SPDX-License-Identifier: MIT
// INDEPENDENT verifier for compiled K3 certificates (curobo_certify).
//
// Written separately from k3_certifier.cpp and sharing no code with it. It trusts
// nothing but the leaf partition claimed by the producer:
//  * recomputes SHA-256 of the canonical input bytes and compares the claim;
//  * recomputes the common knot partition by a two-pointer merge;
//  * re-derives every residual DIRECTLY in Bernstein form on each common span by
//    B-spline blossoming (de Boor with distinct parameters) of the float32
//    controls, Bernstein differentiation, products and degree elevation
//    (the producer instead uses power basis + composition);
//  * replays every leaf by de Casteljau along its dyadic path and re-checks the
//    hull (and strict positivity rules) with its own interval arithmetic
//    (relative widening, not nextafter);
//  * checks that leaves of every required residual form an exact dyadic
//    partition of [0,1] (prefix-free, measure exactly 1), rejects unknown,
//    duplicate or overlapping leaves;
//  * re-checks endpoint/strict-rest/integrability obligations with exact
//    integer arithmetic.
// Requires IEEE binary64, round-to-nearest, no FTZ/DAZ, no fast-math/contraction.
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace vfy {

// ---------- SHA-256 (FIPS 180-4) ----------
struct Sha {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  unsigned char buf[64]; size_t n = 0; uint64_t total = 0;
  static uint32_t r(uint32_t x, int k) { return (x >> k) | (x << (32 - k)); }
  void block(const unsigned char *p) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = (uint32_t(p[4 * i]) << 24) | (p[4 * i + 1] << 16) | (p[4 * i + 2] << 8) | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = r(w[i - 15], 7) ^ r(w[i - 15], 18) ^ (w[i - 15] >> 3), s1 = r(w[i - 2], 17) ^ r(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t t1 = hh + (r(e, 6) ^ r(e, 11) ^ r(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
      uint32_t t2 = (r(a, 2) ^ r(a, 13) ^ r(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  void update(const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data; total += len;
    while (len--) { buf[n++] = *p++; if (n == 64) { block(buf); n = 0; } }
  }
  std::string hex() {
    uint64_t bitsz = total * 8; unsigned char pad = 0x80; update(&pad, 1);
    unsigned char z = 0; while (n != 56) update(&z, 1);
    for (int i = 7; i >= 0; --i) { unsigned char b = (unsigned char)(bitsz >> (8 * i)); update(&b, 1); }
    static const char *d = "0123456789abcdef"; std::string s;
    for (int i = 0; i < 8; ++i) for (int k = 7; k >= 0; --k) s += d[(h[i] >> (4 * k)) & 15];
    return s;
  }
};

// ---------- interval arithmetic by relative widening ----------
struct Iv { double a, b; };
const double REL = 0x1p-52, ABS = 0x1p-1074;
inline bool good(Iv x) { return std::isfinite(x.a) && std::isfinite(x.b) && x.a <= x.b; }
inline Iv widen(double lo, double hi) { return {lo - (std::fabs(lo) * REL + ABS), hi + (std::fabs(hi) * REL + ABS)}; }
inline Iv C(double v) { return {v, v}; }
// Degenerate (point) operands keep an exact point result when an error-free
// transformation proves the floating result exact (Knuth TwoSum / fma residual).
inline bool exact_sum(double x, double y, double &s) {
  s = x + y; const double z = s - x;
  return std::isfinite(s) && ((x - (s - z)) + (y - z)) == 0;
}
inline Iv plus(Iv x, Iv y) {
  double s;
  if (x.a == x.b && y.a == y.b && exact_sum(x.a, y.a, s)) return {s, s};
  return widen(x.a + y.a, x.b + y.b);
}
inline Iv minus(Iv x, Iv y) { return plus(x, Iv{-y.b, -y.a}); }
inline Iv times(Iv x, Iv y) {
  if ((x.a == 0 && x.b == 0 && std::isfinite(y.a) && std::isfinite(y.b)) ||
      (y.a == 0 && y.b == 0 && std::isfinite(x.a) && std::isfinite(x.b))) return {0, 0};
  if (x.a == x.b && y.a == y.b) {
    const double p0 = x.a * y.a;
    if (std::isfinite(p0) && p0 != 0 && std::fma(x.a, y.a, -p0) == 0) return {p0, p0};
  }
  double p[4] = {x.a * y.a, x.a * y.b, x.b * y.a, x.b * y.b};
  return widen(*std::min_element(p, p + 4), *std::max_element(p, p + 4));
}
inline Iv over(Iv x, Iv y) {  // y strictly positive
  if (x.a == 0 && x.b == 0 && y.a > 0 && std::isfinite(y.b)) return {0, 0};
  if (x.a == x.b && y.a == y.b) {
    const double q0 = x.a / y.a;
    if (std::isfinite(q0) && q0 != 0 && std::fma(-q0, y.a, x.a) == 0) return {q0, q0};
  }
  double p[4] = {x.a / y.a, x.a / y.b, x.b / y.a, x.b / y.b};
  return widen(*std::min_element(p, p + 4), *std::max_element(p, p + 4));
}
inline Iv q(long long p, long long d) { return over(C(double(p)), C(double(d))); }

// ---------- B-spline blossom (de Boor with distinct parameters) ----------
// Polar value f(u1,u2,u3) of the cubic piece on [knots[i], knots[i+1]] with controls P[i-3..i].
Iv blossom(const std::vector<Iv> &knots, const std::vector<Iv> &P, int i, const Iv u[3]) {
  Iv d[4];
  for (int j = 0; j < 4; ++j) d[j] = P[i - 3 + j];
  for (int r = 1; r <= 3; ++r) {
    for (int j = 3; j >= r; --j) {
      const int idx = i - 3 + j;
      const Iv kl = knots[idx], kr = knots[idx + 4 - r];
      const Iv den = minus(kr, kl);
      const Iv alpha = over(minus(u[r - 1], kl), den);
      d[j] = plus(times(minus(C(1), alpha), d[j - 1]), times(alpha, d[j]));
    }
  }
  return d[3];
}
// Bernstein coefficients on [L,R] of the cubic piece i: f(L,L,L), f(L,L,R), f(L,R,R), f(R,R,R).
std::vector<Iv> bernstein_piece(const std::vector<Iv> &knots, const std::vector<Iv> &P, int i, Iv L, Iv R) {
  std::vector<Iv> out;
  for (int k = 0; k <= 3; ++k) {
    Iv u[3];
    for (int r = 0; r < 3; ++r) u[r] = r < 3 - k ? L : R;
    out.push_back(blossom(knots, P, i, u));
  }
  return out;
}
using B = std::vector<Iv>;
double binom(int n, int k) { double r = 1; for (int j = 1; j <= k; ++j) r = r * (n - j + 1) / j; return r; }  // exact small ints
B deriv(const B &b, Iv invh) {
  if (b.size() < 2) return B{C(0)};
  const int n = int(b.size()) - 1;
  B d;
  for (int i = 0; i < n; ++i) d.push_back(times(times(C(n), minus(b[i + 1], b[i])), invh));
  return d;
}
B product(const B &f, const B &g) {
  const int m = int(f.size()) - 1, n = int(g.size()) - 1;
  B h(m + n + 1, C(0));
  for (int k = 0; k <= m + n; ++k) {
    Iv s = C(0);
    for (int i = std::max(0, k - n); i <= std::min(m, k); ++i)
      s = plus(s, times(times(f[i], g[k - i]), over(C(binom(m, i) * binom(n, k - i)), C(binom(m + n, k)))));
    h[k] = s;
  }
  return h;
}
B elevate_to(B b, int target) {
  while (int(b.size()) - 1 < target) {
    const int n = int(b.size()) - 1;
    B e(n + 2, C(0));
    e[0] = b[0]; e[n + 1] = b[n];
    for (int i = 1; i <= n; ++i) e[i] = plus(times(q(i, n + 1), b[i - 1]), times(q(n + 1 - i, n + 1), b[i]));
    b = e;
  }
  return b;
}
B sum(B x, B y) {
  const int n = std::max(int(x.size()), int(y.size())) - 1;
  x = elevate_to(x, n); y = elevate_to(y, n);
  for (int i = 0; i <= n; ++i) x[i] = plus(x[i], y[i]);
  return x;
}
B scale(B x, Iv s) { for (auto &v : x) v = times(v, s); return x; }
B shift(B x, double c) { for (auto &v : x) v = plus(v, C(c)); return x; }
void halves(const B &b, B &L, B &R) {
  const int n = int(b.size()) - 1;
  B w = b; L.assign(n + 1, C(0)); R.assign(n + 1, C(0));
  L[0] = w[0]; R[n] = w[n];
  for (int r = 1; r <= n; ++r) {
    for (int i = 0; i + r <= n; ++i) w[i] = times(plus(w[i], w[i + 1]), C(0.5));
    L[r] = w[0]; R[n - r] = w[n - r];
  }
}

// ---------- exact integer arithmetic on float32 values ----------
// Every float32 is m*2^(e) with |m|<2^24, e>=-149. Scale all by 2^149+? We use
// pairs (value as __int128 numerator, common exponent) per check.
struct Ex { __int128 v; int e; bool ok; };
Ex ex_f(float f) {
  if (!std::isfinite(f)) return {0, 0, false};
  if (f == 0) return {0, 0, true};
  int e; float m = std::frexp(f, &e);
  return {(__int128)(long long)std::ldexp(m, 24), e - 24, true};
}
int bl(__int128 v) { if (v < 0) v = -v; int b = 0; while (v) { ++b; v >>= 1; } return b; }
Ex align_add(Ex x, Ex y, long long ky = 1) {  // x + ky*y
  if (!x.ok || !y.ok) return {0, 0, false};
  if (y.v == 0 || ky == 0) return x;
  y.v *= ky;
  if (x.v == 0) return y;
  int e = std::min(x.e, y.e);
  if (bl(x.v) + (x.e - e) > 120 || bl(y.v) + (y.e - e) > 120) return {0, 0, false};
  return {x.v * ((__int128)1 << (x.e - e)) + y.v * ((__int128)1 << (y.e - e)), e, true};
}
Ex ex_mul(Ex x, Ex y) {
  if (!x.ok || !y.ok) return {0, 0, false};
  if (bl(x.v) + bl(y.v) > 124) return {0, 0, false};
  return {x.v * y.v, x.e + y.e, true};
}
int sgn(Ex x) { return x.v > 0 ? 1 : x.v < 0 ? -1 : 0; }

}  // namespace vfy

extern "C" {

struct VLeaf { int32_t span, family, joint, depth; uint32_t bits; double upper; };

// Return 0 if the certificate verifies; otherwise a positive reason code.
// reason_detail receives the failing residual index (or -1).
__attribute__((visibility("default"))) int k3v_verify(const float *geometry, int Cc, int D, const float *timing, const float *q_start,
               const float *q_goal, const double *vlim, const double *alim, const double *jlim,
               const double *tighten, int strict_rest, const char *claimed_sha256,
               const VLeaf *leaves, int nleaves, int *reason_detail) {
  using namespace vfy;
  *reason_detail = -1;
  if (std::fegetround() != FE_TONEAREST) return 90;
#if defined(__SSE__)
  if (_mm_getcsr() & (0x8000u | 0x0040u)) return 91;
#endif
  if (Cc < 4 || D < 1 || D > 64 || nleaves < 1) return 1;
  // 1. input digest
  Sha sh;
  int32_t shape[2] = {Cc, D};
  sh.update(shape, sizeof shape);  // little-endian host assumed (x86-64)
  sh.update(geometry, sizeof(float) * size_t(Cc) * D);
  sh.update(timing, sizeof(float) * 32);
  sh.update(q_start, sizeof(float) * D);
  sh.update(q_goal, sizeof(float) * D);
  sh.update(vlim, sizeof(double) * D); sh.update(alim, sizeof(double) * D); sh.update(jlim, sizeof(double) * D);
  const int S = Cc - 3;
  // 2. partition by two-pointer merge of i/S and m/29
  std::vector<std::pair<long long, long long>> cuts;
  {
    int i = 0, m = 0;
    while (i <= S || m <= 29) {
      if (m > 29 || (i <= S && (long long)i * 29 < (long long)m * S)) cuts.push_back({i++, S});
      else if (i > S || (long long)m * S < (long long)i * 29) cuts.push_back({m++, 29});
      else { cuts.push_back({i, S}); ++i; ++m; }
    }
  }
  const int ns = int(cuts.size()) - 1;
  if (tighten) sh.update(tighten, sizeof(double) * size_t(ns) * D * 3);
  if (sh.hex() != std::string(claimed_sha256)) return 2;
  for (int i = 0; i < Cc * D; ++i) if (!std::isfinite(geometry[i])) return 3;
  for (int i = 0; i < 32; ++i) if (!std::isfinite(timing[i])) return 3;
  // 3. exact endpoint obligations
  for (int side = 0; side < 2; ++side) {
    const int base = side ? Cc - 3 : 0;
    const Ex X = ex_f(timing[side ? 31 : 0]);
    const Ex Xd = align_add(ex_f(timing[side ? 31 : 1]), ex_f(timing[side ? 30 : 0]), -1);  // x'/87
    if (!X.ok || !Xd.ok) return 10;
    if (sgn(X) < 0) return 11;
    if (sgn(X) == 0) {
      const int inward = side ? -sgn(Xd) : sgn(Xd);
      if (inward <= 0) return 12;  // nonpositive or root of order >= 2: not finitely traversable
    }
    for (int j = 0; j < D; ++j) {
      const Ex a = ex_f(geometry[base * D + j]), b = ex_f(geometry[(base + 1) * D + j]), c = ex_f(geometry[(base + 2) * D + j]);
      Ex q6 = align_add(align_add(a, b, 4), c);
      q6 = align_add(q6, ex_f(side ? q_goal[j] : q_start[j]), -6);
      if (!q6.ok) return 13;
      if (sgn(q6) != 0) return 14;
      if (strict_rest) {
        const Ex vel = align_add(c, a, -1), accq = align_add(align_add(a, b, -2), c);
        if (!vel.ok || !accq.ok) return 15;
        if (sgn(X) > 0 && sgn(vel) != 0) return 16;
        // 4*a(end) = 4 S^2 accq X + S vel * 87 Xd   (all exact)
        Ex t1 = ex_mul(ex_mul(accq, X), Ex{(__int128)4 * S * S, 0, true});
        Ex t2 = ex_mul(ex_mul(vel, Xd), Ex{(__int128)87 * S, 0, true});
        Ex acc = align_add(t1, t2);
        if (!acc.ok) return 15;
        if (sgn(acc) != 0) return 17;
      }
    }
  }
  // 4. knot vectors and controls as intervals
  std::vector<Iv> gk, tk;  // geometry uniform knots (i-3)/S for i=0..C+3 ; timing clamped
  for (int i = 0; i <= Cc + 3; ++i) gk.push_back(q(i - 3, S));
  for (int i = 0; i < 3; ++i) tk.push_back(C(0));
  for (int i = 0; i <= 29; ++i) tk.push_back(q(i, 29));
  for (int i = 0; i < 3; ++i) tk.push_back(C(1));
  std::vector<Iv> tc(32);
  for (int i = 0; i < 32; ++i) tc[i] = C(double(timing[i]));
  // 5. group leaves by residual id
  const int per_span = 1 + 4 * D;
  std::vector<std::vector<std::pair<int, uint32_t>>> groups(size_t(ns) * per_span);
  for (int l = 0; l < nleaves; ++l) {
    const VLeaf &v = leaves[l];
    if (v.span < 0 || v.span >= ns || v.family < 0 || v.family > 4 || v.depth < 0 || v.depth > 24) return 20;
    if ((v.family == 0) != (v.joint == -1) || (v.family > 0 && (v.joint < 0 || v.joint >= D))) return 20;
    if (v.depth < 32 && (uint64_t(v.bits) >> v.depth) != 0) return 20;
    const int id = v.span * per_span + (v.family == 0 ? 0 : 1 + v.joint * 4 + (v.family - 1));
    groups[id].push_back({v.depth, v.bits});
  }
  for (int id = 0; id < int(groups.size()); ++id) {
    auto &gr = groups[id];
    if (gr.empty()) { *reason_detail = id; return 21; }
    std::sort(gr.begin(), gr.end());
    uint64_t measure = 0;
    for (size_t a = 0; a < gr.size(); ++a) {
      if (a && gr[a] == gr[a - 1]) { *reason_detail = id; return 22; }
      measure += uint64_t(1) << (24 - gr[a].first);
      for (size_t b = 0; b < gr.size(); ++b) {  // overlap: ancestor relation
        if (gr[b].first < gr[a].first && (gr[a].second >> (gr[a].first - gr[b].first)) == gr[b].second) { *reason_detail = id; return 23; }
      }
    }
    if (measure != (uint64_t(1) << 24)) { *reason_detail = id; return 24; }
  }
  // 6. rebuild residuals and replay leaves
  for (int sp = 0; sp < ns; ++sp) {
    const Iv L = q(cuts[sp].first, cuts[sp].second), R = q(cuts[sp + 1].first, cuts[sp + 1].second);
    const Iv invh = over(C(1), minus(R, L));
    const long long mn = cuts[sp].first * cuts[sp + 1].second + cuts[sp + 1].first * cuts[sp].second;
    const long long md = 2 * cuts[sp].second * cuts[sp + 1].second;
    const int k = std::min(int((__int128)mn * S / md), S - 1);
    const int m = std::min(int((__int128)mn * 29 / md), 28);
    const B x = bernstein_piece(tk, tc, m + 3, L, R);
    const B x1 = deriv(x, invh), x2 = deriv(x1, invh);
    std::vector<B> res;
    res.push_back(scale(x, C(-1)));
    for (int j = 0; j < D; ++j) {
      std::vector<Iv> gc(Cc);
      for (int r = 0; r < Cc; ++r) gc[r] = C(double(geometry[r * D + j]));
      const B qq = bernstein_piece(gk, gc, k + 3, L, R);
      const B q1 = deriv(qq, invh), q2 = deriv(q1, invh), q3 = deriv(q2, invh);
      const double tv = tighten ? tighten[(size_t(sp) * D + j) * 3] : 0, ta = tighten ? tighten[(size_t(sp) * D + j) * 3 + 1] : 0,
                   tj = tighten ? tighten[(size_t(sp) * D + j) * 3 + 2] : 0;
      const Iv vl = minus(C(vlim[j]), C(tv)), al = minus(C(alim[j]), C(ta)), jl = minus(C(jlim[j]), C(tj));
      if (!(vl.a > 0 && al.a > 0 && jl.a > 0)) return 30;
      const Iv vmin = C(vl.a), amin = C(al.a), jmin = C(jl.a);
      res.push_back(shift(scale(product(product(q1, q1), x), over(C(1), times(vmin, vmin))), -1));
      const B acc = sum(product(q2, x), scale(product(q1, x1), C(0.5)));
      res.push_back(shift(scale(acc, over(C(1), amin)), -1));
      res.push_back(shift(scale(acc, over(C(-1), amin)), -1));
      const B ell = sum(sum(product(q3, x), scale(product(q2, x1), C(1.5))), scale(product(q1, x2), C(0.5)));
      res.push_back(shift(scale(product(x, product(ell, ell)), over(C(1), times(jmin, jmin))), -1));
    }
    for (int r = 0; r < per_span; ++r) {
      for (const B &bb : {res[r]}) for (auto &v : bb) if (!good(v)) return 31;
      const int id = sp * per_span + r;
      const bool strict = r == 0;
      for (auto &leaf : groups[id]) {
        B b = res[r];
        for (int d = leaf.first - 1; d >= 0; --d) {
          B Lh, Rh; halves(b, Lh, Rh);
          b = ((leaf.second >> d) & 1u) ? Rh : Lh;
        }
        const int n = int(b.size()) - 1;
        bool neg = false;
        for (auto &v : b) { if (!good(v) || v.b > 0) { *reason_detail = id; return 40; } if (v.b < 0) neg = true; }
        if (strict) {
          const bool gl = sp == 0 && leaf.second == 0;
          const bool gr = sp == ns - 1 && uint64_t(leaf.second) == (uint64_t(1) << leaf.first) - 1;
          if (!neg || (!gl && !(b[0].b < 0)) || (!gr && !(b[n].b < 0))) { *reason_detail = id; return 41; }
        }
      }
    }
  }
  return 0;
}
}

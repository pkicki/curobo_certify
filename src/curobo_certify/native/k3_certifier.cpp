// SPDX-License-Identifier: MIT
// Complete compiled K3 certificate PRODUCER for cuRobo TimeOpt outputs.
//
// Input: float32 effective geometry [C][D] (uniform cardinal cubic B-spline),
// 32 float32 timing controls (clamped cubic squared-speed spline, knots k/29),
// requested start/goal, per-joint limits (optionally per common span tightened
// by a native-evaluation error bound), and outward binary64 intervals of the
// exact timing basis power coefficients.
// Output: overall status, per-family status, compact proof leaves (dyadic paths)
// or a rigorous violation witness. Proof replay is done by k3_verifier.cpp, an
// independently written program; this producer's flags are never trusted.
//
// Arithmetic: binary64 with explicit nextafter outward rounding of every primitive
// (requires round-to-nearest, no FTZ/DAZ, -fno-fast-math -ffp-contract=off).
// Knot regularity is structural: q is a uniform cubic B-spline (C2) and x a
// clamped cubic B-spline with simple interior knots (C2), both generated here
// from the controls; endpoint/strict-rest/integrability obligations are checked
// EXACTLY with a dyadic integer type.
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

#define K3_API __attribute__((visibility("default")))
namespace {
constexpr double INF = std::numeric_limits<double>::infinity();
struct I { double lo, hi; };
// Inline IEEE-754 binary64 successor/predecessor (equivalent to nextafter towards
// +/-inf; verified against std::nextafter by k3c_selftest_nextafter).
inline double up(double x) {
  if (!(x == x) || x == INF) return x;
  if (x == 0) return 4.9406564584124654e-324;
  uint64_t b; std::memcpy(&b, &x, 8);
  b = x > 0 ? b + 1 : b - 1;
  double r; std::memcpy(&r, &b, 8); return r;
}
inline double dn(double x) { return -up(-x); }
inline void chk(const I &a) {
  if (!std::isfinite(a.lo) || !std::isfinite(a.hi) || a.lo > a.hi) throw std::range_error("interval overflow/NaN");
}
inline I pt(double x) { return {x, x}; }
// Point operands whose result is exactly representable (checked error-free with
// TwoSum / fma) stay points; this keeps exact zeros (e.g. x(0)=0) exact.
inline bool point(I a) { return a.lo == a.hi; }
inline I add(I a, I b) {
  if (a.lo == 0 && a.hi == 0) return b;
  if (b.lo == 0 && b.hi == 0) return a;
  if (point(a) && point(b)) {
    const double s = a.lo + b.lo, bb = s - a.lo, err = (a.lo - (s - bb)) + (b.lo - bb);
    if (err == 0 && std::isfinite(s)) return pt(s);
  }
  I c{dn(a.lo + b.lo), up(a.hi + b.hi)}; chk(c); return c;
}
inline I neg(I a) { return {-a.hi, -a.lo}; }
inline I sub(I a, I b) { return add(a, neg(b)); }
inline I mul(I a, I b) {
  if ((a.lo == 0 && a.hi == 0) || (b.lo == 0 && b.hi == 0)) return pt(0);
  if (a.lo == 1 && a.hi == 1) return b;
  if (b.lo == 1 && b.hi == 1) return a;
  if (point(a) && point(b)) {
    const double p = a.lo * b.lo;
    if (std::isfinite(p) && p != 0 && std::fma(a.lo, b.lo, -p) == 0) return pt(p);
  }
  const double v[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
  I c{dn(*std::min_element(v, v + 4)), up(*std::max_element(v, v + 4))}; chk(c); return c;
}
inline I half(I a) { return {a.lo * 0.5, a.hi * 0.5}; }  // exact unless subnormal (flagged by chk on use)
inline I divpos(I a, I b) {
  if (!(b.lo > 0)) throw std::range_error("nonpositive divisor");
  if (point(a) && point(b)) {
    if (a.lo == 0) return pt(0);
    const double q = a.lo / b.lo;
    if (std::isfinite(q) && q != 0 && std::fma(-q, b.lo, a.lo) == 0) return pt(q);
  }
  const double v[4] = {a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi};
  I c{dn(*std::min_element(v, v + 4)), up(*std::max_element(v, v + 4))}; chk(c); return c;
}
inline I rat(long long p, long long q) { return divpos(pt(double(p)), pt(double(q))); }  // |p|,q < 2^53

using P = std::vector<I>;  // ascending power basis
P padd(const P &a, const P &b) {
  P c(std::max(a.size(), b.size()), pt(0));
  for (size_t k = 0; k < c.size(); ++k) c[k] = add(k < a.size() ? a[k] : pt(0), k < b.size() ? b[k] : pt(0));
  return c;
}
P pscale(const P &a, I s) { P c; for (auto v : a) c.push_back(mul(v, s)); return c; }
P pmul(const P &a, const P &b) {
  P c(a.size() + b.size() - 1, pt(0));
  for (size_t i = 0; i < a.size(); ++i) for (size_t j = 0; j < b.size(); ++j) c[i + j] = add(c[i + j], mul(a[i], b[j]));
  return c;
}
P pdiff(const P &a, I invh) {
  if (a.size() < 2) return {pt(0)};
  P c; for (size_t i = 1; i < a.size(); ++i) c.push_back(mul(mul(a[i], pt(double(i))), invh));
  return c;
}
// p(alpha + beta*u) by Horner composition.
// Degree-exact: the result has the same length as p (no padded zero degree), so
// residual Bernstein degrees are 3/7/4/4/9 exactly as in the independent verifier.
P compose(const P &p, I alpha, I beta) {
  P out{p.back()};
  for (size_t k = p.size() - 1; k-- > 0;) out = padd(pmul(out, P{alpha, beta}), P{p[k]});
  return out;
}
unsigned binom(unsigned n, unsigned k) { unsigned r = 1; for (unsigned j = 1; j <= k; ++j) r = r * (n - j + 1) / j; return r; }
std::vector<I> to_bernstein(const P &p) {
  const unsigned n = unsigned(p.size() - 1);
  std::vector<I> b;
  for (unsigned k = 0; k <= n; ++k) {
    I s = pt(0);
    for (unsigned i = 0; i <= k; ++i) s = add(s, mul(p[i], rat(binom(k, i), binom(n, i))));
    b.push_back(s);
  }
  return b;
}
void split(const std::vector<I> &b, std::vector<I> &L, std::vector<I> &R) {
  const size_t n = b.size() - 1;
  std::vector<I> w = b; L.assign(n + 1, pt(0)); R.assign(n + 1, pt(0));
  L[0] = w[0]; R[n] = w[n];
  for (size_t r = 1; r <= n; ++r) {
    for (size_t i = 0; i + r <= n; ++i) { I s = add(w[i], w[i + 1]); w[i] = {s.lo * 0.5, s.hi * 0.5}; chk(w[i]); }
    L[r] = w[0]; R[n - r] = w[n - r];
  }
}

// ---------------- exact dyadic (for boundary obligations) ----------------
// Exact dyadic m*2^e with __int128 mantissa; any possible overflow fails closed (ok=false).
struct Dy { __int128 m; int e; bool ok; };
inline int bits(__int128 v) { if (v < 0) v = -v; int b = 0; while (v) { ++b; v >>= 1; } return b; }
inline Dy norm(Dy a) {
  if (!a.ok) return a;
  if (a.m == 0) return {0, 0, true};
  while ((a.m & 1) == 0) { a.m >>= 1; ++a.e; }  // arithmetic shift of an even value is exact
  return a;
}
Dy dy(double x) {  // exact for finite doubles (float32 inputs)
  if (!std::isfinite(x)) return {0, 0, false};
  if (x == 0) return {0, 0, true};
  int e; double f = std::frexp(x, &e);
  long long m = (long long)std::ldexp(f, 53);
  return norm({m, e - 53, true});
}
Dy dmul(Dy a, Dy b) {
  if (!a.ok || !b.ok) return {0, 0, false};
  if (a.m == 0 || b.m == 0) return {0, 0, true};
  if (bits(a.m) + bits(b.m) > 125) return {0, 0, false};
  return norm({a.m * b.m, a.e + b.e, true});
}
Dy dadd(Dy a, Dy b) {
  if (!a.ok || !b.ok) return {0, 0, false};
  if (a.m == 0) return b;
  if (b.m == 0) return a;
  if (a.e < b.e) std::swap(a, b);
  const int sh = a.e - b.e;
  if (sh > 124 || bits(a.m) + sh > 125 || bits(b.m) > 125) return {0, 0, false};
  return norm({a.m * ((__int128)1 << sh) + b.m, b.e, true});
}
Dy dint(long long k) { return dy(double(k)); }
int dsign(Dy a) { return a.m > 0 ? 1 : a.m < 0 ? -1 : 0; }

enum Status { CERTIFIED = 0, VIOLATED = 1, UNKNOWN_DEPTH = 2, UNKNOWN_NODES = 3, INVALID_INPUT = 4,
              UNKNOWN_BOUNDARY = 5, VIOLATED_BOUNDARY = 6, NUMERICAL_ERROR = 7, UNKNOWN_UNIFORM = 8 };

struct Leaf { int32_t span, family, joint, depth; uint32_t bits; double upper; };
struct Witness { int32_t span = -1, family = -1, joint = -1, depth = 0; uint32_t bits = 0; int32_t where = 0; double lo = 0; };

struct Ctx {
  int max_depth = 0, max_nodes = 0, mode = 0, witnesses = 1;  // mode 0 adaptive, 1 uniform at max_depth
  long long nodes = 0;
  std::vector<Leaf> *leaves = nullptr;
  Witness w;
  int depth_hits = 0;
};

// One residual: returns CERTIFIED / VIOLATED / UNKNOWN_*.
int run_residual(Ctx &c, const std::vector<I> &root, int span, int family, int joint, bool strict,
                 bool left_global, bool right_global) {
  struct Node { std::vector<I> b; int depth; uint32_t bits; };
  std::vector<Node> queue{{root, 0, 0}};
  size_t head = 0;
  int result = CERTIFIED;
  const long long one = 1;
  while (head < queue.size()) {
    if (c.nodes >= c.max_nodes) return UNKNOWN_NODES;
    Node nd = std::move(queue[head++]);
    ++c.nodes;
    const auto &b = nd.b;
    const size_t n = b.size() - 1;
    // node interval in [0,1]: [bits, bits+1] / 2^depth
    const bool at_left = left_global && nd.bits == 0;
    const bool at_right = right_global && (long long)nd.bits == (one << nd.depth) - 1;
    double hi = -INF; bool any_neg = false;
    for (auto &v : b) { hi = std::max(hi, v.hi); if (v.hi < 0) any_neg = true; }
    bool pass = hi <= 0;
    if (strict) pass = pass && any_neg && (at_left || b[0].hi < 0) && (at_right || b[n].hi < 0);
    if (c.mode == 1) {  // uniform: no early acceptance before the target level, no witnesses
      if (nd.depth < c.max_depth) {
        std::vector<I> L, R; split(b, L, R);
        queue.push_back({L, nd.depth + 1, nd.bits << 1});
        queue.push_back({R, nd.depth + 1, (nd.bits << 1) | 1u});
        continue;
      }
      if (pass) c.leaves->push_back({span, family, joint, nd.depth, nd.bits, hi});
      else result = UNKNOWN_UNIFORM;
      continue;
    }
    if (c.witnesses) {
      std::vector<I> L, R; split(b, L, R);
      const I m = L[n];
      int where = 0; double lo = 0;
      auto bad = [&](I v, bool interior) { return strict ? (interior && v.lo >= 0) : v.lo > 0; };
      if (bad(b[0], !at_left)) { where = 1; lo = b[0].lo; }
      else if (bad(b[n], !at_right)) { where = 2; lo = b[n].lo; }
      else if (bad(m, true)) { where = 3; lo = m.lo; }
      if (where) {
        c.w = {span, family, joint, nd.depth, nd.bits, where, lo};
        return VIOLATED;
      }
    }
    if (pass) { c.leaves->push_back({span, family, joint, nd.depth, nd.bits, hi}); continue; }
    if (nd.depth >= c.max_depth) { result = UNKNOWN_DEPTH; ++c.depth_hits; continue; }
    std::vector<I> L, R; split(b, L, R);
    queue.push_back({std::move(L), nd.depth + 1, nd.bits << 1});
    queue.push_back({std::move(R), nd.depth + 1, (nd.bits << 1) | 1u});
  }
  return result;
}

struct Cut { long long num, den; };
inline bool lt(Cut a, Cut b) { return (__int128)a.num * b.den < (__int128)b.num * a.den; }
inline bool eqc(Cut a, Cut b) { return (__int128)a.num * b.den == (__int128)b.num * a.den; }
}  // namespace

extern "C" {

K3_API int k3c_abi_version() { return 1; }

K3_API int k3c_environment_ok() {
#if defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0
  return 0;
#endif
  if (std::fegetround() != FE_TONEAREST) return 0;
#if defined(__SSE__)
  const unsigned csr = _mm_getcsr();
  if (csr & (0x8000u | 0x0040u)) return 0;  // FTZ or DAZ
#endif
  volatile double tiny = 4.9406564584124654e-324;
  return tiny / 2.0 == 0.0 && tiny > 0 ? 1 : 0;
}

// Returns number of common spans, writes cut numerators/denominators (capacity cap).
K3_API int k3c_common_spans(int C, long long *num, long long *den, int cap) {
  const int S = C - 3;
  std::vector<Cut> cuts;
  for (int i = 0; i <= S; ++i) cuts.push_back({i, S});
  for (int i = 0; i <= 29; ++i) cuts.push_back({i, 29});
  std::sort(cuts.begin(), cuts.end(), lt);
  std::vector<Cut> u;
  for (auto c : cuts) if (u.empty() || !eqc(u.back(), c)) u.push_back(c);
  const int ns = int(u.size()) - 1;
  for (int i = 0; i <= ns && i < cap; ++i) { num[i] = u[i].num; den[i] = u[i].den; }
  return ns;
}

// timing_lo/hi: [29][4][4] outward intervals of exact coefficient of (s - m/29)^p
// (ascending p) multiplying control m+k.
// tighten: nullptr or [ns][D][3] nonnegative error bounds subtracted from limits.
// leaf_out: capacity leaf_cap records of 6 int64-sized fields (see Leaf).
K3_API int k3c_certify_mt(const float *geometry, int C, int D, const float *timing,
                const float *q_start, const float *q_goal,
                const double *vlim, const double *alim, const double *jlim,
                const double *timing_lo, const double *timing_hi, const double *tighten,
                int max_depth, int max_nodes, int mode, int witnesses, int strict_rest,
                int *family_status, long long *nodes_out, int *boundary_status,
                Leaf *leaf_out, int leaf_cap, int *leaf_count, Witness *witness_out, int threads) ;
K3_API int k3c_certify(const float *geometry, int C, int D, const float *timing,
                const float *q_start, const float *q_goal,
                const double *vlim, const double *alim, const double *jlim,
                const double *timing_lo, const double *timing_hi, const double *tighten,
                int max_depth, int max_nodes, int mode, int witnesses, int strict_rest,
                int *family_status, long long *nodes_out, int *boundary_status,
                Leaf *leaf_out, int leaf_cap, int *leaf_count, Witness *witness_out) {
  return k3c_certify_mt(geometry, C, D, timing, q_start, q_goal, vlim, alim, jlim, timing_lo, timing_hi, tighten,
                        max_depth, max_nodes, mode, witnesses, strict_rest, family_status, nodes_out,
                        boundary_status, leaf_out, leaf_cap, leaf_count, witness_out, 1);
}
K3_API int k3c_certify_mt(const float *geometry, int C, int D, const float *timing,
                const float *q_start, const float *q_goal,
                const double *vlim, const double *alim, const double *jlim,
                const double *timing_lo, const double *timing_hi, const double *tighten,
                int max_depth, int max_nodes, int mode, int witnesses, int strict_rest,
                int *family_status, long long *nodes_out, int *boundary_status,
                Leaf *leaf_out, int leaf_cap, int *leaf_count, Witness *witness_out, int threads) {
  try {
    if (!k3c_environment_ok()) return NUMERICAL_ERROR;
    if (C < 4 || D < 1 || D > 64 || C > 4096 || max_depth < 0 || max_depth > 24 || max_nodes < 1) return INVALID_INPUT;
    for (int i = 0; i < C * D; ++i) if (!std::isfinite(geometry[i])) return INVALID_INPUT;
    for (int i = 0; i < 32; ++i) if (!std::isfinite(timing[i])) return INVALID_INPUT;
    for (int j = 0; j < D; ++j)
      if (!(vlim[j] > 0) || !(alim[j] > 0) || !(jlim[j] > 0) || !std::isfinite(vlim[j] + alim[j] + jlim[j])
          || !std::isfinite(q_start[j]) || !std::isfinite(q_goal[j])) return INVALID_INPUT;
    const int S = C - 3;
    auto g = [&](int r, int j) { return double(geometry[r * D + j]); };

    // ---------- exact boundary obligations ----------
    int bstat = 0;  // 0 pass, 1 violated, 2 unknown
    auto note = [&](int s) { bstat = std::max(bstat, s == 1 ? 3 : s); };  // violated dominates
    const Dy x0 = dy(timing[0]), x1 = dy(timing[31]);
    const Dy dx0 = dmul(dint(87), dadd(dy(timing[1]), dmul(dint(-1), dy(timing[0]))));    // x'(0)
    const Dy dx1 = dmul(dint(87), dadd(dy(timing[31]), dmul(dint(-1), dy(timing[30]))));  // x'(1)
    for (int side = 0; side < 2; ++side) {
      const int r = side == 0 ? 0 : C - 3;
      const double target = side == 0 ? q_start[0] : q_goal[0];
      (void)target;
      const Dy X = side ? x1 : x0, DX = side ? dx1 : dx0;
      if (!X.ok || !DX.ok) { note(2); continue; }
      for (int j = 0; j < D; ++j) {
        const Dy c0 = dy(g(r, j)), c1 = dy(g(r + 1, j)), c2 = dy(g(r + 2, j));
        const Dy q6 = dadd(dadd(c0, dmul(dint(4), c1)), c2);
        const Dy want = dmul(dint(6), dy(side ? q_goal[j] : q_start[j]));
        const Dy diff = dadd(q6, dmul(dint(-1), want));
        if (!diff.ok) note(2); else if (dsign(diff) != 0) note(1);
        if (strict_rest) {
          const Dy qs = dadd(c2, dmul(dint(-1), c0));                          // 2/S * q'(end)
          const Dy qss = dadd(dadd(c0, dmul(dint(-2), c1)), c2);               // q''/S^2
          if (!qs.ok || !qss.ok) { note(2); continue; }
          if (dsign(X) < 0) { note(2); continue; }
          if (dsign(X) > 0 && dsign(qs) != 0) note(1);                      // nonzero velocity
          // a = q'' x + q' x'/2 ; scaled exactly: S^2*qss*X + (S/2)*qs*DX/2
          const Dy acc = dadd(dmul(dmul(dint((long long)S * S), qss), X),
                              dmul(dmul(dint(S), qs), dmul(DX, dy(0.25))));
          if (!acc.ok) note(2); else if (dsign(acc) != 0) note(1);
        }
      }
      // local integrability of ds/sqrt(x) at the endpoint
      const int inward = side == 0 ? dsign(DX) : -dsign(DX);
      if (dsign(X) > 0) {}
      else if (dsign(X) < 0) note(1);
      else if (inward > 0) {}
      else if (inward < 0) note(1);
      else note(1);  // root of order >= 2 (or identically zero x): infinite/undefined traversal
    }
    *boundary_status = bstat == 0 ? 0 : (bstat == 3 ? 1 : 2);

    // ---------- common spans ----------
    std::vector<long long> num(S + 40), den(S + 40);
    const int ns = k3c_common_spans(C, num.data(), den.data(), S + 40);
    // Per-span independent proof search (v2 budget semantics, identical for every thread
    // count): each span runs with a local node cap; spans are merged in order and the
    // first VIOLATED (span, family, joint order) wins; a cumulative node count above the
    // budget fails closed as UNKNOWN_NODES at that span.
    struct SpanResult { int overall = CERTIFIED; int fam[5] = {0, 0, 0, 0, 0}; long long nodes = 0;
                        std::vector<Leaf> leaves; Witness w; bool error = false; };
    std::vector<SpanResult> res(static_cast<size_t>(ns));
    auto worse = [](int a, int b) {  // VIOLATED > NODES > DEPTH/UNIFORM > CERTIFIED
      auto rank = [](int s) { return s == VIOLATED ? 4 : s == UNKNOWN_NODES ? 3 : s == CERTIFIED ? 0 : 2; };
      return rank(b) > rank(a) ? b : a;
    };
    auto run_span = [&](int sp, SpanResult &SR) {
      Ctx lc;
      lc.max_depth = max_depth; lc.max_nodes = max_nodes; lc.mode = mode; lc.witnesses = witnesses;
      lc.leaves = &SR.leaves;
      struct Fin { Ctx &c; SpanResult &r; ~Fin() { r.nodes = c.nodes; r.w = c.w; } } fin{lc, SR};
      const Cut L{num[sp], den[sp]}, R{num[sp + 1], den[sp + 1]};
      // h = R-L exactly as rational with denominator den_L*den_R
      const I h = rat(R.num * L.den - L.num * R.den, L.den * R.den);
      const I invh = divpos(pt(1), h);
      // geometry segment k and local t range
      const Cut M{L.num * R.den + R.num * L.den, 2 * L.den * R.den};
      const int k = std::min(int(((__int128)M.num * S) / M.den), S - 1);
      const I t0 = rat(L.num * S - (long long)k * L.den, L.den), t1 = rat(R.num * S - (long long)k * R.den, R.den);
      const I dt = sub(t1, t0);
      const int m = std::min(int(((__int128)M.num * 29) / M.den), 28);
      const I tau0 = rat(L.num * 29 - (long long)m * L.den, 29 * L.den);
      P xt(4, pt(0));
      for (int p = 0; p < 4; ++p)
        for (int kk = 0; kk < 4; ++kk) {
          const int idx = (m * 4 + p) * 4 + kk;
          xt[p] = add(xt[p], mul(I{timing_lo[idx], timing_hi[idx]}, pt(double(timing[m + kk]))));
        }
      const P x = compose(xt, tau0, h);
      const P xp = pdiff(x, invh), xpp = pdiff(xp, invh);
      const bool lg = sp == 0, rg = sp == ns - 1;
      int st = run_residual(lc, to_bernstein(pscale(x, pt(-1))), sp, 0, -1, true, lg, rg);
      SR.fam[0] = worse(SR.fam[0], st); SR.overall = worse(SR.overall, st);
      if (st == VIOLATED || st == UNKNOWN_NODES) return;
      for (int j = 0; j < D; ++j) {
        const double c0 = g(k, j), c1 = g(k + 1, j), c2 = g(k + 2, j), c3 = g(k + 3, j);
        const I six = pt(6);
        // Cardinal cubic segment in local t (k*float32 is exact in binary64 for k<=6).
        P qt{divpos(add(add(pt(c0), pt(4 * c1)), pt(c2)), six),
             divpos(sub(pt(3 * c2), pt(3 * c0)), six),
             divpos(add(sub(pt(3 * c0), pt(6 * c1)), pt(3 * c2)), six),
             divpos(add(sub(add(neg(pt(c0)), pt(3 * c1)), pt(3 * c2)), pt(c3)), six)};
        const P q = compose(qt, t0, dt);
        const P q1 = pdiff(q, invh), q2 = pdiff(q1, invh), q3 = pdiff(q2, invh);
        const double tv = tighten ? tighten[(size_t(sp) * D + j) * 3 + 0] : 0.0;
        const double ta = tighten ? tighten[(size_t(sp) * D + j) * 3 + 1] : 0.0;
        const double tj = tighten ? tighten[(size_t(sp) * D + j) * 3 + 2] : 0.0;
        const I vl = sub(pt(vlim[j]), pt(tv)), al = sub(pt(alim[j]), pt(ta)), jl = sub(pt(jlim[j]), pt(tj));
        if (!(vl.lo > 0 && al.lo > 0 && jl.lo > 0)) { SR.overall = worse(SR.overall, UNKNOWN_DEPTH); continue; }
        // conservative: divide by the SMALLEST limit value (lo) => residual upper bound
        const I vinv = divpos(pt(1), mul(pt(vl.lo), pt(vl.lo))), ainv = divpos(pt(1), pt(al.lo)),
                jinv = divpos(pt(1), mul(pt(jl.lo), pt(jl.lo)));
        const P vres = padd(pscale(pmul(pmul(q1, q1), x), vinv), P{pt(-1)});
        const P acc = padd(pmul(q2, x), pscale(pmul(q1, xp), pt(0.5)));
        const P ap = padd(pscale(acc, ainv), P{pt(-1)}), am = padd(pscale(acc, neg(ainv)), P{pt(-1)});
        const P ell = padd(padd(pmul(q3, x), pscale(pmul(q2, xp), pt(1.5))), pscale(pmul(q1, xpp), pt(0.5)));
        const P jres = padd(pscale(pmul(x, pmul(ell, ell)), jinv), P{pt(-1)});
        const P *fams[4] = {&vres, &ap, &am, &jres};
        for (int f = 0; f < 4; ++f) {
          st = run_residual(lc, to_bernstein(*fams[f]), sp, f + 1, j, false, lg, rg);
          SR.fam[f + 1] = worse(SR.fam[f + 1], st); SR.overall = worse(SR.overall, st);
          if (st == VIOLATED || st == UNKNOWN_NODES) break;
        }
        if (SR.overall == VIOLATED || SR.overall == UNKNOWN_NODES) return;
      }
    };
    if (threads < 1) threads = 1;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int sp = 0; sp < ns; ++sp) {
      try {
        if (!k3c_environment_ok()) { res[size_t(sp)].error = true; continue; }
        run_span(sp, res[size_t(sp)]);
      } catch (const std::exception &) { res[size_t(sp)].error = true; }
    }
    std::vector<Leaf> leaves; leaves.reserve(size_t(ns) * D * 5);
    for (int f = 0; f < 5; ++f) family_status[f] = CERTIFIED;
    int overall = CERTIFIED;
    long long nodes_total = 0;
    Witness witness;
    for (int sp = 0; sp < ns; ++sp) {
      const SpanResult &R = res[size_t(sp)];
      if (R.error) return NUMERICAL_ERROR;
      nodes_total += R.nodes;
      if (nodes_total > max_nodes) { overall = worse(overall, UNKNOWN_NODES); break; }
      for (int f = 0; f < 5; ++f) family_status[f] = worse(family_status[f], R.fam[f]);
      overall = worse(overall, R.overall);
      leaves.insert(leaves.end(), R.leaves.begin(), R.leaves.end());
      if (R.overall == VIOLATED) { witness = R.w; break; }
      if (R.overall == UNKNOWN_NODES) break;
    }
    struct { long long nodes; Witness w; } ctx{nodes_total, witness};
    *nodes_out = ctx.nodes;
    *witness_out = ctx.w;
    *leaf_count = int(leaves.size());
    if (int(leaves.size()) > leaf_cap) return INVALID_INPUT;  // caller must enlarge buffer
    std::memcpy(leaf_out, leaves.data(), leaves.size() * sizeof(Leaf));
    if (overall == CERTIFIED && *boundary_status != 0) return *boundary_status == 1 ? VIOLATED_BOUNDARY : UNKNOWN_BOUNDARY;
    if (*boundary_status == 1 && overall != VIOLATED) return VIOLATED_BOUNDARY;
    return overall;
  } catch (const std::exception &) {
    return NUMERICAL_ERROR;
  }
}


}  // extern "C"  (internal helpers below MUST NOT have C linkage: they would be
   // exported unmangled as add/mul/div/... and can be interposed by libc's div())

// ---------------------------------------------------------------------------
// Native FP32 evaluation error bounds (port of curobo_certify.native_error_bound).
// For each common span and joint: upper bounds E_v, E_a, E_j on
// |native sample_with_progress_kernel output - exact model| for every float32 s
// mapped into that span, under assumptions A1-A5 (see docs/NATIVE_LINKAGE_20261116.md).
// ---------------------------------------------------------------------------
namespace nb {
struct ET { double lo, hi, e; };
struct M { double u, div_rel, sqrt_rel, ftz; };
inline double mag(const ET &a) { return std::max(std::fabs(a.lo), std::fabs(a.hi)); }
inline double rnd(const M &m, double mg, double prop, double rel) { return up(up(prop + up(rel * up(mg + prop))) + m.ftz); }
inline ET cst(double v) { return {v, v, 0.0}; }
inline ET iv(double lo, double hi) { return {lo, hi, 0.0}; }
inline ET add(const M &m, ET a, ET b) { double lo = dn(a.lo + b.lo), hi = up(a.hi + b.hi); double p = up(a.e + b.e); return {lo, hi, rnd(m, std::max(std::fabs(lo), std::fabs(hi)), p, m.u)}; }
inline ET neg(ET a) { return {-a.hi, -a.lo, a.e}; }
inline ET sub(const M &m, ET a, ET b) { return add(m, a, neg(b)); }
inline ET mul(const M &m, ET a, ET b) {
  const double v[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
  const double lo = dn(*std::min_element(v, v + 4)), hi = up(*std::max_element(v, v + 4));
  const double p = up(up(up(mag(a) * b.e) + up(mag(b) * a.e)) + up(a.e * b.e));
  return {lo, hi, rnd(m, std::max(std::fabs(lo), std::fabs(hi)), p, m.u)};
}
inline ET div(const M &m, ET a, ET b) {
  if (b.lo <= 0 && b.hi >= 0) throw std::range_error("divisor contains zero");
  const double bmin = std::min(std::fabs(b.lo), std::fabs(b.hi));
  if (bmin <= b.e) throw std::range_error("divisor error reaches zero");
  const double v[4] = {a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi};
  const double lo = dn(*std::min_element(v, v + 4)), hi = up(*std::max_element(v, v + 4));
  const double mg = std::max(std::fabs(lo), std::fabs(hi));
  const double p = up(up(a.e + up(mg * b.e)) / dn(bmin - b.e));
  return {lo, hi, rnd(m, mg, p, m.div_rel)};
}
inline ET sqrtn(const M &m, ET a) {
  if (a.lo < 0) a = {0.0, std::max(a.hi, 0.0), a.e};
  const double lo = dn(std::sqrt(a.lo)), hi = up(std::sqrt(a.hi));
  double p = up(std::sqrt(a.e));
  if (a.lo > 0) p = std::min(p, up(a.e / dn(std::sqrt(a.lo))));
  return {std::max(lo, 0.0), hi, rnd(m, hi, p, m.sqrt_rel)};
}
inline ET widen(ET a, double x) { return {dn(a.lo - x), up(a.hi + x), up(a.e + x)}; }
inline ET exact_diff(const M &m, double a, double b) {  // fl(a-b) of float32 a,b vs exact a-b
  const double s = a - b, z = s - a, err = (a - (s - z)) + (-b - z);
  const ET d = err == 0 ? ET{s, s, 0.0} : ET{dn(s), up(s), 0.0};
  return {d.lo, d.hi, rnd(m, mag(d), 0.0, m.u)};
}
void native_geom(const M &m, const double c[4], int S, double tlo, double thi, ET q[3]) {
  ET cen[4] = {cst(0), exact_diff(m, c[1], c[0]), exact_diff(m, c[2], c[0]), exact_diff(m, c[3], c[0])};
  const ET t = iv(tlo, thi), one = cst(1), three = cst(3), six = cst(6), nine = cst(9), twelve = cst(12);
  const ET t2 = mul(m, t, t), omt = sub(m, one, t);
  const ET vb[4] = {div(m, mul(m, mul(m, cst(-3), omt), omt), six), div(m, sub(m, mul(m, nine, t2), mul(m, twelve, t)), six),
                    div(m, add(m, add(m, mul(m, cst(-9), t2), mul(m, six, t)), three), six), div(m, mul(m, three, t2), six)};
  const ET ab[4] = {div(m, mul(m, six, sub(m, one, t)), six), div(m, sub(m, mul(m, cst(18), t), cst(12)), six),
                    div(m, add(m, mul(m, cst(-18), t), six), six), div(m, mul(m, six, t), six)};
  const ET jb[4] = {cst(-1), cst(3), cst(-3), cst(1)};
  auto dot = [&](const ET *b) { ET r = cst(0); for (int k = 0; k < 4; ++k) r = add(m, r, mul(m, cen[k], b[k])); return r; };
  const ET kdt = div(m, one, cst(double(S))), kdt2 = mul(m, kdt, kdt), kdt3 = mul(m, kdt2, kdt);
  q[0] = div(m, dot(vb), kdt); q[1] = div(m, dot(ab), kdt2); q[2] = div(m, dot(jb), kdt3);
}
}  // namespace nb

extern "C" K3_API int k3c_native_bounds_mt(const float *geometry, int C, int D, const float *timing, const float *native_table,
                                 const double *timing_lo, const double *timing_hi, int pieces, double delta,
                                 double div_ulps, double sqrt_ulps, double *out, int threads);
extern "C" K3_API int k3c_native_bounds(const float *geometry, int C, int D, const float *timing, const float *native_table,
                                 const double *timing_lo, const double *timing_hi, int pieces, double delta,
                                 double div_ulps, double sqrt_ulps, double *out) {
  return k3c_native_bounds_mt(geometry, C, D, timing, native_table, timing_lo, timing_hi, pieces, delta, div_ulps,
                              sqrt_ulps, out, 1);
}
// Same bounds; common spans are independent and are processed in parallel (OpenMP).
// Results are identical to the single-thread computation (no shared accumulation).
extern "C" K3_API int k3c_native_bounds_mt(const float *geometry, int C, int D, const float *timing, const float *native_table,
                                 const double *timing_lo, const double *timing_hi, int pieces, double delta,
                                 double div_ulps, double sqrt_ulps, double *out, int threads) {
  using namespace nb;
  try {
    if (!k3c_environment_ok()) return NUMERICAL_ERROR;
    const double u = 0x1p-24;
    if (!(delta >= 12 * u) || pieces < 1 || C < 4 || D < 1) return INVALID_INPUT;
    const M m{u, up(div_ulps * 0x1p-23), up(sqrt_ulps * 0x1p-23), 0x1p-126};
    const int S = C - 3;
    std::vector<long long> num(S + 40), den(S + 40);
    const int ns = k3c_common_spans(C, num.data(), den.data(), S + 40);
    // exact timing span polynomial magnitudes: a_p = sum T c (ascending p), outward
    std::vector<std::array<I, 4>> tp(29);
    for (int s = 0; s < 29; ++s)
      for (int p = 0; p < 4; ++p) {
        I acc = pt(0);
        for (int k = 0; k < 4; ++k) acc = ::add(acc, ::mul(I{timing_lo[(s * 4 + p) * 4 + k], timing_hi[(s * 4 + p) * 4 + k]}, pt(timing[s + k])));
        tp[s][p] = acc;
      }
    auto amax = [](I a) { return std::max(std::fabs(a.lo), std::fabs(a.hi)); };
    auto deriv_bound = [&](int s, int order, double lo, double hi) {  // sup |p^(order)| on [lo,hi]
      const double r = std::max(std::fabs(lo), std::fabs(hi));
      double sum = 0, pw = 1;
      for (int i = order; i < 4; ++i) {
        double f = 1; for (int k = 0; k < order; ++k) f *= double(i - k);
        sum = up(sum + up(up(f * amax(tp[s][i])) * pw));
        pw = up(pw * r);
      }
      return sum;
    };
    int failed = 0;
    if (threads < 1) threads = 1;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) reduction(| : failed)
    for (int sp = 0; sp < ns; ++sp) {
     try {
      if (!k3c_environment_ok()) { failed |= 1; continue; }  // each worker's own FP environment
      const long long Ln = num[sp], Ld = den[sp], Rn = num[sp + 1], Rd = den[sp + 1];
      const long long Mn = Ln * Rd + Rn * Ld, Md = 2 * Ld * Rd;
      const int k = std::min(int((__int128)Mn * S / Md), S - 1);
      const int mid_t = std::min(int((__int128)Mn * 29 / Md), 28);
      std::vector<double> worst(size_t(D) * 3, 0.0);
      std::vector<std::array<ET, 3>> qs(static_cast<size_t>(D));
      for (int pc = 0; pc < pieces; ++pc) {
        // piece [a,b] = L + (R-L)*[pc,pc+1]/pieces, as outward doubles
        const I L = rat(Ln, Ld), R = rat(Rn, Rd);
        const I w = ::sub(R, L);
        const I a = ::add(L, ::mul(w, rat(pc, pieces))), b = ::add(L, ::mul(w, rat(pc + 1, pieces)));
        const double tlo = dn(dn(a.lo * S) - k), thi = up(up(b.hi * S) - k);
        // Geometry error analysis depends only on (piece, joint): computed once, reused per timing span.
        for (int j = 0; j < D; ++j) {
          double c[4]; for (int r = 0; r < 4; ++r) c[r] = geometry[(k + r) * D + j];
          ET q[3]; native_geom(m, c, S, dn(tlo - delta * S), up(thi + delta * S), q);
          q[0] = widen(q[0], up(mag(q[1]) * delta));
          q[1] = widen(q[1], up(mag(q[2]) * delta));
          qs[size_t(j)] = {q[0], q[1], q[2]};
        }
        for (int ms = mid_t - 1; ms <= mid_t + 1; ++ms) {
          if (ms < 0 || ms > 28) continue;
          const I left = rat(ms, 29), right = rat(ms + 1, 29);
          const double lo_s = std::max(dn(a.lo - delta), dn(left.lo - delta));
          const double hi_s = std::min(up(b.hi + delta), up(right.hi + delta));
          if (lo_s > hi_s) continue;
          const double llo = dn(lo_s - left.hi), lhi = up(hi_s - left.lo);
          // native timing_state ETs
          ET coef[4];
          for (int P = 0; P < 4; ++P) {
            const int p = 3 - P;
            ET acc = cst(0);
            for (int kk = 0; kk < 4; ++kk) {
              const I ex{timing_lo[(ms * 4 + p) * 4 + kk], timing_hi[(ms * 4 + p) * 4 + kk]};
              const double nat = native_table[(ms * 4 + P) * 32 + ms + kk];
              const ET entry{ex.lo, ex.hi, up(std::max(std::fabs(nat - ex.lo), std::fabs(nat - ex.hi)))};
              acc = add(m, acc, mul(m, entry, cst(timing[ms + kk])));
            }
            coef[P] = acc;
          }
          const ET loc = iv(llo, lhi);
          ET x0 = add(m, mul(m, add(m, mul(m, add(m, mul(m, coef[0], loc), coef[1]), loc), coef[2]), loc), coef[3]);
          ET x1 = add(m, mul(m, add(m, mul(m, mul(m, cst(3), coef[0]), loc), mul(m, cst(2), coef[1])), loc), coef[2]);
          ET x2 = add(m, mul(m, mul(m, cst(6), coef[0]), loc), mul(m, cst(2), coef[1]));
          const double elo = dn(llo - delta), ehi = up(lhi + delta);
          const bool cross = ms != mid_t || a.lo - delta < left.hi || b.hi + delta > right.lo;
          ET xs[3] = {x0, x1, x2};
          for (int i = 0; i < 3; ++i) {
            double shift = up(deriv_bound(ms, i + 1, elo, ehi) * delta);
            if (cross) {
              double jump = 0;
              for (int n = ms - 1; n <= ms + 1; n += 2)
                if (n >= 0 && n <= 28) jump = std::max(jump, up(6 * up(amax(tp[n][3]) + amax(tp[ms][3]))));
              double f = 1, dp = 1; for (int r = 0; r < 3 - i; ++r) { dp = up(dp * delta); f *= (r + 1); }
              shift = up(shift + up(jump * dp / f));
            }
            xs[i] = widen(xs[i], shift);
          }
          const ET speed = sqrtn(m, xs[0]);
          for (int j = 0; j < D; ++j) {
            const ET *q = qs[size_t(j)].data();
            const ET vel = mul(m, q[0], speed);
            const ET acc = add(m, mul(m, q[1], xs[0]), mul(m, mul(m, cst(0.5), q[0]), xs[1]));
            const ET inner = add(m, add(m, mul(m, q[2], xs[0]), mul(m, mul(m, cst(1.5), q[1]), xs[1])),
                                 mul(m, mul(m, cst(0.5), q[0]), xs[2]));
            const ET jerk = mul(m, speed, inner);
            double *wv = &worst[size_t(j) * 3];
            wv[0] = std::max(wv[0], vel.e); wv[1] = std::max(wv[1], acc.e); wv[2] = std::max(wv[2], jerk.e);
          }
        }
      }
      for (int j = 0; j < D; ++j) for (int f = 0; f < 3; ++f) out[(size_t(sp) * D + j) * 3 + f] = worst[size_t(j) * 3 + f];
     } catch (const std::exception &) {
      failed |= 2;
     }
    }
    if (failed) return -NUMERICAL_ERROR;
    return ns;
  } catch (const std::exception &) {
    return -NUMERICAL_ERROR;
  }
}

extern "C" {
K3_API int k3c_leaf_size() { return int(sizeof(Leaf)); }
// Compare the inline successor/predecessor with std::nextafter on n given values.
K3_API int k3c_selftest_nextafter(const double *x, int n) {
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    const double u1 = up(x[i]), u2 = std::nextafter(x[i], INF), d1 = dn(x[i]), d2 = std::nextafter(x[i], -INF);
    if (std::memcmp(&u1, &u2, 8) != 0 || std::memcmp(&d1, &d2, 8) != 0) ++bad;
  }
  return bad;
}
K3_API int k3c_witness_size() { return int(sizeof(Witness)); }
}

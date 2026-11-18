// K4 compiled interval RNEA coefficient producer (curobo_certify).
// Outward-rounded binary64 intervals (each op rounded to nearest then widened one ulp with nextafter),
// rigorous sin/cos (Taylor order 30 at the midpoint with interval coefficients from exact-ish interval
// division + Lagrange remainder bound valid for |mid| <= 8, Lipschitz radius, clamp to [-1,1]; |mid|>8 ->
// [-1,1]), forward-mode dual intervals for first-order Taylor models, and the link-frame recursive
// Newton-Euler algorithm of k4_panda_model.py templated over the number type.
// Model parameters arrive as interval enclosures prepared EXACTLY in Python from contract v2.
// Must be compiled WITHOUT fast-math / FMA contraction (checked by the Python build wrapper).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>

#if defined(__FAST_MATH__)
#error "k4_rnea must not be compiled with -ffast-math"
#endif

namespace {
constexpr double INF = std::numeric_limits<double>::infinity();
// IEEE-754 binary64 successor / predecessor by bit manipulation (same as std::nextafter towards
// +/-inf for every input; checked against std::nextafter in tests/test_k4_compiled.py via k4_selftest_next).
inline double up(double x) {
  if (!(x == x) || x == INF) return x;
  if (x == 0) return 4.9406564584124654e-324;
  uint64_t b; std::memcpy(&b, &x, 8);
  b = x > 0 ? b + 1 : b - 1;
  double r; std::memcpy(&r, &b, 8); return r;
}
inline double dn(double x) { return -up(-x); }

struct I {
  double lo, hi;
  I() : lo(0), hi(0) {}
  I(double x) : lo(x), hi(x) {}
  I(double a, double b) : lo(a), hi(b) {}
};
inline I operator+(I a, I b) { return I(dn(a.lo + b.lo), up(a.hi + b.hi)); }
inline I operator-(I a, I b) { return I(dn(a.lo - b.hi), up(a.hi - b.lo)); }
inline I operator-(I a) { return I(-a.hi, -a.lo); }
inline I operator*(I a, I b) {
  double p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
  double lo = p[0], hi = p[0];
  bool nan = false;
  for (double v : p) { if (std::isnan(v)) nan = true; lo = std::min(lo, v); hi = std::max(hi, v); }
  if (nan) return I(-INF, INF);
  return I(dn(lo), up(hi));
}
inline I idiv_pos(I a, double c) {  // a / c for exact c > 0
  return I(dn(a.lo / c), up(a.hi / c));
}
inline I sqr(I a) {
  double x = a.lo * a.lo, y = a.hi * a.hi;
  double lo = (a.lo <= 0 && a.hi >= 0) ? 0.0 : dn(std::min(x, y));
  return I(lo, up(std::max(x, y)));
}

constexpr int ORDER = 30;
constexpr int ORDER_SMALL = 14;   // |mid| <= 4: sine degree 29 -> 4^30/30! < 4.4e-15 ; cosine degree 28 -> 4^29/29! < 1.1e-14
constexpr double REM_SMALL = 2e-14;
// Lagrange remainders for |m| <= 8: sine degree 61 -> 8^62/62! < 3.2e-30 ; cosine degree 60 -> 8^61/61! < 2.5e-29.
constexpr double REM = 1e-27;

I coeff(int k, int e) {  // (-1)^k / (2k+e)!  computed by outward interval division
  I c(1.0);
  for (int n = 2; n <= 2 * k + e; ++n) c = idiv_pos(c, (double)n);
  return (k % 2) ? -c : c;
}
I COS_C[ORDER + 1], SIN_C[ORDER + 1];
bool coeff_ready = false;
void init_coeff() {
  if (coeff_ready) return;
  for (int k = 0; k <= ORDER; ++k) { COS_C[k] = coeff(k, 0); SIN_C[k] = coeff(k, 1); }
  coeff_ready = true;
}

I trig(I a, bool cosine) {
  double m = 0.5 * a.lo + 0.5 * a.hi;
  if (!(std::fabs(m) <= 8.0)) return I(-1.0, 1.0);
  double r = up(std::max(up(a.hi - m), up(m - a.lo)));
  I x(m), x2 = sqr(x);
  const I* C = cosine ? COS_C : SIN_C;
  const bool small = std::fabs(m) <= 4.0;
  const int ord = small ? ORDER_SMALL : ORDER;
  const double rem = small ? REM_SMALL : REM;
  I acc = C[ord];
  for (int k = ord - 1; k >= 0; --k) acc = acc * x2 + C[k];
  if (!cosine) acc = acc * x;
  double lo = dn(dn(acc.lo - rem) - r), hi = up(up(acc.hi + rem) + r);
  return I(std::max(lo, -1.0), std::min(hi, 1.0));
}

struct D {  // dual interval
  I v, d;
  D() {}
  D(I a) : v(a), d(0.0) {}
  D(I a, I b) : v(a), d(b) {}
};
inline D operator+(D a, D b) { return D(a.v + b.v, a.d + b.d); }
inline D operator-(D a, D b) { return D(a.v - b.v, a.d - b.d); }
inline D operator-(D a) { return D(-a.v, -a.d); }
inline D operator*(D a, D b) { return D(a.v * b.v, a.d * b.v + a.v * b.d); }
inline I tsin(I a) { return trig(a, false); }
inline I tcos(I a) { return trig(a, true); }
inline D tsin(D a) { return D(trig(a.v, false), trig(a.v, true) * a.d); }
inline D tcos(D a) { return D(trig(a.v, true), -(trig(a.v, false) * a.d)); }
inline I mk(I a, I*) { return a; }
inline D mk(I a, D*) { return D(a); }

constexpr int MAXL = 32, MAXJ = 16;
struct Model {
  int n, dof;
  int parent[MAXL], joint[MAXL];
  I R[MAXL][3][3], p[MAXL][3], m[MAXL], c[MAXL][3], Ic[MAXL][3][3], g[3];
};

template <typename T> inline void cross(const T a[3], const T b[3], T o[3]) {
  o[0] = a[1] * b[2] - a[2] * b[1];
  o[1] = a[2] * b[0] - a[0] * b[2];
  o[2] = a[0] * b[1] - a[1] * b[0];
}

// tau = RNEA(q, v, a); v/a may be null (exact zeros); gravity on/off.
template <typename T>
void rnea(const Model& M, const T* q, const T* v, const T* a, bool gravity, T* tau, const T* cqs = nullptr, const T* sqs = nullptr) {
  T* tag = nullptr;
  T zero = mk(I(0.0), tag);
  T w[MAXL][3], wd[MAXL][3], acc[MAXL][3], R[MAXL][3][3], f[MAXL][3], nm[MAXL][3];
  for (int k = 0; k < 3; ++k) {
    w[0][k] = zero; wd[0][k] = zero;
    acc[0][k] = gravity ? mk(-M.g[k], tag) : zero;
  }
  for (int i = 1; i < M.n; ++i) {
    int pa = M.parent[i], j = M.joint[i];
    T Ri[3][3];  // Ri[r][k]
    if (j >= 0) {
      T cq = cqs ? cqs[j] : tcos(q[j]), sq = sqs ? sqs[j] : tsin(q[j]);
      for (int r = 0; r < 3; ++r) {
        Ri[r][0] = cq * mk(M.R[i][r][0], tag) + sq * mk(M.R[i][r][1], tag);
        Ri[r][1] = cq * mk(M.R[i][r][1], tag) - sq * mk(M.R[i][r][0], tag);
        Ri[r][2] = mk(M.R[i][r][2], tag);
      }
    } else {
      for (int r = 0; r < 3; ++r) for (int k = 0; k < 3; ++k) Ri[r][k] = mk(M.R[i][r][k], tag);
    }
    std::memcpy(R[i], Ri, sizeof(Ri));
    T pf[3] = {mk(M.p[i][0], tag), mk(M.p[i][1], tag), mk(M.p[i][2], tag)};
    T t1[3], t2[3], ap[3];
    cross(wd[pa], pf, t1);
    for (int k = 0; k < 3; ++k) ap[k] = acc[pa][k] + t1[k];
    if (v) { T t3[3]; cross(w[pa], pf, t3); cross(w[pa], t3, t2); for (int k = 0; k < 3; ++k) ap[k] = ap[k] + t2[k]; }
    T wr[3], wdr[3];
    for (int k = 0; k < 3; ++k) {
      acc[i][k] = Ri[0][k] * ap[0] + Ri[1][k] * ap[1] + Ri[2][k] * ap[2];
      wr[k] = v ? Ri[0][k] * w[pa][0] + Ri[1][k] * w[pa][1] + Ri[2][k] * w[pa][2] : zero;
      wdr[k] = Ri[0][k] * wd[pa][0] + Ri[1][k] * wd[pa][1] + Ri[2][k] * wd[pa][2];
    }
    if (j >= 0) {
      T qd = v ? v[j] : zero, qdd = a ? a[j] : zero;
      w[i][0] = wr[0]; w[i][1] = wr[1]; w[i][2] = v ? wr[2] + qd : zero;
      wd[i][0] = wdr[0]; wd[i][1] = wdr[1]; wd[i][2] = wdr[2] + qdd;
      if (v) { wd[i][0] = wd[i][0] + wr[1] * qd; wd[i][1] = wd[i][1] - wr[0] * qd; }
    } else {
      for (int k = 0; k < 3; ++k) { w[i][k] = wr[k]; wd[i][k] = wdr[k]; }
    }
  }
  for (int i = 0; i < M.n; ++i) for (int k = 0; k < 3; ++k) { f[i][k] = zero; nm[i][k] = zero; }
  for (int i = 1; i < M.n; ++i) {
    T c[3] = {mk(M.c[i][0], tag), mk(M.c[i][1], tag), mk(M.c[i][2], tag)};
    T mass = mk(M.m[i], tag);
    T t[3], ac[3];
    cross(wd[i], c, t);
    for (int k = 0; k < 3; ++k) ac[k] = acc[i][k] + t[k];
    if (v) { T u[3], u2[3]; cross(w[i], c, u); cross(w[i], u, u2); for (int k = 0; k < 3; ++k) ac[k] = ac[k] + u2[k]; }
    T F[3], N[3];
    for (int k = 0; k < 3; ++k) F[k] = ac[k] * mass;
    for (int r = 0; r < 3; ++r)
      N[r] = mk(M.Ic[i][r][0], tag) * wd[i][0] + mk(M.Ic[i][r][1], tag) * wd[i][1] + mk(M.Ic[i][r][2], tag) * wd[i][2];
    if (v) {
      T Iw[3], x[3];
      for (int r = 0; r < 3; ++r)
        Iw[r] = mk(M.Ic[i][r][0], tag) * w[i][0] + mk(M.Ic[i][r][1], tag) * w[i][1] + mk(M.Ic[i][r][2], tag) * w[i][2];
      cross(w[i], Iw, x);
      for (int k = 0; k < 3; ++k) N[k] = N[k] + x[k];
    }
    T cF[3];
    cross(c, F, cF);
    for (int k = 0; k < 3; ++k) { f[i][k] = f[i][k] + F[k]; nm[i][k] = nm[i][k] + N[k] + cF[k]; }
  }
  for (int i = M.n - 1; i >= 1; --i) {
    if (M.joint[i] >= 0) tau[M.joint[i]] = nm[i][2];
    int pa = M.parent[i];
    T Rf[3], Rn[3], pf[3] = {mk(M.p[i][0], tag), mk(M.p[i][1], tag), mk(M.p[i][2], tag)}, pxf[3];
    for (int r = 0; r < 3; ++r) {
      Rf[r] = R[i][r][0] * f[i][0] + R[i][r][1] * f[i][1] + R[i][r][2] * f[i][2];
      Rn[r] = R[i][r][0] * nm[i][0] + R[i][r][1] * nm[i][1] + R[i][r][2] * nm[i][2];
    }
    cross(pf, Rf, pxf);
    for (int k = 0; k < 3; ++k) { f[pa][k] = f[pa][k] + Rf[k]; nm[pa][k] = nm[pa][k] + Rn[k] + pxf[k]; }
  }
}

bool load_model(Model& M, int n, int dof, const int* parent, const int* joint, const double* P) {
  // P layout per link: R 9x(lo,hi), p 3x, m 1x, c 3x, Ic 9x (row-major) -> 25 intervals = 50 doubles; then g 3x
  if (n > MAXL || dof > MAXJ) return false;
  M.n = n; M.dof = dof;
  for (int i = 0; i < n; ++i) {
    M.parent[i] = parent[i]; M.joint[i] = joint[i];
    const double* b = P + 50 * i;
    int k = 0;
    for (int r = 0; r < 3; ++r) for (int s = 0; s < 3; ++s, ++k) M.R[i][r][s] = I(b[2 * k], b[2 * k + 1]);
    for (int r = 0; r < 3; ++r, ++k) M.p[i][r] = I(b[2 * k], b[2 * k + 1]);
    M.m[i] = I(b[2 * k], b[2 * k + 1]); ++k;
    for (int r = 0; r < 3; ++r, ++k) M.c[i][r] = I(b[2 * k], b[2 * k + 1]);
    for (int r = 0; r < 3; ++r) for (int s = 0; s < 3; ++s, ++k) M.Ic[i][r][s] = I(b[2 * k], b[2 * k + 1]);
  }
  const double* g = P + 50 * n;
  for (int k = 0; k < 3; ++k) M.g[k] = I(g[2 * k], g[2 * k + 1]);
  return true;
}
}  // namespace

extern "C" {
// Box coefficients. in: n_boxes x (3*dof) intervals (q, q', q'') as lo,hi pairs.
// out: n_boxes x (3*dof) intervals (A, B, G).
int k4_box_coefficients(int nlinks, int dof, const int* parent, const int* joint, const double* params,
                        int nbox, const double* in, double* out) {
  init_coeff();
  Model M;
  if (!load_model(M, nlinks, dof, parent, joint, params)) return 1;
  for (int b = 0; b < nbox; ++b) {
    const double* x = in + (size_t)b * 6 * dof;
    I q[MAXJ], qp[MAXJ], qpp[MAXJ], A[MAXJ], B[MAXJ], G[MAXJ];
    for (int j = 0; j < dof; ++j) {
      q[j] = I(x[2 * j], x[2 * j + 1]);
      qp[j] = I(x[2 * (dof + j)], x[2 * (dof + j) + 1]);
      qpp[j] = I(x[2 * (2 * dof + j)], x[2 * (2 * dof + j) + 1]);
    }
    rnea<I>(M, q, nullptr, nullptr, true, G);
    rnea<I>(M, q, nullptr, qp, false, B);
    rnea<I>(M, q, qp, qpp, false, A);
    double* o = out + (size_t)b * 6 * dof;
    for (int j = 0; j < dof; ++j) {
      I bh = B[j] * I(0.5);
      o[2 * j] = A[j].lo; o[2 * j + 1] = A[j].hi;
      o[2 * (dof + j)] = bh.lo; o[2 * (dof + j) + 1] = bh.hi;
      o[2 * (2 * dof + j)] = G[j].lo; o[2 * (2 * dof + j) + 1] = G[j].hi;
    }
  }
  return 0;
}

// Taylor-model derivative boxes: in: n_boxes x (4*dof) intervals (q, q', q'', q''' over the leaf);
// out: n_boxes x (3*dof) intervals (dA/ds, dB/ds, dG/ds over the leaf).
int k4_tm1_slopes(int nlinks, int dof, const int* parent, const int* joint, const double* params,
                  int nbox, const double* in, double* out) {
  init_coeff();
  Model M;
  if (!load_model(M, nlinks, dof, parent, joint, params)) return 1;
  for (int b = 0; b < nbox; ++b) {
    const double* x = in + (size_t)b * 8 * dof;
    D q[MAXJ], qp[MAXJ], qpp[MAXJ], A[MAXJ], B[MAXJ], G[MAXJ];
    auto at = [&](int blk, int j) { return I(x[2 * (blk * dof + j)], x[2 * (blk * dof + j) + 1]); };
    for (int j = 0; j < dof; ++j) {
      q[j] = D(at(0, j), at(1, j));
      qp[j] = D(at(1, j), at(2, j));
      qpp[j] = D(at(2, j), at(3, j));
    }
    rnea<D>(M, q, nullptr, nullptr, true, G);
    rnea<D>(M, q, nullptr, qp, false, B);
    rnea<D>(M, q, qp, qpp, false, A);
    double* o = out + (size_t)b * 6 * dof;
    for (int j = 0; j < dof; ++j) {
      I bh = B[j].d * I(0.5);
      o[2 * j] = A[j].d.lo; o[2 * j + 1] = A[j].d.hi;
      o[2 * (dof + j)] = bh.lo; o[2 * (dof + j) + 1] = bh.hi;
      o[2 * (2 * dof + j)] = G[j].d.lo; o[2 * (2 * dof + j) + 1] = G[j].d.hi;
    }
  }
  return 0;
}

int k4_rnea_abi_version() { return 1; }
double k4_selftest_next(double x, int dir) { return dir > 0 ? up(x) : dn(x); }
}

// =====================================================================================================
// Compiled K4 path producer (tm1): interval span polynomials built as in the frozen K3 producer,
// Bernstein subdivision, first-order Taylor-model torque bound per leaf, rigorous point witnesses.
// =====================================================================================================
#include <vector>
#include <stdexcept>
namespace {
inline void chk(const I& a) { if (!std::isfinite(a.lo) || !std::isfinite(a.hi) || a.lo > a.hi) throw std::range_error("interval"); }
inline I divpos(I a, I b) {  // b > 0
  if (!(b.lo > 0)) throw std::range_error("divpos");
  double p[4] = {a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi};
  return I(dn(std::min(std::min(p[0], p[1]), std::min(p[2], p[3]))), up(std::max(std::max(p[0], p[1]), std::max(p[2], p[3]))));
}
inline I rat(long long p, long long q) { return divpos(I((double)p), I((double)q)); }
using Poly = std::vector<I>;
Poly padd(const Poly& a, const Poly& b) { Poly c(std::max(a.size(), b.size()), I(0.0)); for (size_t k = 0; k < c.size(); ++k) c[k] = (k < a.size() ? a[k] : I(0.0)) + (k < b.size() ? b[k] : I(0.0)); return c; }
Poly pmul(const Poly& a, const Poly& b) { Poly c(a.size() + b.size() - 1, I(0.0)); for (size_t i = 0; i < a.size(); ++i) for (size_t j = 0; j < b.size(); ++j) c[i + j] = c[i + j] + a[i] * b[j]; return c; }
Poly pdiff(const Poly& a, I invh) { if (a.size() < 2) return {I(0.0)}; Poly c; for (size_t i = 1; i < a.size(); ++i) c.push_back(a[i] * I((double)i) * invh); return c; }
Poly compose(const Poly& p, I alpha, I beta) { Poly out{p.back()}; for (size_t k = p.size() - 1; k-- > 0;) out = padd(pmul(out, Poly{alpha, beta}), Poly{p[k]}); return out; }
unsigned binom(unsigned n, unsigned k) { unsigned r = 1; for (unsigned j = 1; j <= k; ++j) r = r * (n - j + 1) / j; return r; }
std::vector<I> to_bern(const Poly& p) {
  const unsigned n = unsigned(p.size() - 1); std::vector<I> b;
  for (unsigned k = 0; k <= n; ++k) { I s(0.0); for (unsigned i = 0; i <= k; ++i) s = s + p[i] * rat(binom(k, i), binom(n, i)); b.push_back(s); }
  return b;
}
void bsplit(const std::vector<I>& b, std::vector<I>& L, std::vector<I>& R) {
  const size_t n = b.size() - 1; std::vector<I> w = b; L.assign(n + 1, I(0.0)); R.assign(n + 1, I(0.0));
  L[0] = w[0]; R[n] = w[n];
  for (size_t r = 1; r <= n; ++r) { for (size_t i = 0; i + r <= n; ++i) { I s = w[i] + w[i + 1]; w[i] = I(s.lo * 0.5, s.hi * 0.5); chk(w[i]); } L[r] = w[0]; R[n - r] = w[n - r]; }
}
std::vector<I> elevate3(std::vector<I> b) {  // Bernstein degree elevation up to degree 3
  while (b.size() < 4) {
    const size_t m = b.size() - 1; std::vector<I> c(m + 2, I(0.0)); c[0] = b[0]; c[m + 1] = b[m];
    for (size_t k = 1; k <= m; ++k) { I w = rat((long long)k, (long long)(m + 1)); c[k] = w * b[k - 1] + (I(1.0) - w) * b[k]; }
    b = c;
  }
  return b;
}
I hull(const std::vector<I>& b) { I h = b[0]; for (auto& v : b) { h.lo = std::min(h.lo, v.lo); h.hi = std::max(h.hi, v.hi); } return h; }
I horner(const Poly& p, I u) { I acc = p.back(); for (size_t k = p.size() - 1; k-- > 0;) acc = acc * u + p[k]; return acc; }
struct Cut { long long num, den; };
inline bool clt(Cut a, Cut b) { return (__int128)a.num * b.den < (__int128)b.num * a.den; }
inline bool ceq(Cut a, Cut b) { return (__int128)a.num * b.den == (__int128)b.num * a.den; }
}  // namespace

extern "C" {
struct K4Leaf { int32_t span, depth; uint32_t bits, pad; double lo[16], hi[16]; };
struct K4Witness { int32_t span, joint, depth, where; uint32_t bits, pad; double lo, hi, limit_lo, limit_hi; };

// status: 0 CERTIFIED 1 VIOLATED_TORQUE 2 VIOLATED_X 3 UNKNOWN_DEPTH 4 UNKNOWN_NODES 5 INVALID_INPUT 6 NUMERICAL_ERROR
// 7 = leaf buffer too small (caller retries with a larger buffer)
int k4_certify_path(const float* geometry, int C, int Dj, const float* timing,
                    const double* timing_lo, const double* timing_hi,
                    int nlinks, const int* parent, const int* joint, const double* params,
                    const double* tlo, const double* thi, int max_depth, int max_nodes,
                    K4Leaf* leaves, int leaf_cap, int* leaf_count, long long* nodes_out, K4Witness* wit, int threads) {
  init_coeff();
  *leaf_count = 0; *nodes_out = 0;
  Model M;
  if (!load_model(M, nlinks, Dj, parent, joint, params)) return 5;
  if (C < 4 || Dj < 1 || Dj > MAXJ || max_depth < 0 || max_depth > 30 || max_nodes < 1) return 5;
  for (int i = 0; i < C * Dj; ++i) if (!std::isfinite(geometry[i])) return 5;
  for (int i = 0; i < 32; ++i) if (!std::isfinite(timing[i])) return 5;
  for (int j = 0; j < Dj; ++j) if (!(std::isfinite(tlo[j]) && std::isfinite(thi[j]) && tlo[j] < thi[j])) return 5;
  try {
    const int S = C - 3;
    std::vector<Cut> cuts;
    for (int i = 0; i <= S; ++i) cuts.push_back({i, S});
    for (int i = 0; i <= 29; ++i) cuts.push_back({i, 29});
    std::sort(cuts.begin(), cuts.end(), clt);
    std::vector<Cut> u;
    for (auto c : cuts) if (u.empty() || !ceq(u.back(), c)) u.push_back(c);
    const int ns = int(u.size()) - 1;
    auto g = [&](int r, int j) { return (double)geometry[r * Dj + j]; };
    struct SpanRes { int status = 0; long long nodes = 0; std::vector<K4Leaf> leaves; K4Witness w{}; };
    std::vector<SpanRes> res(ns);
    // Each span is independent: local node cap = max_nodes, results merged in span order below, so the
    // outcome is identical for every thread count (first failing span decides; cumulative node budget).
    auto run_span = [&](int sp, SpanRes& SR) {
      long long nodes = 0;
      auto fail = [&](int st) { SR.status = st; SR.nodes = nodes; };
      const Cut L = u[sp], R = u[sp + 1];
      const I h = rat(R.num * L.den - L.num * R.den, L.den * R.den);
      const I invh = divpos(I(1.0), h);
      const Cut Mid{L.num * R.den + R.num * L.den, 2 * L.den * R.den};
      const int k = std::min(int(((__int128)Mid.num * S) / Mid.den), S - 1);
      const I t0 = rat(L.num * S - (long long)k * L.den, L.den), t1 = rat(R.num * S - (long long)k * R.den, R.den);
      const I dt = t1 - t0;
      const int m = std::min(int(((__int128)Mid.num * 29) / Mid.den), 28);
      const I tau0 = rat(L.num * 29 - (long long)m * L.den, 29 * L.den);
      Poly xt(4, I(0.0));
      for (int p = 0; p < 4; ++p) for (int kk = 0; kk < 4; ++kk) { const int idx = (m * 4 + p) * 4 + kk; xt[p] = xt[p] + I(timing_lo[idx], timing_hi[idx]) * I((double)timing[m + kk]); }
      const Poly x = compose(xt, tau0, h), xp = pdiff(x, invh);
      std::vector<Poly> q(Dj), q1(Dj), q2(Dj), q3(Dj);
      for (int j = 0; j < Dj; ++j) {
        const double c0 = g(k, j), c1 = g(k + 1, j), c2 = g(k + 2, j), c3 = g(k + 3, j);
        const I six(6.0);
        Poly qt{divpos(I(c0) + I(4 * c1) + I(c2), six), divpos(I(3 * c2) - I(3 * c0), six),
                divpos(I(3 * c0) - I(6 * c1) + I(3 * c2), six), divpos(-I(c0) + I(3 * c1) - I(3 * c2) + I(c3), six)};
        q[j] = compose(qt, t0, dt); q1[j] = pdiff(q[j], invh); q2[j] = pdiff(q1[j], invh); q3[j] = pdiff(q2[j], invh);
      }
      std::vector<Poly> polys;  // order: q (Dj), q1 (Dj), q2 (Dj), q3 (Dj), x, xp
      for (int j = 0; j < Dj; ++j) polys.push_back(q[j]);
      for (int j = 0; j < Dj; ++j) polys.push_back(q1[j]);
      for (int j = 0; j < Dj; ++j) polys.push_back(q2[j]);
      for (int j = 0; j < Dj; ++j) polys.push_back(q3[j]);
      polys.push_back(x); polys.push_back(xp);
      const int NP = (int)polys.size();
      struct Node { std::vector<std::vector<I>> b; int depth; uint32_t bits; };
      std::vector<Node> queue;
      { Node r; for (auto& pp : polys) r.b.push_back(to_bern(pp)); r.depth = 0; r.bits = 0; queue.push_back(r); }
      size_t head = 0;
      while (head < queue.size()) {
        if (++nodes > max_nodes) { fail(4); return; }
        Node nd = std::move(queue[head++]);
        const double scale = std::ldexp(1.0, -nd.depth);
        const I a((double)nd.bits * scale), b(((double)nd.bits + 1.0) * scale);
        const I cu(((double)nd.bits + 0.5) * scale);  // exact dyadic
        I qb[MAXJ], q1b[MAXJ], q2b[MAXJ], q3b[MAXJ];
        for (int j = 0; j < Dj; ++j) { qb[j] = hull(nd.b[j]); q1b[j] = hull(nd.b[Dj + j]); q2b[j] = hull(nd.b[2 * Dj + j]); q3b[j] = hull(nd.b[3 * Dj + j]); }
        const I X = hull(nd.b[4 * Dj]), XP = hull(nd.b[4 * Dj + 1]);
        bool ok = X.lo >= 0;
        double tl[MAXJ], th[MAXJ];
        if (ok) {
          I qc[MAXJ], q1c[MAXJ], q2c[MAXJ];
          for (int j = 0; j < Dj; ++j) { qc[j] = horner(q[j], cu); q1c[j] = horner(q1[j], cu); q2c[j] = horner(q2[j], cu); }
          I A0[MAXJ], B0[MAXJ], G0[MAXJ], cc[MAXJ], sc[MAXJ];
          for (int j = 0; j < Dj; ++j) { cc[j] = tcos(qc[j]); sc[j] = tsin(qc[j]); }
          rnea<I>(M, qc, nullptr, nullptr, true, G0, cc, sc); rnea<I>(M, qc, nullptr, q1c, false, B0, cc, sc);
          rnea<I>(M, qc, q1c, q2c, false, A0, cc, sc);
          D dq[MAXJ], dq1[MAXJ], dq2[MAXJ], A1[MAXJ], B1[MAXJ], G1[MAXJ], dc[MAXJ], ds[MAXJ];
          for (int j = 0; j < Dj; ++j) { dq[j] = D(qb[j], q1b[j]); dq1[j] = D(q1b[j], q2b[j]); dq2[j] = D(q2b[j], q3b[j]);
                                        dc[j] = tcos(dq[j]); ds[j] = tsin(dq[j]); }
          rnea<D>(M, dq, nullptr, nullptr, true, G1, dc, ds); rnea<D>(M, dq, nullptr, dq1, false, B1, dc, ds);
          rnea<D>(M, dq, dq1, dq2, false, A1, dc, ds);
          const I Hh = h * (b - a) * I(0.5);
          const I Hb(-Hh.hi, Hh.hi);
          const std::vector<I> bx = elevate3(nd.b[4 * Dj]), bxp = elevate3(nd.b[4 * Dj + 1]);
          for (int j = 0; j < Dj; ++j) {
            const I Bh0 = B0[j] * I(0.5), Bh1 = B1[j].d * I(0.5);
            I poly = G0[j] + A0[j] * bx[0] + Bh0 * bxp[0];
            for (size_t kk = 1; kk < bx.size(); ++kk) { I ck = G0[j] + A0[j] * bx[kk] + Bh0 * bxp[kk]; poly.lo = std::min(poly.lo, ck.lo); poly.hi = std::max(poly.hi, ck.hi); }
            const I tq = poly + (G1[j].d + A1[j].d * X + Bh1 * XP) * Hb;
            tl[j] = tq.lo; th[j] = tq.hi;
            if (!(std::isfinite(tq.lo) && std::isfinite(tq.hi) && tq.lo >= tlo[j] && tq.hi <= thi[j])) ok = false;
          }
        }
        if (ok) {
          K4Leaf lf{};
          lf.span = sp; lf.depth = nd.depth; lf.bits = nd.bits; lf.pad = 0;
          for (int j = 0; j < Dj; ++j) { lf.lo[j] = tl[j]; lf.hi[j] = th[j]; }
          SR.leaves.push_back(lf);
          continue;
        }
        const I pts[3] = {a, cu, b};
        for (int w = 0; w < 3; ++w) {
          I pq[MAXJ], pq1[MAXJ], pq2[MAXJ];
          for (int j = 0; j < Dj; ++j) { pq[j] = horner(q[j], pts[w]); pq1[j] = horner(q1[j], pts[w]); pq2[j] = horner(q2[j], pts[w]); }
          const I px = horner(x, pts[w]), pxp = horner(xp, pts[w]);
          if (px.hi < 0) { K4Witness& W = SR.w; W.span = sp; W.joint = -1; W.depth = nd.depth; W.bits = nd.bits; W.where = w; W.lo = px.lo; W.hi = px.hi; fail(2); return; }
          I G[MAXJ], Bv[MAXJ], A[MAXJ], pc[MAXJ], ps[MAXJ];
          for (int j = 0; j < Dj; ++j) { pc[j] = tcos(pq[j]); ps[j] = tsin(pq[j]); }
          rnea<I>(M, pq, nullptr, nullptr, true, G, pc, ps); rnea<I>(M, pq, nullptr, pq1, false, Bv, pc, ps);
          rnea<I>(M, pq, pq1, pq2, false, A, pc, ps);
          for (int j = 0; j < Dj; ++j) {
            const I t = G[j] + A[j] * px + Bv[j] * I(0.5) * pxp;
            if (t.lo > thi[j] || t.hi < tlo[j]) {
              K4Witness& W = SR.w; W.span = sp; W.joint = j; W.depth = nd.depth; W.bits = nd.bits; W.where = w;
              W.lo = t.lo; W.hi = t.hi; W.limit_lo = tlo[j]; W.limit_hi = thi[j]; fail(1); return;
            }
          }
        }
        if (nd.depth >= max_depth) { fail(3); return; }
        Node Lc, Rc; Lc.depth = Rc.depth = nd.depth + 1; Lc.bits = nd.bits << 1; Rc.bits = (nd.bits << 1) | 1u;
        for (int r = 0; r < NP; ++r) { std::vector<I> l, rr; bsplit(nd.b[r], l, rr); Lc.b.push_back(l); Rc.b.push_back(rr); }
        queue.push_back(std::move(Lc)); queue.push_back(std::move(Rc));
      }
      SR.nodes = nodes;
    };
    bool err = false;
#pragma omp parallel for schedule(dynamic) num_threads(threads > 0 ? threads : 1)
    for (int sp = 0; sp < ns; ++sp) {
      try { run_span(sp, res[sp]); } catch (const std::exception&) {
#pragma omp critical
        err = true;
      }
    }
    if (err) return 6;
    long long nodes = 0;
    for (int sp = 0; sp < ns; ++sp) {
      nodes += res[sp].nodes;
      if (nodes > max_nodes) { *nodes_out = nodes; return 4; }
      for (auto& lf : res[sp].leaves) { if (*leaf_count >= leaf_cap) { *nodes_out = nodes; return 7; } leaves[(*leaf_count)++] = lf; }
      if (res[sp].status != 0) { *wit = res[sp].w; *nodes_out = nodes; return res[sp].status; }
    }
    *nodes_out = nodes;
    return 0;
  } catch (const std::exception&) {
    return 6;
  }
}
int k4_leaf_size() { return (int)sizeof(K4Leaf); }
int k4_witness_size() { return (int)sizeof(K4Witness); }
}  // extern "C"

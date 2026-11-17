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
inline double dn(double x) { return std::nextafter(x, -INF); }
inline double up(double x) { return std::nextafter(x, INF); }

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
  I acc = C[ORDER];
  for (int k = ORDER - 1; k >= 0; --k) acc = acc * x2 + C[k];
  if (!cosine) acc = acc * x;
  double lo = dn(dn(acc.lo - REM) - r), hi = up(up(acc.hi + REM) + r);
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
void rnea(const Model& M, const T* q, const T* v, const T* a, bool gravity, T* tau) {
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
      T cq = tcos(q[j]), sq = tsin(q[j]);
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
}

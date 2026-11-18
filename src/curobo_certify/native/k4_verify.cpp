// Independent compiled MODEL_K4 verifier (curobo_certify). Separate from the producer (k4_rnea.cpp):
//  * own outward interval type (successor/predecessor by bit manipulation);
//  * span polynomials enclosed on each claimed leaf by the CENTRED (mean-value) form
//    p(u) in p(c) + p'([a,b]) (u - c), with p' evaluated by interval Horner - not Bernstein hulls;
//  * WORLD-FRAME Newton-Euler: world rotations/points by forward kinematics, world inertias R Ic R^T,
//    explicit subtree sums tau_j = z_j . sum_k [(c_k - o_j) x F_k + N_k]  (producer: link-frame recursion);
//  * box-mode torque G + A X + B X' (no Taylor model), refined by its OWN bisection up to `refine` levels;
//  * exact dyadic partition check per span and recorded-box midpoint containment check.
// Model parameter intervals come from the verifier's own contract parser (k4_verify.WorldFrameDynamics).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>
#include <stdexcept>

#if defined(__FAST_MATH__)
#error "k4_verify must not be compiled with -ffast-math"
#endif

namespace kv {
constexpr double INF = std::numeric_limits<double>::infinity();
inline double nu(double x) {
  if (!(x == x) || x == INF) return x;
  if (x == 0) return 4.9406564584124654e-324;
  uint64_t b; std::memcpy(&b, &x, 8); b = x > 0 ? b + 1 : b - 1;
  double r; std::memcpy(&r, &b, 8); return r;
}
inline double nd(double x) { return -nu(-x); }
struct V { double l, h; };
inline V c(double x) { return {x, x}; }
inline V operator+(V a, V b) { return {nd(a.l + b.l), nu(a.h + b.h)}; }
inline V operator-(V a, V b) { return {nd(a.l - b.h), nu(a.h - b.l)}; }
inline V operator-(V a) { return {-a.h, -a.l}; }
inline V operator*(V a, V b) {
  double p[4] = {a.l * b.l, a.l * b.h, a.h * b.l, a.h * b.h};
  double lo = INF, hi = -INF; bool nan = false;
  for (double x : p) { if (x != x) nan = true; lo = std::min(lo, x); hi = std::max(hi, x); }
  if (nan) return {-INF, INF};
  return {nd(lo), nu(hi)};
}
inline V vdiv(V a, V b) {
  if (!(b.l > 0)) throw std::range_error("div");
  double p[4] = {a.l / b.l, a.l / b.h, a.h / b.l, a.h / b.h};
  return {nd(std::min(std::min(p[0], p[1]), std::min(p[2], p[3]))), nu(std::max(std::max(p[0], p[1]), std::max(p[2], p[3])))};
}
inline V hullv(V a, V b) { return {std::min(a.l, b.l), std::max(a.h, b.h)}; }
inline V vsqr(V a) { double x = a.l * a.l, y = a.h * a.h; return {(a.l <= 0 && a.h >= 0) ? 0.0 : nd(std::min(x, y)), nu(std::max(x, y))}; }

// sin / cos: Taylor (order 30) at the midpoint, Lagrange remainder for |m| <= 8 (< 3e-29, bound 1e-27),
// 1-Lipschitz radius widening, clamp; |m| > 8 -> [-1, 1].
V tc[31], ts[31];
bool tready = false;
void tinit() {
  if (tready) return;
  for (int e = 0; e < 2; ++e)
    for (int k = 0; k <= 30; ++k) {
      V x = c(1.0);
      for (int n = 2; n <= 2 * k + e; ++n) x = vdiv(x, c((double)n));
      if (k % 2) x = -x;
      (e ? ts : tc)[k] = x;
    }
  tready = true;
}
V trig(V a, bool cosine) {
  const double m = 0.5 * a.l + 0.5 * a.h;
  if (!(std::fabs(m) <= 8.0)) return {-1.0, 1.0};
  const double r = nu(std::max(nu(a.h - m), nu(m - a.l)));
  V x = c(m), x2 = vsqr(x);
  const V* C = cosine ? tc : ts;
  V acc = C[30];
  for (int k = 29; k >= 0; --k) acc = acc * x2 + C[k];
  if (!cosine) acc = acc * x;
  return {std::max(-1.0, nd(nd(acc.l - 1e-27) - r)), std::min(1.0, nu(nu(acc.h + 1e-27) + r))};
}

constexpr int NL = 32, NJ = 16;
struct Model {
  int n, dof, parent[NL], joint[NL];
  V Rf[NL][3][3], p[NL][3], m[NL], cm[NL][3], I[NL][3][3], g[3];
  bool sub[NL][NL];
};

void cross(const V a[3], const V b[3], V o[3]) { o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0]; }

// World-frame inverse dynamics; v / a may be null (zero).
void world_tau(const Model& M, const V* q, const V* v, const V* a, bool gravity, V* tau) {
  V R[NL][3][3], o[NL][3], w[NL][3], wd[NL][3], ao[NL][3], z[NL][3];
  for (int r = 0; r < 3; ++r) for (int k = 0; k < 3; ++k) R[0][r][k] = c(r == k ? 1.0 : 0.0);
  for (int k = 0; k < 3; ++k) { o[0][k] = c(0); w[0][k] = c(0); wd[0][k] = c(0); ao[0][k] = c(0); }
  for (int i = 1; i < M.n; ++i) {
    const int pa = M.parent[i], j = M.joint[i];
    V Rl[3][3];
    if (j >= 0) {
      V cq = trig(q[j], true), sq = trig(q[j], false);
      for (int r = 0; r < 3; ++r) { Rl[r][0] = cq * M.Rf[i][r][0] + sq * M.Rf[i][r][1]; Rl[r][1] = cq * M.Rf[i][r][1] - sq * M.Rf[i][r][0]; Rl[r][2] = M.Rf[i][r][2]; }
    } else {
      for (int r = 0; r < 3; ++r) for (int k = 0; k < 3; ++k) Rl[r][k] = M.Rf[i][r][k];
    }
    for (int r = 0; r < 3; ++r) for (int k = 0; k < 3; ++k) R[i][r][k] = R[pa][r][0] * Rl[0][k] + R[pa][r][1] * Rl[1][k] + R[pa][r][2] * Rl[2][k];
    V dp[3];
    for (int r = 0; r < 3; ++r) dp[r] = R[pa][r][0] * M.p[i][0] + R[pa][r][1] * M.p[i][1] + R[pa][r][2] * M.p[i][2];
    for (int r = 0; r < 3; ++r) o[i][r] = o[pa][r] + dp[r];
    V t[3]; cross(wd[pa], dp, t);
    for (int r = 0; r < 3; ++r) ao[i][r] = ao[pa][r] + t[r];
    if (v) { V u[3], u2[3]; cross(w[pa], dp, u); cross(w[pa], u, u2); for (int r = 0; r < 3; ++r) ao[i][r] = ao[i][r] + u2[r]; }
    if (j >= 0) {
      for (int r = 0; r < 3; ++r) z[i][r] = R[i][r][2];
      const V qd = v ? v[j] : c(0), qdd = a ? a[j] : c(0);
      for (int r = 0; r < 3; ++r) { w[i][r] = v ? w[pa][r] + z[i][r] * qd : c(0); wd[i][r] = wd[pa][r] + z[i][r] * qdd; }
      if (v) { V zq[3] = {z[i][0] * qd, z[i][1] * qd, z[i][2] * qd}, u[3]; cross(w[pa], zq, u); for (int r = 0; r < 3; ++r) wd[i][r] = wd[i][r] + u[r]; }
    } else {
      for (int r = 0; r < 3; ++r) { w[i][r] = w[pa][r]; wd[i][r] = wd[pa][r]; }
    }
  }
  V cw[NL][3], F[NL][3], N[NL][3];
  for (int k = 1; k < M.n; ++k) {
    V rc[3];
    for (int r = 0; r < 3; ++r) rc[r] = R[k][r][0] * M.cm[k][0] + R[k][r][1] * M.cm[k][1] + R[k][r][2] * M.cm[k][2];
    for (int r = 0; r < 3; ++r) cw[k][r] = o[k][r] + rc[r];
    V acc[3], t[3]; cross(wd[k], rc, t);
    for (int r = 0; r < 3; ++r) acc[r] = ao[k][r] + t[r];
    if (v) { V u[3], u2[3]; cross(w[k], rc, u); cross(w[k], u, u2); for (int r = 0; r < 3; ++r) acc[r] = acc[r] + u2[r]; }
    if (gravity) for (int r = 0; r < 3; ++r) acc[r] = acc[r] - M.g[r];
    for (int r = 0; r < 3; ++r) F[k][r] = acc[r] * M.m[k];
    auto Iw = [&](const V u[3], V out[3]) {
      V bu[3], ib[3];
      for (int cc = 0; cc < 3; ++cc) bu[cc] = R[k][0][cc] * u[0] + R[k][1][cc] * u[1] + R[k][2][cc] * u[2];
      for (int r = 0; r < 3; ++r) ib[r] = bu[0] * M.I[k][r][0] + bu[1] * M.I[k][r][1] + bu[2] * M.I[k][r][2];
      for (int r = 0; r < 3; ++r) out[r] = R[k][r][0] * ib[0] + R[k][r][1] * ib[1] + R[k][r][2] * ib[2];
    };
    Iw(wd[k], N[k]);
    if (v) { V iw[3], x[3]; Iw(w[k], iw); cross(w[k], iw, x); for (int r = 0; r < 3; ++r) N[k][r] = N[k][r] + x[r]; }
  }
  for (int i = 1; i < M.n; ++i) {
    if (M.joint[i] < 0) continue;
    V s[3] = {c(0), c(0), c(0)};
    for (int k = 1; k < M.n; ++k) {
      if (!M.sub[i][k]) continue;
      V arm[3] = {cw[k][0] - o[i][0], cw[k][1] - o[i][1], cw[k][2] - o[i][2]}, t[3];
      cross(arm, F[k], t);
      for (int r = 0; r < 3; ++r) s[r] = s[r] + t[r] + N[k][r];
    }
    tau[M.joint[i]] = z[i][0] * s[0] + z[i][1] * s[1] + z[i][2] * s[2];
  }
}

using Pl = std::vector<V>;
Pl add(const Pl& a, const Pl& b) { Pl r(std::max(a.size(), b.size()), c(0)); for (size_t k = 0; k < r.size(); ++k) r[k] = (k < a.size() ? a[k] : c(0)) + (k < b.size() ? b[k] : c(0)); return r; }
Pl mulp(const Pl& a, const Pl& b) { Pl r(a.size() + b.size() - 1, c(0)); for (size_t i = 0; i < a.size(); ++i) for (size_t j = 0; j < b.size(); ++j) r[i + j] = r[i + j] + a[i] * b[j]; return r; }
Pl deriv(const Pl& a, V s) { if (a.size() < 2) return {c(0)}; Pl r; for (size_t i = 1; i < a.size(); ++i) r.push_back(a[i] * c((double)i) * s); return r; }
Pl affine(const Pl& p, V al, V be) { Pl r{p.back()}; for (size_t k = p.size() - 1; k-- > 0;) r = add(mulp(r, Pl{al, be}), Pl{p[k]}); return r; }
V horn(const Pl& p, V u) { V acc = p.back(); for (size_t k = p.size() - 1; k-- > 0;) acc = acc * u + p[k]; return acc; }
// centred form enclosure of p over [a,b] (u real in [a,b])
V centred(const Pl& p, const Pl& dp, double a, double b) {
  const double m = 0.5 * a + 0.5 * b;  // exact dyadic
  V pc = horn(p, c(m)), d = horn(dp, V{a, b});
  return pc + d * V{nd(a - m), nu(b - m)};
}
}  // namespace kv

extern "C" {
struct KVLeaf { int32_t span, depth; uint32_t bits, pad; double lo[16], hi[16]; };
// return: 0 PASS, 1 partition invalid, 2 leaf not verified within refinement budget, 3 recorded box violates
// midpoint containment, 4 invalid input / numerical error. counters[0] verified boxes, [1] refined splits.
int k4v_verify(const float* geometry, int C, int Dj, const float* timing, const double* tlo_tab, const double* thi_tab,
               int nlinks, const int* parent, const int* joint, const double* params, const double* tlo,
               const double* thi, const KVLeaf* leaves, int nleaves, int refine, long long* counters, int threads) {
  using namespace kv;
  tinit();
  counters[0] = counters[1] = 0;
  try {
    if (C < 4 || Dj < 1 || Dj > NJ || nlinks > NL || refine < 0 || refine > 20) return 4;
    for (int j = 0; j < Dj; ++j) if (!(std::isfinite(tlo[j]) && std::isfinite(thi[j]) && tlo[j] < thi[j])) return 4;
    Model M; M.n = nlinks; M.dof = Dj;
    for (int i = 0; i < nlinks; ++i) {
      M.parent[i] = parent[i]; M.joint[i] = joint[i];
      const double* b = params + 50 * i; int k = 0;
      for (int r = 0; r < 3; ++r) for (int s = 0; s < 3; ++s, ++k) M.Rf[i][r][s] = {b[2 * k], b[2 * k + 1]};
      for (int r = 0; r < 3; ++r, ++k) M.p[i][r] = {b[2 * k], b[2 * k + 1]};
      M.m[i] = {b[2 * k], b[2 * k + 1]}; ++k;
      for (int r = 0; r < 3; ++r, ++k) M.cm[i][r] = {b[2 * k], b[2 * k + 1]};
      for (int r = 0; r < 3; ++r) for (int s = 0; s < 3; ++s, ++k) M.I[i][r][s] = {b[2 * k], b[2 * k + 1]};
    }
    for (int k = 0; k < 3; ++k) M.g[k] = {params[50 * nlinks + 2 * k], params[50 * nlinks + 2 * k + 1]};
    for (int i = 0; i < nlinks; ++i) for (int k = 0; k < nlinks; ++k) M.sub[i][k] = false;
    for (int k = 0; k < nlinks; ++k) { int i = k; while (true) { M.sub[i][k] = true; if (i == 0) break; if (M.parent[i] >= i) return 4; i = M.parent[i]; } }
    // ---- common spans (own construction) ----
    const int S = C - 3;
    std::vector<std::pair<long long, long long>> cuts;
    for (int i = 0; i <= S; ++i) cuts.push_back({i, S});
    for (int i = 0; i <= 29; ++i) cuts.push_back({i, 29});
    auto less = [](std::pair<long long, long long> a, std::pair<long long, long long> b) { return (__int128)a.first * b.second < (__int128)b.first * a.second; };
    std::sort(cuts.begin(), cuts.end(), less);
    std::vector<std::pair<long long, long long>> u;
    for (auto x : cuts) if (u.empty() || less(u.back(), x)) u.push_back(x);
    const int ns = (int)u.size() - 1;
    // ---- partition check (exact, depth <= 30) ----
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> segs(ns);
    for (int i = 0; i < nleaves; ++i) {
      const KVLeaf& L = leaves[i];
      if (L.span < 0 || L.span >= ns || L.depth < 0 || L.depth > 30 || (uint64_t)L.bits >= (1ull << L.depth)) return 1;
      const uint64_t w = 1ull << (30 - L.depth);
      segs[L.span].push_back({(uint64_t)L.bits * w, (uint64_t)L.bits * w + w});
    }
    for (int sp = 0; sp < ns; ++sp) {
      auto& v = segs[sp];
      std::sort(v.begin(), v.end());
      uint64_t pos = 0;
      for (auto& s : v) { if (s.first != pos) return 1; pos = s.second; }
      if (pos != (1ull << 30)) return 1;
    }
    auto rv = [](long long p, long long q) { return vdiv(c((double)p), c((double)q)); };
    auto g = [&](int r, int j) { return (double)geometry[r * Dj + j]; };
    std::vector<int> code(ns, 0);
    std::vector<long long> c0(ns, 0), c1(ns, 0);
    auto run_span = [&](int sp) -> int {
      long long* ctr0 = &c0[sp]; long long* ctr1 = &c1[sp];
      const auto L = u[sp], R = u[sp + 1];
      const V h = rv(R.first * L.second - L.first * R.second, L.second * R.second), invh = vdiv(c(1), h);
      const long long mn = L.first * R.second + R.first * L.second, md = 2 * L.second * R.second;
      const int k = std::min((int)(((__int128)mn * S) / md), S - 1);
      const V t0 = rv(L.first * S - (long long)k * L.second, L.second), t1 = rv(R.first * S - (long long)k * R.second, R.second);
      const int m = std::min((int)(((__int128)mn * 29) / md), 28);
      const V tau0 = rv(L.first * 29 - (long long)m * L.second, 29 * L.second);
      Pl xt(4, c(0));
      for (int p = 0; p < 4; ++p) for (int kk = 0; kk < 4; ++kk) { const int idx = (m * 4 + p) * 4 + kk; xt[p] = xt[p] + V{tlo_tab[idx], thi_tab[idx]} * c((double)timing[m + kk]); }
      const Pl x = affine(xt, tau0, h), x1 = deriv(x, invh), x2 = deriv(x1, invh);
      std::vector<Pl> q(Dj), q1(Dj), q2(Dj), q3(Dj);
      for (int j = 0; j < Dj; ++j) {
        const double c0 = g(k, j), c1 = g(k + 1, j), c2 = g(k + 2, j), c3 = g(k + 3, j);
        Pl qt{vdiv(c(c0) + c(4 * c1) + c(c2), c(6)), vdiv(c(3 * c2) - c(3 * c0), c(6)),
              vdiv(c(3 * c0) - c(6 * c1) + c(3 * c2), c(6)), vdiv(-c(c0) + c(3 * c1) - c(3 * c2) + c(c3), c(6))};
        q[j] = affine(qt, t0, t1 - t0); q1[j] = deriv(q[j], invh); q2[j] = deriv(q1[j], invh); q3[j] = deriv(q2[j], invh);
      }
      // verify one u-interval, refining on failure
      std::vector<std::pair<std::pair<double, double>, int>> work;
      for (int i = 0; i < nleaves; ++i) if (leaves[i].span == sp) {
        const KVLeaf& Lf = leaves[i];
        const double a = std::ldexp((double)Lf.bits, -Lf.depth), b = std::ldexp((double)Lf.bits + 1.0, -Lf.depth);
        // recorded-box midpoint containment (rejects zeroed / too-narrow recorded boxes)
        const double mu = 0.5 * a + 0.5 * b;
        V pq[NJ], pq1[NJ], pq2[NJ], G[NJ], Bv[NJ], A[NJ];
        for (int j = 0; j < Dj; ++j) { pq[j] = horn(q[j], c(mu)); pq1[j] = horn(q1[j], c(mu)); pq2[j] = horn(q2[j], c(mu)); }
        const V px = horn(x, c(mu)), pxp = horn(x1, c(mu));
        world_tau(M, pq, nullptr, nullptr, true, G); world_tau(M, pq, nullptr, pq1, false, Bv); world_tau(M, pq, pq1, pq2, false, A);
        for (int j = 0; j < Dj; ++j) {
          const V t = G[j] + A[j] * px + Bv[j] * c(0.5) * pxp;
          const double tol_l = 1e-9 * std::max(1.0, std::fabs(Lf.lo[j])), tol_h = 1e-9 * std::max(1.0, std::fabs(Lf.hi[j]));
          if (!(t.l >= Lf.lo[j] - tol_l && t.h <= Lf.hi[j] + tol_h)) return 3;
          if (!(Lf.lo[j] >= tlo[j] && Lf.hi[j] <= thi[j])) return 3;
        }
        work.push_back({{a, b}, 0});
      }
      while (!work.empty()) {
        auto it = work.back(); work.pop_back();
        const double a = it.first.first, b = it.first.second;
        V qb[NJ], q1b[NJ], q2b[NJ], G[NJ], Bv[NJ], A[NJ];
        // centred form needs the u-derivative (q1 is the s-derivative): differentiate in u directly
        for (int j = 0; j < Dj; ++j) {
          Pl dqu = deriv(q[j], c(1.0)), dq1u = deriv(q1[j], c(1.0)), dq2u = deriv(q2[j], c(1.0));
          qb[j] = centred(q[j], dqu, a, b); q1b[j] = centred(q1[j], dq1u, a, b); q2b[j] = centred(q2[j], dq2u, a, b);
        }
        const V X = centred(x, deriv(x, c(1.0)), a, b), XP = centred(x1, deriv(x1, c(1.0)), a, b);
        bool ok = X.l >= 0;
        if (ok) {
          world_tau(M, qb, nullptr, nullptr, true, G); world_tau(M, qb, nullptr, q1b, false, Bv); world_tau(M, qb, q1b, q2b, false, A);
          for (int j = 0; j < Dj && ok; ++j) {
            const V t = G[j] + A[j] * X + Bv[j] * c(0.5) * XP;
            if (!(std::isfinite(t.l) && std::isfinite(t.h) && t.l >= tlo[j] && t.h <= thi[j])) ok = false;
          }
        }
        ++*ctr0;
        if (ok) continue;
        if (it.second >= refine) return 2;
        ++*ctr1;
        const double mid = 0.5 * a + 0.5 * b;
        work.push_back({{a, mid}, it.second + 1}); work.push_back({{mid, b}, it.second + 1});
      }
      return 0;
    };
    bool err = false;
#pragma omp parallel for schedule(dynamic) num_threads(threads > 0 ? threads : 1)
    for (int sp = 0; sp < ns; ++sp) {
      try { code[sp] = run_span(sp); } catch (const std::exception&) {
#pragma omp critical
        err = true;
      }
    }
    if (err) return 4;
    for (int sp = 0; sp < ns; ++sp) { counters[0] += c0[sp]; counters[1] += c1[sp]; }
    for (int sp = 0; sp < ns; ++sp) if (code[sp]) return code[sp];
    return 0;
  } catch (const std::exception&) {
    return 4;
  }
}
int k4v_leaf_size() { return (int)sizeof(KVLeaf); }
}

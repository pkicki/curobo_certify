/* Exact-rational reconstruction of the serialized TimeOpt model in the browser.
 * float32 controls are decoded bit-exactly into BigInt rationals; path q(s) is the
 * uniform cardinal cubic B-spline, x(s) the clamped cubic timing spline (exact basis
 * table supplied as rational strings). Residuals are built exactly and handed to
 * BernsteinProof (proof_core.js). Floats are used for plotting only.
 * Scope: EXACT_SERIALIZED_MODEL. Not a native FP32 or whole-path certificate. */
(function (global, factory) {
  const api = factory(global.BernsteinProof || (typeof require !== 'undefined' ? require('./proof_core.js') : null));
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.ModelCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (P) {
  'use strict';
  const Z = P.R(0), ONE = P.R(1);
  function f32(bits) {  // exact rational of a float32 bit pattern (uint32)
    const sign = bits >>> 31 ? -1n : 1n, exp = (bits >>> 23) & 0xff, man = BigInt(bits & 0x7fffff);
    if (exp === 0xff) throw Error('non-finite float32');
    let m = exp === 0 ? man : man | 0x800000n, e = (exp === 0 ? 1 : exp) - 127 - 23;
    return e >= 0 ? P.R(sign * m * (1n << BigInt(e))) : P.R(sign * m, 1n << BigInt(-e));
  }
  function decode(b64) {  // base64 little-endian float32 -> {bits:Uint32Array, values:Float32Array}
    const raw = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
    const buf = new ArrayBuffer(raw.length), u8 = new Uint8Array(buf);
    for (let i = 0; i < raw.length; i++) u8[i] = raw.charCodeAt(i);
    return {bits: new Uint32Array(buf), values: new Float32Array(buf)};
  }
  const padd = (a, b) => Array.from({length: Math.max(a.length, b.length)}, (_, i) => P.add(a[i] || Z, b[i] || Z));
  const pscale = (a, c) => a.map(x => P.mul(x, c));
  function pmul(a, b) {
    const c = Array.from({length: a.length + b.length - 1}, () => Z);
    a.forEach((x, i) => b.forEach((y, j) => { c[i + j] = P.add(c[i + j], P.mul(x, y)); }));
    return c;
  }
  const pder = (a, invh) => a.length < 2 ? [Z] : a.slice(1).map((x, i) => P.mul(P.mul(x, P.R(i + 1)), invh));
  function compose(p, alpha, beta) {  // p(alpha + beta u), degree-exact
    let out = [p[p.length - 1]];
    for (let k = p.length - 2; k >= 0; k--) out = padd(pmul(out, [alpha, beta]), [p[k]]);
    return out;
  }
  function binom(n, k) { let r = 1n; for (let j = 1n; j <= BigInt(k); j++) r = r * (BigInt(n) - j + 1n) / j; return r; }
  function bernstein(power) {
    const n = power.length - 1;
    return Array.from({length: n + 1}, (_, k) => {
      let s = Z;
      for (let i = 0; i <= k; i++) s = P.add(s, P.mul(power[i], P.R(binom(k, i), binom(n, i))));
      return s;
    });
  }
  const evalp = (p, u) => p.reduceRight((acc, c) => P.add(P.mul(acc, u), c), Z);
  function evalf(p, u) { let r = 0; for (let k = p.length - 1; k >= 0; k--) r = r * u + P.num(p[k]); return r; }

  function commonSpans(C) {
    const S = C - 3, cuts = [];
    for (let i = 0; i <= S; i++) cuts.push(P.R(i, S));
    for (let i = 0; i <= 29; i++) cuts.push(P.R(i, 29));
    cuts.sort(P.cmp);
    const u = cuts.filter((c, i) => i === 0 || P.cmp(c, cuts[i - 1]) !== 0);
    return u.slice(0, -1).map((l, i) => [l, u[i + 1]]);
  }
  /* model = {C, D, geometry: rationals [C][D], timing: rationals [32], table: rationals [29][4 asc p][4 k]} */
  function build(path, tableStrings) {
    const g = decode(path.geometry_f32), t = decode(path.timing_f32);
    const C = path.shape[0], D = path.shape[1];
    const geometry = Array.from({length: C}, (_, r) => Array.from({length: D}, (_, j) => f32(g.bits[r * D + j])));
    const timing = Array.from(t.bits, f32);
    const table = tableStrings.map(m => m.map(p => p.map(P.parse)));
    return {C, D, S: C - 3, geometry, timing, table, floats: {geometry: g.values, timing: t.values}, spans: commonSpans(C)};
  }
  function spanPolys(model, si) {
    const [L, R] = model.spans[si], S = model.S, h = P.sub(R, L), mid = P.div(P.add(L, R), P.R(2));
    const k = Math.min(Number((mid.n * BigInt(S)) / mid.d), S - 1);
    const m = Math.min(Number((mid.n * 29n) / mid.d), 28);
    const t0 = P.sub(P.mul(L, P.R(S)), P.R(k)), dt = P.mul(h, P.R(S));
    const q = [];
    for (let j = 0; j < model.D; j++) {
      const c = [0, 1, 2, 3].map(r => model.geometry[k + r][j]);
      const six = P.R(6);
      const qt = [P.div(P.add(P.add(c[0], P.mul(P.R(4), c[1])), c[2]), six),
                  P.div(P.sub(P.mul(P.R(3), c[2]), P.mul(P.R(3), c[0])), six),
                  P.div(P.add(P.sub(P.mul(P.R(3), c[0]), P.mul(P.R(6), c[1])), P.mul(P.R(3), c[2])), six),
                  P.div(P.add(P.sub(P.add(P.sub(Z, c[0]), P.mul(P.R(3), c[1])), P.mul(P.R(3), c[2])), c[3]), six)];
      q.push(compose(qt, t0, dt));
    }
    const xt = [0, 1, 2, 3].map(p => [0, 1, 2, 3].reduce((acc, kk) => P.add(acc, P.mul(model.table[m][p][kk], model.timing[m + kk])), Z));
    const x = compose(xt, P.sub(L, P.R(m, 29)), h);
    return {L, R, h, k, m, q, x};
  }
  const FAMILIES = ['positive_x', 'velocity', 'acceleration_plus', 'acceleration_minus', 'jerk'];
  function residual(model, si, family, joint, limits) {
    const sp = spanPolys(model, si), invh = P.div(ONE, sp.h);
    const x = sp.x, x1 = pder(x, invh), x2 = pder(x1, invh);
    if (family === 'positive_x') return {sp, power: pscale(x, P.R(-1))};
    const q = sp.q[joint], q1 = pder(q, invh), q2 = pder(q1, invh), q3 = pder(q2, invh);
    // limits MUST be exact rational strings (e.g. str(Fraction(float32_limit))): a decimal
    // rendering such as "2.1" is NOT the binary value used by the native solver.
    const lim = name => { const v = limits[name][joint]; if (typeof v !== 'string') throw Error('limits must be exact rational strings'); return P.parse(v); };
    if (family === 'velocity') {
      const v = lim('velocity');
      return {sp, power: padd(pscale(pmul(pmul(q1, q1), x), P.div(ONE, P.mul(v, v))), [P.R(-1)])};
    }
    const acc = padd(pmul(q2, x), pscale(pmul(q1, x1), P.R(1, 2)));
    if (family === 'acceleration_plus') return {sp, power: padd(pscale(acc, P.div(ONE, lim('acceleration'))), [P.R(-1)])};
    if (family === 'acceleration_minus') return {sp, power: padd(pscale(acc, P.div(P.R(-1), lim('acceleration'))), [P.R(-1)])};
    const ell = padd(padd(pmul(q3, x), pscale(pmul(q2, x1), P.R(3, 2))), pscale(pmul(q1, x2), P.R(1, 2)));
    const j = lim('jerk');
    return {sp, power: padd(pscale(pmul(x, pmul(ell, ell)), P.div(ONE, P.mul(j, j))), [P.R(-1)])};
  }
  /* Float evaluation (plots only) of physical v, a, j at progress s for one joint. */
  function physical(model, joint, s) {
    const si = Math.min(model.spans.length - 1, model.spans.findIndex(([, R]) => s <= P.num(R)));
    const sp = model._cache && model._cache[si] || (model._cache = model._cache || {}, model._cache[si] = spanPolys(model, si));
    const hn = P.num(sp.h), u = (s - P.num(sp.L)) / hn;
    const d = (p, o) => { let c = p.map(P.num); for (let r = 0; r < o; r++) c = c.slice(1).map((v, i) => v * (i + 1) / hn); return c; };
    const ev = c => c.reduceRight((a, v) => a * u + v, 0);
    const X = [ev(d(sp.x, 0)), ev(d(sp.x, 1)), ev(d(sp.x, 2))];
    const Q = [ev(d(sp.q[joint], 0)), ev(d(sp.q[joint], 1)), ev(d(sp.q[joint], 2)), ev(d(sp.q[joint], 3))];
    const speed = Math.sqrt(Math.max(X[0], 0));
    return {q: Q[0], x: X[0], v: Q[1] * speed, a: Q[2] * X[0] + 0.5 * Q[1] * X[1],
            j: speed * (Q[3] * X[0] + 1.5 * Q[2] * X[1] + 0.5 * Q[1] * X[2])};
  }
  return {f32, decode, commonSpans, build, spanPolys, residual, bernstein, evalp, evalf, physical, FAMILIES};
});

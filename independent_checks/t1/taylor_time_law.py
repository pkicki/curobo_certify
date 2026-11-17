"""Independent rigorous enclosure of T(s) = int_0^s x(u)^{-1/2} du and of its inverse.

Written for the R0 review from the mathematics only (no formulas or code from
curobo_certify.time_law_grid / time_law_enclosures / native_error_bound).

Method (all arithmetic exact: Python integers and Fractions; no floating point):

Cell [a, b] inside one polynomial piece, centre c, half width r, tau = (s-c)/r in [-1,1].
X(tau) = x(c + r tau) is an exact rational cubic.  Pick a positive dyadic x0 ~ X(0) and write
    X(tau) = x0 (1 + u(tau)),   u = (X - x0)/x0,   rho := sum_i |u_i| >= max_{|tau|<=1} |u(tau)|.
If rho < 1 then X > 0 on the cell and, by Taylor's theorem with Lagrange remainder for
g(u) = (1+u)^{-1/2}  (g^{(n)}(u)/n! = beta_n (1+u)^{-1/2-n},  beta_n = (-1)^n C(2n,n)/4^n):
    g(u) = sum_{n<=K} beta_n u^n + E,   |E| <= |beta_{K+1}| rho^{K+1} (1-rho)^{-(K+3/2)}
                                            <= |beta_{K+1}| rho^{K+1} (1-rho)^{-(K+2)} =: Rb.
Hence for tau_s in [-1, 1]
    int_a^s x^{-1/2} = r x0^{-1/2} ( A(tau_s) + theta (tau_s + 1) Rb ),  |theta| <= 1,
with A(tau) = int_{-1}^{tau} sum_n beta_n u(t)^n dt an exact rational polynomial.
x0^{-1/2} = sqrt(den/num) is enclosed by integer isqrt (both directions checked by integer
comparisons).  Cell bounds are rounded outward to integer multiples of 2^-P (exact
integers), and prefix sums are exact integer sums -- there is no floating-point
summation error to account for.  K is chosen per cell (<= KMAX) so that Rb <= 2^-TOL_BITS;
otherwise the cell is bisected (up to MAX_DEPTH).  A cell where X(0) <= 0 or rho >= 1 at
MAX_DEPTH yields status UNKNOWN_POSITIVITY (fail closed).

The inverse s*(t) (defined as 1 for t >= T(1): the trajectory holds its end) is bracketed
by bisection on dyadic s using only the two monotonicity facts
    T_hi(s) <= t  =>  s <= s*(t),        T_lo(s) >= t  =>  s*(t) <= s.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from fractions import Fraction as Q
from math import comb, isqrt

from . import exact_spline as ES

P_BITS = 200          # dyadic output grid 2^-P for all bounds (resolves subnormal-size times)
TOL_BITS = 72         # per-cell relative truncation target
KMAX = 14
MAX_DEPTH = 40


# ----------------------------------------------------------------- polynomials

def _compose_affine(c, alpha: Q, beta: Q):
    """p(alpha + beta*tau) for ascending coefficients c (exact)."""
    out = [Q(0)]
    for coef in reversed(list(c)):
        new = [Q(0)] * (len(out) + 1)
        for k, v in enumerate(out):
            new[k] += v * alpha
            new[k + 1] += v * beta
        new[0] += coef
        out = new
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _to_int_poly(c):
    """Fractions -> (integer coefficients, common positive denominator)."""
    from math import lcm
    D = 1
    for v in c:
        D = lcm(D, v.denominator)
    return [int(v * D) for v in c], D


def _imul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def beta(n: int) -> Q:
    return Q((-1) ** n * comb(2 * n, n), 4 ** n)


def inv_sqrt_bracket(num: int, den: int, bits: int = P_BITS) -> tuple[Q, Q]:
    """[lo, hi] with lo <= sqrt(den/num) <= hi, num, den > 0, width <= 2^-bits."""
    if num <= 0 or den <= 0:
        raise ValueError("positive rational required")
    scale = 1 << (2 * bits)
    z = isqrt((den * scale) // num)
    # z^2 <= floor(den*4^b/num) <= den*4^b/num
    assert z * z * num <= den * scale
    hi = z if z * z * num == den * scale else z + 1
    assert hi * hi * num >= den * scale
    return Q(z, 1 << bits), Q(hi, 1 << bits)


def _floor_dy(v: Q) -> int:
    return (v.numerator << P_BITS) // v.denominator


def _ceil_dy(v: Q) -> int:
    return -((-v.numerator << P_BITS) // v.denominator)


@dataclass
class Cell:
    a: Q
    b: Q
    piece: int
    c: Q
    r: Q
    g_lo: Q
    g_hi: Q
    A_int: list           # integer coefficients of antiderivative from -1, ascending in tau
    A_den: int
    Rb: Q
    K: int
    rho: Q

    def partial(self, s: Q) -> tuple[Q, Q]:
        """Rigorous [lo, hi] of int_a^s x^{-1/2}, a <= s <= b."""
        if s <= self.a:
            return Q(0), Q(0)
        tau = (s - self.c) / self.r
        if tau > 1:
            tau = Q(1)
        p, q = tau.numerator, tau.denominator
        deg = len(self.A_int) - 1
        # homogeneous Horner: sum_i A_i p^i q^(deg-i), exact integers
        acc = 0
        qpow = 1
        for i in range(deg, -1, -1):
            acc = acc * p + self.A_int[i] * qpow
            qpow *= q
        A = Q(acc, self.A_den * q ** deg)
        err = (tau + 1) * self.Rb
        lo_core, hi_core = A - err, A + err
        cand = [self.r * g * v for g in (self.g_lo, self.g_hi) for v in (lo_core, hi_core)]
        return min(cand), max(cand)


@dataclass
class TimeLaw:
    pieces: list                     # [(left, right, ascending coeffs in (s-left))]
    cells: list = field(default_factory=list)
    prefix_lo: list = field(default_factory=list)   # integers * 2^-P, T bounds at cell starts
    prefix_hi: list = field(default_factory=list)
    starts: list = field(default_factory=list)
    status: str = "UNBUILT"
    max_K: int = 0
    max_rho: Q = Q(0)

    # ---------------------------------------------------------------- build
    @classmethod
    def from_pieces(cls, pieces, init_split: int = 4) -> "TimeLaw":
        self = cls(pieces=[(Q(l), Q(r), tuple(Q(v) for v in c)) for l, r, c in pieces])
        for i in range(len(self.pieces) - 1):
            if self.pieces[i][1] != self.pieces[i + 1][0]:
                raise ValueError("pieces must be contiguous")
        stack_status = "BOUNDED_EXACT_MODEL"
        for pi, (L, R, c) in enumerate(self.pieces):
            w = (R - L) / init_split
            todo = [(L + w * k, L + w * (k + 1), 0) for k in range(init_split)]
            done = []
            while todo:
                a, b, depth = todo.pop()
                cell = _make_cell(pi, L, c, a, b)
                if cell is None:
                    if depth >= MAX_DEPTH:
                        self.status = "UNKNOWN_POSITIVITY"
                        return self
                    m = (a + b) / 2
                    todo.append((m, b, depth + 1))
                    todo.append((a, m, depth + 1))
                    continue
                done.append(cell)
            done.sort(key=lambda z: z.a)
            self.cells.extend(done)
        lo = hi = 0
        for cell in self.cells:
            self.starts.append(cell.a)
            self.prefix_lo.append(lo)
            self.prefix_hi.append(hi)
            plo, phi = cell.partial(cell.b)
            lo += _floor_dy(plo)
            hi += _ceil_dy(phi)
            self.max_K = max(self.max_K, cell.K)
            self.max_rho = max(self.max_rho, cell.rho)
        self.total = (lo, hi)
        self.status = stack_status
        return self

    @classmethod
    def from_timing_controls(cls, timing_controls, init_split: int = 4) -> "TimeLaw":
        model = ES.model_from_controls(timing_controls)
        return cls.from_model(model, init_split)

    @classmethod
    def from_model(cls, model, init_split: int = 4) -> "TimeLaw":
        pieces = [(Q(m, ES.SPANS), Q(m + 1, ES.SPANS), model[m]) for m in range(ES.SPANS)]
        return cls.from_pieces(pieces, init_split)

    # ---------------------------------------------------------------- queries
    @property
    def end(self) -> Q:
        return self.pieces[-1][1]

    def T_bounds(self, s) -> tuple[Q, Q]:
        """Rigorous dyadic [lo, hi] containing T(s), 0 <= s <= end."""
        if self.status != "BOUNDED_EXACT_MODEL":
            raise RuntimeError(self.status)
        s = Q(s)
        if s < self.pieces[0][0] or s > self.end:
            raise ValueError("s outside model domain")
        if s == self.pieces[0][0]:
            return Q(0), Q(0)
        i = bisect.bisect_left(self.starts, s) - 1
        cell = self.cells[i]
        plo, phi = cell.partial(s)
        return (Q(self.prefix_lo[i] + _floor_dy(plo), 1 << P_BITS),
                Q(self.prefix_hi[i] + _ceil_dy(phi), 1 << P_BITS))

    def total_bounds(self) -> tuple[Q, Q]:
        return Q(self.total[0], 1 << P_BITS), Q(self.total[1], 1 << P_BITS)

    def inverse(self, t, bits: int = 52) -> tuple[Q, Q, str]:
        """Rigorous [s_lo, s_hi] containing s*(t) (s* = end for t >= T(end)).

        Status: EXACT_ZERO, SATURATED (proved t >= T(end)), BRACKETED (width <= 2^-bits),
        or OVERLAP_LIMITED (bisection stopped because T bounds straddle t)."""
        t = Q(t)
        if t < 0:
            raise ValueError("negative time")
        L, R = self.pieces[0][0], self.end
        if t == 0:
            return L, L, "EXACT_ZERO"
        Tlo1, Thi1 = self.total_bounds()
        if Thi1 <= t:
            return R, R, "SATURATED"
        lo, hi = L, R          # T(L)=0 < t ; s* <= R always (hold after the end)
        width = Q(1, 1 << bits)
        while hi - lo > width:
            mid = (lo + hi) / 2
            tl, th = self.T_bounds(mid)
            if th <= t:
                lo = mid
            elif tl >= t:
                hi = mid
            else:
                return lo, hi, "OVERLAP_LIMITED"
        return lo, hi, "BRACKETED"

    def x_range_on(self, a: Q, b: Q) -> tuple[Q, Q]:
        """Rigorous [lower bound of min x, upper bound of max x] on [a,b] inside the domain.

        Extremes of a cubic on a closed interval lie at the endpoints (exact values) or at
        interior critical points; each critical point is enclosed in a tiny rational
        interval (isqrt enclosure of the discriminant) on which a centred form is used."""
        a, b = Q(a), Q(b)
        lo, hi = None, None
        cuts = sorted(set([a, b] + [p[0] for p in self.pieces if a < p[0] < b]))
        for u, v in zip(cuts, cuts[1:]):
            pi = max(k for k, p in enumerate(self.pieces) if p[0] <= u)
            Lp, _, c = self.pieces[pi]
            c = list(c) + [Q(0)] * (4 - len(c))
            wu, wv = u - Lp, v - Lp
            vals = [_ev(c, wu), _ev(c, wv)]
            lows, highs = list(vals), list(vals)
            for w1, w2 in _critical_enclosures(c):
                w1, w2 = max(w1, wu), min(w2, wv)
                if w1 > w2:
                    continue
                cc, rr = (w1 + w2) / 2, (w2 - w1) / 2
                X = _compose_affine(c, cc, rr)
                rad = sum(abs(t) for t in X[1:])
                lows.append(X[0] - rad)
                highs.append(X[0] + rad)
            lo = min(lows) if lo is None else min(lo, min(lows))
            hi = max(highs) if hi is None else max(hi, max(highs))
        return lo, hi


def _ev(c, w: Q) -> Q:
    return ((c[3] * w + c[2]) * w + c[1]) * w + c[0]


def _critical_enclosures(c, bits: int = 200):
    """Rational intervals each containing one real root of x'(w) = c1 + 2c2 w + 3c3 w^2."""
    A, B, C = 3 * c[3], 2 * c[2], c[1]
    if A == 0:
        if B == 0:
            return []
        r = -C / B
        return [(r, r)]
    D = B * B - 4 * A * C
    if D < 0:
        return []
    # sqrt(D) in [sl, sh]
    scale = 1 << (2 * bits)
    z = isqrt((D.numerator * scale) // D.denominator)
    sl = Q(z, 1 << bits)
    sh = Q(z + 1, 1 << bits)
    assert sl * sl <= D <= sh * sh
    out = []
    for sign in (1, -1):
        ends = [(-B + sign * sq) / (2 * A) for sq in (sl, sh)]
        out.append((min(ends), max(ends)))
    return out


def _dyadic_near(v: Q, rel_bits: int = 60) -> Q:
    """A positive dyadic close to v > 0 (only needs to be positive and close)."""
    e = v.numerator.bit_length() - v.denominator.bit_length()
    sh = rel_bits - e
    if sh >= 0:
        n = (v.numerator << sh) // v.denominator
        return Q(max(n, 1), 1 << sh)
    n = v.numerator // (v.denominator << -sh)
    return Q(max(n, 1) << -sh)


def _make_cell(pi, left, c, a: Q, b: Q):
    cc, rr = (a + b) / 2, (b - a) / 2
    X = _compose_affine(c, cc - left, rr)
    X += [Q(0)] * (4 - len(X))
    if X[0] <= 0:
        return None
    x0 = _dyadic_near(X[0])
    u = [(X[0] - x0) / x0] + [v / x0 for v in X[1:]]
    rho = sum(abs(v) for v in u)
    if rho >= Q(1, 2):
        return None
    one_minus = 1 - rho
    tol = Q(1, 1 << TOL_BITS)
    K = None
    for k in range(1, KMAX + 1):
        Rb = abs(beta(k + 1)) * rho ** (k + 1) / one_minus ** (k + 2)
        if Rb <= tol:
            K = k
            break
    if K is None:
        return None
    # integer polynomial arithmetic for sum_{n<=K} beta_n u^n
    ui, ud = _to_int_poly(u)
    # beta_n u^n = (-1)^n C(2n,n) ui^n / (4^n ud^n); common denominator 4^K ud^K
    total = [0]
    power = [1]
    for n in range(K + 1):
        coef = (-1) ** n * comb(2 * n, n) * (4 * ud) ** (K - n)
        term = [coef * v for v in power]
        if len(term) > len(total):
            total += [0] * (len(term) - len(total))
        for i, v in enumerate(term):
            total[i] += v
        power = _imul(power, ui)
    Qden = (4 * ud) ** K
    # antiderivative from -1: int_{-1}^{tau} t^i dt = (tau^{i+1} - (-1)^{i+1})/(i+1)
    from math import lcm
    deg = len(total) - 1
    Lc = 1
    for i in range(1, deg + 2):
        Lc = lcm(Lc, i)
    A = [0] * (deg + 2)
    for i, v in enumerate(total):
        f = Lc // (i + 1)
        A[i + 1] += v * f
        A[0] -= v * f * (-1) ** (i + 1)
    g_lo, g_hi = inv_sqrt_bracket(x0.numerator, x0.denominator)
    return Cell(a, b, pi, cc, rr, g_lo, g_hi, A, Qden * Lc, Rb, K, rho)

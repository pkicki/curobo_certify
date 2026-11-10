"""Exact-rational reference for TimeOpt spline boundary and knot obligations.

This module checks *only* the stated boundary/regularity sub-contract. It
cannot authorize a velocity/acceleration/jerk/torque certificate. Floating
coefficients are interpreted as their exact binary values (not rounded to a
nearby rational). No fitted geometry, tolerance or sampling is used.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Sequence


def rational(value: int | float | str | Fraction) -> Fraction:
    """Parse a finite exact integer, binary float, or rational string."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a polynomial coefficient")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("nonfinite polynomial coefficient")
    try:
        result = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"invalid rational value: {value!r}") from exc
    return result


def polynomial(values: Sequence[int | float | str | Fraction]) -> tuple[Fraction, ...]:
    if not values:
        raise ValueError("polynomials must have at least one coefficient")
    if len(values) > 4:
        raise ValueError("only degree <= 3 polynomials are supported")
    return tuple(rational(v) for v in values)


def evaluate(coefficients: Sequence[Fraction], u: Fraction) -> Fraction:
    """Evaluate local power coefficients, in increasing degree order."""
    result = Fraction(0)
    for coefficient in reversed(coefficients):
        result = result * u + coefficient
    return result


def derivative(coefficients: Sequence[Fraction], order: int, h: Fraction) -> tuple[Fraction, ...]:
    """Differentiate with respect to the global path coordinate s."""
    if order < 0 or h <= 0:
        raise ValueError("invalid derivative order or span length")
    current = tuple(coefficients)
    for _ in range(order):
        current = tuple(Fraction(k) * current[k] / h for k in range(1, len(current)))
        if not current:
            return (Fraction(0),)
    return current


@dataclass(frozen=True)
class Span:
    """q_i(u), x(u), where u=(s-left)/(right-left), 0<=u<=1."""

    left: Fraction
    right: Fraction
    q: tuple[tuple[Fraction, ...], ...]
    x: tuple[Fraction, ...]

    @classmethod
    def from_values(cls, left, right, q, x) -> "Span":
        left, right = rational(left), rational(right)
        if right <= left:
            raise ValueError("spans must have positive finite length")
        if not q:
            raise ValueError("at least one joint is required")
        return cls(left, right, tuple(polynomial(p) for p in q), polynomial(x))

    def at(self, kind: str, u: int, order: int = 0) -> tuple[Fraction, ...]:
        polynomials = self.q if kind == "q" else (self.x,)
        if kind not in ("q", "x") or u not in (0, 1):
            raise ValueError("unknown polynomial or one-sided evaluation point")
        return tuple(evaluate(derivative(p, order, self.right - self.left), Fraction(u))
                     for p in polynomials)

    def acceleration(self, u: int) -> tuple[Fraction, ...]:
        x = self.at("x", u)[0]
        dx = self.at("x", u, 1)[0]
        return tuple(q2 * x + q1 * dx / 2 for q1, q2 in
                     zip(self.at("q", u, 1), self.at("q", u, 2)))


@dataclass(frozen=True)
class Boundaries:
    """All four target fields mandatory; strict_rest also demands a=0.

    q_start/q_end are joint coordinate vectors; x_start/x_end are squared
    path-speeds. acc_start/end optionally specify desired joint acceleration.
    """

    q_start: tuple[Fraction, ...] | None = None
    q_end: tuple[Fraction, ...] | None = None
    x_start: Fraction | None = None
    x_end: Fraction | None = None
    strict_rest: bool = False
    acc_start: tuple[Fraction, ...] | None = None
    acc_end: tuple[Fraction, ...] | None = None

    @classmethod
    def from_values(cls, q_start=None, q_end=None, x_start=None, x_end=None,
                    strict_rest=False, acc_start=None, acc_end=None) -> "Boundaries":
        as_vec = lambda v: None if v is None else tuple(rational(z) for z in v)
        return cls(as_vec(q_start), as_vec(q_end),
                   None if x_start is None else rational(x_start),
                   None if x_end is None else rational(x_end),
                   strict_rest, as_vec(acc_start), as_vec(acc_end))


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # PASS, VIOLATED, UNKNOWN
    detail: str


@dataclass(frozen=True)
class BoundaryResult:
    status: str  # PASS, VIOLATED, UNKNOWN, INVALID_INPUT
    checks: tuple[Check, ...]


def _local_duration_check(poly: tuple[Fraction, ...], endpoint: int) -> Check:
    """Local integrability of ds/sqrt(x), not a global positivity proof."""
    if endpoint == 0:
        series = poly
    else:
        # Coefficient of w^k in x(1-w), using exact binomial expansion.
        from math import comb
        series = tuple(sum((poly[j] * comb(j, k) * (-1) ** k
                            for j in range(k, len(poly))), Fraction(0))
                       for k in range(len(poly)))
    for order, coefficient in enumerate(series):
        if coefficient == 0:
            continue
        if coefficient < 0:
            return Check(f"duration_endpoint_{endpoint}", "VIOLATED", "x negative immediately inside endpoint")
        if order >= 2:
            return Check(f"duration_endpoint_{endpoint}", "VIOLATED", "infinite local traversal time: root order >=2")
        return Check(f"duration_endpoint_{endpoint}", "PASS", "locally integrable if x stays positive")
    return Check(f"duration_endpoint_{endpoint}", "UNKNOWN", "identically zero x; stationary contract not implemented")


def check_boundaries(spans: Sequence[Span], targets: Boundaries) -> BoundaryResult:
    """Check exact C2(q), C1(x), one-sided knots and declared endpoints.

    A PASS is *only* a boundary/regularity result. Interior x positivity and
    all K3 residual inequalities must be certified by a separate module.
    """
    if not spans:
        return BoundaryResult("INVALID_INPUT", (Check("spans", "UNKNOWN", "empty span sequence"),))
    joint_count = len(spans[0].q)
    if (joint_count == 0 or any(s.right <= s.left or len(s.q) != joint_count
                                or len(s.x) > 4 or any(len(p) > 4 or not p for p in s.q)
                                for s in spans)):
        return BoundaryResult("INVALID_INPUT", (Check("geometry", "UNKNOWN", "invalid span topology or polynomial"),))
    checks = []

    def add(name, lhs, rhs):
        status = "PASS" if lhs == rhs else "VIOLATED"
        checks.append(Check(name, status, "exact equality" if status == "PASS" else f"left={lhs}, right={rhs}"))

    for index, (a, b) in enumerate(zip(spans, spans[1:])):
        add(f"knot_{index}_location", a.right, b.left)
        if a.right != b.left:
            continue
        for order in range(3):
            add(f"knot_{index}_q_d{order}", a.at("q", 1, order), b.at("q", 0, order))
        for order in range(2):
            add(f"knot_{index}_x_d{order}", a.at("x", 1, order), b.at("x", 0, order))
        # A moving interior knot must not have zero or negative squared speed.
        speed = a.at("x", 1)[0]
        checks.append(Check(f"knot_{index}_moving", "PASS" if speed > 0 else "VIOLATED" if speed < 0 else "UNKNOWN",
                            "positive at knot" if speed > 0 else "negative x at knot" if speed < 0 else "stationary interior knot requires separate traversal contract"))

    for side, span, u, expected_q, expected_x, expected_acc in (
        ("start", spans[0], 0, targets.q_start, targets.x_start, targets.acc_start),
        ("end", spans[-1], 1, targets.q_end, targets.x_end, targets.acc_end),
    ):
        if expected_q is None or expected_x is None:
            checks.append(Check(f"{side}_targets", "UNKNOWN", "explicit q and x targets required"))
        else:
            if len(expected_q) != joint_count:
                return BoundaryResult("INVALID_INPUT", (Check(f"{side}_targets", "UNKNOWN", "joint target dimension mismatch"),))
            add(f"{side}_q", span.at("q", u), expected_q)
            add(f"{side}_x", span.at("x", u)[0], expected_x)
        if targets.strict_rest:
            add(f"{side}_strict_rest_x", span.at("x", u)[0], Fraction(0))
            add(f"{side}_strict_rest_acc", span.acceleration(u), (Fraction(0),) * joint_count)
        if expected_acc is not None:
            if len(expected_acc) != joint_count:
                return BoundaryResult("INVALID_INPUT", (Check(f"{side}_acc", "UNKNOWN", "acceleration target dimension mismatch"),))
            add(f"{side}_acc", span.acceleration(u), expected_acc)
        checks.append(_local_duration_check(span.x, u))

    status = ("VIOLATED" if any(c.status == "VIOLATED" for c in checks)
              else "UNKNOWN" if any(c.status == "UNKNOWN" for c in checks) else "PASS")
    return BoundaryResult(status, tuple(checks))

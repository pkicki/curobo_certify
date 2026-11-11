"""Exact-rational bounds on polynomial maximum and Bernstein hull looseness.

This is a *model-level analysis* tool, not a native-kernel certificate.
The extrema bracket uses rational point witnesses and exact Bernstein hulls.
No sampling-based bound or rounded floating-point arithmetic is involved.
"""

from dataclasses import dataclass
from fractions import Fraction
import heapq

from curobo_certify.exact_k3_reference import bernstein_coeffs, midpoint_split


@dataclass(frozen=True)
class PeakBracket:
    status: str  # BOUNDED, UNKNOWN_DEPTH, UNKNOWN_NODES
    lower: Fraction  # observed exact maximum over dyadic points
    upper: Fraction  # rigorous maximum upper bound on [0,1]
    initial_hull_upper: Fraction
    nodes: int
    depth: int

    @property
    def overestimation_lower(self) -> Fraction:
        return max(Fraction(0), self.initial_hull_upper - self.upper)

    @property
    def overestimation_upper(self) -> Fraction:
        return max(Fraction(0), self.initial_hull_upper - self.lower)

    @property
    def margin_lower(self) -> Fraction:
        return -self.upper

    @property
    def margin_upper(self) -> Fraction:
        return -self.lower


def bracket_polynomial_peak(power, *, tolerance=Fraction(1, 1000000),
                            max_depth=20, max_nodes=10000) -> PeakBracket:
    """Find guaranteed lower/upper bounds on max_{u in [0,1]} p(u).

    Stop when upper-lower <= tolerance; a budget-exhausted result still
    supplies mathematically valid brackets. For per-family residual analysis,
    use negative upper as a certified margin and positive lower as a witness.
    """
    if not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    if not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    tolerance = Fraction(tolerance)
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    root = bernstein_coeffs(tuple(Fraction(c) for c in power))
    hull = max(root)
    root_l, root_r = midpoint_split(root)
    best = max(root[0], root[-1], root_l[-1])
    # Priority queue of (-upper, tie-breaker, coeffs, depth).
    # Only active leaf intervals are in the heap; each subdivided parent is removed.
    queue = [(-hull, 0, root, 0)]
    counter = 0
    processed = 1
    achieved_depth = 0
    while queue:
        global_upper = -queue[0][0]
        if global_upper - best <= tolerance:
            return PeakBracket("BOUNDED", best, global_upper, hull, processed, achieved_depth)
        neg_up, ordinal, coefficients, depth = heapq.heappop(queue)
        if depth >= max_depth:
            heapq.heappush(queue, (neg_up, ordinal, coefficients, depth))
            return PeakBracket("UNKNOWN_DEPTH", best, -queue[0][0], hull, processed, achieved_depth)
        if processed + 2 > max_nodes:
            heapq.heappush(queue, (neg_up, ordinal, coefficients, depth))
            return PeakBracket("UNKNOWN_NODES", best, -queue[0][0], hull, processed, achieved_depth)
        left, right = midpoint_split(coefficients)
        best = max(best, left[-1])
        achieved_depth = max(achieved_depth, depth + 1)
        for child in (left, right):
            counter += 1
            heapq.heappush(queue, (-max(child), counter, child, depth + 1))
        processed += 2
    raise AssertionError("a polynomial always has at least one active leaf")


def diagnostic_dict(bracket: PeakBracket) -> dict:
    """Emit exact rational strings; never float-round the rigorous bracket."""
    return {"peak_bracket_status": bracket.status,
            "peak_lower": str(bracket.lower), "peak_upper": str(bracket.upper),
            "initial_hull_upper": str(bracket.initial_hull_upper),
            "overestimation_lower": str(bracket.overestimation_lower),
            "overestimation_upper": str(bracket.overestimation_upper),
            "margin_lower": str(bracket.margin_lower),
            "margin_upper": str(bracket.margin_upper),
            "nodes": bracket.nodes, "depth": bracket.depth,
            "sampled": False, "rigorous_for": "EXACT_SERIALIZED_POLYNOMIAL"}


def uniform_subdivision_diagnostics(power, *, max_depth=16):
    """Exact Bernstein hull result at fixed uniform dyadic depths.

    This reports representation complexity only; it does not time a certifier.
    """
    if not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    root = bernstein_coeffs(tuple(Fraction(c) for c in power))
    frontier = [root]
    levels = []
    required = None
    for depth in range(max_depth + 1):
        upper = max(max(piece) for piece in frontier)
        levels.append({"depth": depth, "leaves": len(frontier), "maximum_hull_upper": str(upper),
                       "all_hulls_certify": upper <= 0})
        if upper <= 0:
            required = depth
            break
        frontier = [child for piece in frontier for child in midpoint_split(piece)]
    comparison_frontier = [root]
    comparison_levels = []
    for depth in range(min(max_depth, 2) + 1):
        upper = max(max(piece) for piece in comparison_frontier)
        comparison_levels.append({"depth": depth, "leaves": len(comparison_frontier),
                                  "maximum_hull_upper": str(upper),
                                  "all_hulls_certify": upper <= 0})
        if depth < min(max_depth, 2):
            comparison_frontier = [child for piece in comparison_frontier
                                   for child in midpoint_split(piece)]
    return {"required_uniform_depth_to_certify": required,
            "uniform_status": "CERTIFIED_BY_HULL" if required is not None else "UNKNOWN_DEPTH",
            "required_depth_search": levels,
            "uniform_comparison_levels_0_1_2": comparison_levels}

"""Randomized adversarial soundness campaign for the reviewed K4 reference.

The REVIEWED code is only *called* here (certify_torque / replay / prepared geometry);
every truth value it is compared against comes from independent_checks.k4.oracle.
"""
from __future__ import annotations

import random
from fractions import Fraction as Q

from . import oracle as O


def rand_two_link(rng):
    r = lambda lo, hi, den=16: Q(rng.randint(int(lo * den), int(hi * den)), den)
    p = dict(m1=r(0.25, 4), m2=r(0.25, 4), l1=r(0.25, 2), c1=r(0.0625, 1.5),
             c2=r(0.0625, 1.5), J1=r(0, 0.5), J2=r(0, 0.5), gravity=rng.choice([Q(0), Q(981, 100), r(0, 20)]))
    return p


def rand_pendulum(rng):
    r = lambda lo, hi, den=16: Q(rng.randint(int(lo * den), int(hi * den)), den)
    mass, com = r(0.25, 4), r(0.0625, 1.5)
    # physical pivot inertia >= m c^2 (reviewed code does not enforce this; see memo)
    return dict(inertia=mass * com * com + r(0, 1), mass=mass, com=com,
                gravity=rng.choice([Q(0), Q(981, 100), r(0, 20)]))


def to_code_spans(PathSpan, ospans):
    return [PathSpan(s.left, s.right, s.q, s.x) for s in ospans]


def sample_grid(ospans, per_span):
    for si, s in enumerate(ospans):
        for k in range(per_span + 1):
            yield si, Q(k, per_span)


def oracle_midpoint(t):
    lo, hi = O.iv_bounds(t)
    return (lo + hi) / 2


def choose_limits(rng, chain, ospans, per_span=24):
    vals = [[] for _ in range(chain.dof)]
    for si, u in sample_grid(ospans, per_span):
        if O.poly_eval(ospans[si].x, u) < 0:
            continue
        for j, t in enumerate(O.path_torque(chain, ospans[si], u)):
            vals[j].append(oracle_midpoint(t))
    lims = []
    for j in range(chain.dof):
        if not vals[j]:
            lims.append((Q(-1), Q(1)))
            continue
        lo, hi = min(vals[j]), max(vals[j])
        rng_ = max(hi - lo, Q(1))
        f = lambda: rng.choice([Q(-1, 20), Q(1, 1000), Q(1, 20), Q(3, 10), Q(1), Q(1), Q(3)])
        a = (lo - rng_ * f()).limit_denominator(4096) - Q(1, 4096)
        b = (hi + rng_ * f()).limit_denominator(4096) + Q(1, 4096)
        if a >= b:
            a, b = b - 1, b
        lims.append((a, b))
    return lims


def make_case(rng, kind=None, nspans=None, allow_negative_x=False, qscale=None):
    kind = kind or rng.choice(["pendulum", "two_link", "two_link"])
    nspans = nspans or rng.choice([1, 1, 2, 3])
    cuts = sorted({Q(rng.randint(1, 15), 16) for _ in range(nspans - 1)})
    knots = [Q(0)] + cuts + [Q(1)]
    if kind == "pendulum":
        params = rand_pendulum(rng)
        chain = O.pendulum_chain(**params)
    else:
        params = rand_two_link(rng)
        chain = O.two_link_chain(**params)
    qscale = qscale if qscale is not None else rng.choice([Q(1, 2), Q(2), Q(4)])
    ospans = O.random_c2_spline(rng, chain.dof, knots, qscale=qscale,
                                allow_negative_x=allow_negative_x)
    return kind, params, chain, ospans


def leaf_samples(rng, a, b, k=3):
    pts = [a, b, (a + b) / 2]
    for _ in range(k):
        pts.append(a + (b - a) * Q(rng.randint(1, 999), 1000))
    return pts


def audit_result(rng, chain, ospans, limits, result, extra_per_leaf=3, dense=64):
    """Check every reviewed claim in `result` against the oracle. Returns stats dict."""
    st = {"status": result.status, "leaf_checks": 0, "disjoint": [], "ambiguous": 0,
          "point_consistent": 0, "witness_ok": None, "x_violation_ok": None}

    def check(si, u, leaf):
        sp = ospans[si]
        x = O.poly_eval(sp.x, u)
        if x < 0:
            st["disjoint"].append({"kind": "x<0 inside CERTIFIED leaf", "span": si, "u": str(u)})
            return
        tau = O.path_torque(chain, sp, u)
        for j in range(chain.dof):
            box = leaf.torque[j]
            v = O.containment(box.lo, box.hi, tau[j])
            if v == "AMBIGUOUS":
                ex = O.exact_torque_if_trig_free(chain, sp, u)
                if ex is not None:
                    v = "EXACT_CONTAINED" if box.lo <= ex[j] <= box.hi else "DISJOINT"
                else:
                    v = O.check_box(box.lo, box.hi, lambda j=j: O.path_torque(chain, sp, u)[j])
            st["leaf_checks"] += 1
            if v == "DISJOINT":
                st["disjoint"].append({"kind": "torque outside leaf box", "span": si, "u": str(u),
                                       "joint": j, "box": [str(box.lo), str(box.hi)],
                                       "oracle": [str(z) for z in O.iv_bounds(tau[j])]})
            elif v == "AMBIGUOUS":
                st["ambiguous"] += 1
            elif v in ("POINT_CONSISTENT", "EDGE_CONSISTENT", "EXACT_CONTAINED"):
                st["point_consistent"] += 1
                st.setdefault(v, 0)
                st[v] += 1
            if status_certified:
                lv = O.containment(Q(limits[j][0]), Q(limits[j][1]), tau[j])
                if lv == "DISJOINT":
                    st["disjoint"].append({"kind": "CERTIFIED but oracle torque outside limit",
                                           "span": si, "u": str(u), "joint": j})

    status_certified = result.status == "CERTIFIED"
    by_span = {}
    for leaf in result.leaves:
        a, b = O.path_interval(leaf.path)
        by_span.setdefault(leaf.span, []).append((a, b, leaf))
        for u in leaf_samples(rng, a, b, extra_per_leaf):
            check(leaf.span, u, leaf)
    if status_certified:
        # dense grid over each span: every point must lie in some leaf and be enclosed there
        for si, u in sample_grid(ospans, dense):
            hits = [lf for (a, b, lf) in by_span.get(si, []) if a <= u <= b]
            if not hits:
                st["disjoint"].append({"kind": "point not covered by any leaf", "span": si, "u": str(u)})
                continue
            for lf in hits:
                check(si, u, lf)
        # independent partition check
        for si in range(len(ospans)):
            iv_ = sorted((a, b) for (a, b, _) in by_span.get(si, []))
            cur = Q(0)
            for a, b in iv_:
                if a != cur:
                    st["disjoint"].append({"kind": "partition gap/overlap", "span": si})
                    break
                cur = b
            if cur != 1:
                st["disjoint"].append({"kind": "partition incomplete", "span": si})
    if result.status == "VIOLATED_TORQUE":
        w = result.witness
        j = w["joint"]
        st["witness_ok"] = O.verify_violation(chain, ospans[w["span"]], w["u"], j, limits[j])
        if not st["witness_ok"]:
            st["disjoint"].append({"kind": "VIOLATED_TORQUE witness not confirmed by oracle",
                                   "span": w["span"], "u": str(w["u"]), "joint": j})
        # the reported point enclosure itself must contain the oracle value
        tau = O.path_torque(chain, ospans[w["span"]], w["u"])
        box = w["torque"][j]
        v = O.check_box(box.lo, box.hi, lambda: O.path_torque(chain, ospans[w["span"]], w["u"])[j])
        if v == "DISJOINT":
            st["disjoint"].append({"kind": "witness enclosure misses oracle torque"})
        if not (box.lo > Q(limits[j][1]) or box.hi < Q(limits[j][0])):
            st["disjoint"].append({"kind": "witness enclosure not strictly outside limits"})
        del tau
    if result.status == "VIOLATED_X":
        w = result.witness
        st["x_violation_ok"] = O.poly_eval(ospans[w["span"]].x, w["u"]) < 0
        if not st["x_violation_ok"]:
            st["disjoint"].append({"kind": "VIOLATED_X witness has x>=0"})
    return st


def run_campaign(seed, n_cases, *, max_depth=6, max_nodes=300, dense=48, extra=2,
                 allow_negative_x_rate=0.15, replay=True, affine_rate=0.3, verbose=False):
    from curobo_certify.k4_reference import (PathSpan, Pendulum, PlanarTwoLink,
                                             certify_torque, replay_torque_leaves)
    from curobo_certify.k4_affine import prepare_geometry, certify_with_prepared_geometry
    rng = random.Random(seed)
    summary = {"seed": seed, "cases": 0, "status": {}, "leaf_checks": 0, "ambiguous": 0,
               "point_consistent": 0, "counterexamples": [], "replay_rejected_certified": 0,
               "witnesses_confirmed": 0, "x_witnesses_confirmed": 0,
               "affine_status": {}, "affine_leaf_checks": 0, "affine_replay_rejected": 0,
               "cases_q2_nonzero_gravity_nonzero": 0, "cases_negative_xprime": 0}
    for ci in range(n_cases):
        neg = rng.random() < allow_negative_x_rate
        kind, params, chain, ospans = make_case(rng, allow_negative_x=neg)
        model = Pendulum(**params) if kind == "pendulum" else PlanarTwoLink(**params)
        if kind == "two_link" and params["gravity"] != 0 and any(any(c != 0 for c in s.q[1]) for s in ospans):
            summary["cases_q2_nonzero_gravity_nonzero"] += 1
        if any(O.poly_deriv_eval(s.x, Q(k, 8), 1, s.h) < 0 for s in ospans for k in range(9)):
            summary["cases_negative_xprime"] += 1
        try:
            limits = choose_limits(rng, chain, ospans)
        except ValueError:
            limits = [(Q(-100), Q(100))] * chain.dof
        cspans = to_code_spans(PathSpan, ospans)
        res = certify_torque(cspans, model, limits, max_depth=max_depth, max_nodes=max_nodes)
        summary["cases"] += 1
        summary["status"][res.status] = summary["status"].get(res.status, 0) + 1
        st = audit_result(rng, chain, ospans, limits, res, extra_per_leaf=extra, dense=dense)
        summary["leaf_checks"] += st["leaf_checks"]
        summary["ambiguous"] += st["ambiguous"]
        summary["point_consistent"] += st["point_consistent"]
        for key in ("POINT_CONSISTENT", "EDGE_CONSISTENT", "EXACT_CONTAINED"):
            summary[key] = summary.get(key, 0) + st.get(key, 0)
        if st["witness_ok"]:
            summary["witnesses_confirmed"] += 1
        if st["x_violation_ok"]:
            summary["x_witnesses_confirmed"] += 1
        for d in st["disjoint"]:
            summary["counterexamples"].append({"case": ci, "kind": kind, "params": {k: str(v) for k, v in params.items()},
                                               "limits": [[str(a), str(b)] for a, b in limits], **d})
        if replay and res.status == "CERTIFIED" and not replay_torque_leaves(cspans, model, limits, res):
            summary["replay_rejected_certified"] += 1
        if rng.random() < affine_rate and res.status in ("CERTIFIED", "UNKNOWN_DEPTH", "UNKNOWN_NODES",
                                                          "VIOLATED_TORQUE"):
            try:
                box = prepare_geometry(cspans, model, depth=rng.choice([1, 2, 3]))
            except ValueError:
                continue
            # alternative timing: scale x and add a signed linear term, keep C1
            alt = []
            sc = Q(rng.randint(1, 16), 8)
            for s in ospans:
                alt.append(O.OSpan(s.left, s.right, s.q, tuple(c * sc for c in s.x)))
            calt = to_code_spans(PathSpan, alt)
            ar = certify_with_prepared_geometry(calt, model, limits, box)
            summary["affine_status"][ar.status] = summary["affine_status"].get(ar.status, 0) + 1
            if ar.status == "CERTIFIED":
                st2 = audit_result(rng, chain, alt, limits, ar, extra_per_leaf=extra, dense=dense)
                summary["affine_leaf_checks"] += st2["leaf_checks"]
                summary["ambiguous"] += st2["ambiguous"]
                for d in st2["disjoint"]:
                    summary["counterexamples"].append({"case": ci, "affine": True, **d})
                if not replay_torque_leaves(calt, model, limits, ar):
                    summary["affine_replay_rejected"] += 1
        if verbose and ci % 10 == 0:
            print(ci, summary["status"], len(summary["counterexamples"]), flush=True)
    return summary

#!/usr/bin/env python3
"""Runtime audit for the CPU interval arithmetic's documented assumptions.

This is a platform conformance check, not a machine-checked proof of Python or
compiler semantics. A failure is blocking; a pass only validates tested facts.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.interval import Interval  # noqa: E402


def encloses_fraction(interval: Interval, value: Fraction) -> bool:
    lower = Fraction.from_float(interval.lo) if math.isfinite(interval.lo) else None
    upper = Fraction.from_float(interval.hi) if math.isfinite(interval.hi) else None
    return (lower is None or lower <= value) and (upper is None or value <= upper)


def main() -> int:
    parser_seed = 20261110
    rng = random.Random(parser_seed)
    tested = {"add": 0, "subtract": 0, "multiply": 0, "divide": 0}
    failures: list[str] = []
    if sys.float_info.radix != 2 or sys.float_info.mant_dig != 53:
        failures.append("binary64 format is not radix-2 with 53-bit significand")
    if sys.float_info.rounds != 1:
        failures.append(f"C floating-point rounding mode is not FE_TONEAREST: {sys.float_info.rounds}")
    for value in (-1.0, -0.0, 0.0, 1.0, 2.0, sys.float_info.min, sys.float_info.max):
        lo, hi = math.nextafter(value, -math.inf), math.nextafter(value, math.inf)
        if not (lo < value < hi):
            failures.append(f"nextafter does not return adjacent neighbors at {value!r}")
    # Directed interval endpoints are checked against exact rational arithmetic.
    for i in range(20_000):
        xa = math.ldexp(rng.uniform(-1, 1), rng.randint(-500, 500))
        xb = math.ldexp(rng.uniform(-1, 1), rng.randint(-500, 500))
        a, b = Interval.point(xa), Interval.point(xb)
        fa, fb = Fraction.from_float(xa), Fraction.from_float(xb)
        for label, result, exact in (
            ("add", a + b, fa + fb), ("subtract", a - b, fa - fb),
            ("multiply", a * b, fa * fb),
        ):
            tested[label] += 1
            if not encloses_fraction(result, exact):
                failures.append(f"{label} under-enclosure at case {i}")
        if xb != 0:
            result = a / b
            tested["divide"] += 1
            if not encloses_fraction(result, fa / fb):
                failures.append(f"divide under-enclosure at case {i}")
    report = {
        "evidence_kind": "cpu_interval_runtime_conformance_audit",
        "campaign_data": False,
        "status": "PASS_TESTED_ASSUMPTIONS" if not failures else "BLOCKED",
        "python": sys.version,
        "platform": platform.platform(),
        "float_info": {"radix": sys.float_info.radix, "mant_dig": sys.float_info.mant_dig,
                       "rounds": sys.float_info.rounds, "max": sys.float_info.max,
                       "min": sys.float_info.min, "epsilon": sys.float_info.epsilon},
        "seed": parser_seed,
        "primitive_cases": tested,
        "failures": failures,
        "scope_limit": "Does not prove compiler flags, all exceptional cases, all inputs, or universal platform correctness.",
    }
    out = ROOT / "runs/cpu_interval_runtime_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    out.write_bytes(raw)
    print(json.dumps({"status": report["status"], "cases": tested,
                      "sha256": hashlib.sha256(raw).hexdigest(), "path": str(out),
                      "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

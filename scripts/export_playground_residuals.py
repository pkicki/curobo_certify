#!/usr/bin/env python3
"""Export analytic polynomial K3 residuals to the offline Proof Lab.

Inputs are exact serialized mathematical models: NEVER label generated residuals
as certificates of native CUDA/cuRobo execution. Each output entry concerns
only ONE span and ONE constraint family/joint.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.boundary_reference import Span  # noqa: E402
from curobo_certify.exact_k3_reference import build_residuals, bernstein_coeffs  # noqa: E402

def export(model: dict, *, span_index: int | None = None,
           joint: int | None = None, family: str | None = None,
           max_examples: int = 250) -> dict:
    if model.get("schema_version") != 1 or model.get("basis") != "local_power_ascending_u":
        raise ValueError("Expected schema_version=1, local_power_ascending_u model")
    limits = model["limits"]
    output = []
    for i, item in enumerate(model["spans"]):
        if span_index is not None and i != span_index:
            continue
        span = Span.from_values(item["left"], item["right"], item["q"], item["x"])
        for name, joint_id, power in build_residuals(span, limits["velocity"], limits["acceleration"], limits["jerk"]):
            if family is not None and name != family:
                continue
            if joint is not None and joint_id != joint:
                continue
            coeff = tuple(str(z) for z in bernstein_coeffs(power))
            if len(coeff) == 1:
                coeff = (coeff[0], coeff[0])  # exactly elevate constant to degree 1
            label = f"span {i} · {name}" + (f" · joint {joint_id}" if joint_id >= 0 else "")
            output.append({
                "id": f"s{i}-{name}-j{joint_id}",
                "title": label, "family": name,
                "strict": name == "positive_x",
                "coefficients": list(coeff),
                "description": (f"Exact mathematical residual on [{span.left}, {span.right}] "
                    f"from serialized polynomial span {i}; not native FP32 equivalent. "
                    "Other spans and constraints are NOT checked by this single-residual demo."),
                "source_span": i, "source_joint": joint_id,
                "span_left": str(span.left), "span_right": str(span.right)
            })
            if len(output)>max_examples:
                raise ValueError(f"Requested more than {max_examples} residual examples; filter span/joint/family")
    if not output:
        raise ValueError("Selection returned no residuals")
    return {
        "schema_version": 1,
        "scope": "ANALYTIC_EXACT_SERIALIZED_SINGLE_RESIDUAL_DEMOS_NOT_COMPLETE_K3",
        "native_parity": "NOT_EVALUATED",
        "source_model": "external serialized mathematical model",
        "examples": output
    }

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--span-index", type=int)
    p.add_argument("--joint", type=int)
    p.add_argument("--family", choices=("positive_x", "velocity", "acceleration_plus", "acceleration_minus", "jerk"))
    p.add_argument("--max-examples", type=int, default=250)
    args = p.parse_args()
    if args.max_examples < 1 or args.max_examples > 250:
        p.error("--max-examples must be in [1,250]")
    result=export(json.loads(args.model.read_text(encoding="utf-8")),
                  span_index=args.span_index, joint=args.joint,
                  family=args.family, max_examples=args.max_examples)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"Exported {len(result['examples'])} exact mathematical residual examples -> {args.out}")

if __name__=="__main__":
    main()

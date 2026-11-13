#!/usr/bin/env python3
"""Build the dependency-free offline Bernstein Proof Lab.

The HTML is a teaching/demonstration artifact for exact user-entered residuals,
not a complete K3 or native cuRobo certificate.
"""
from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "web/proof_template.html"
CORE = ROOT / "web/proof_core.js"
APP = ROOT / "web/proof_app.js"
DEFAULT = ROOT / "reports/bernstein_proof_lab.html"

def build(out: Path) -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("/* __BERNSTEIN_CORE__ */") != 1 or template.count("/* __BERNSTEIN_APP__ */") != 1:
        raise ValueError("Missing or duplicate script placeholders")
    core, app = CORE.read_text(encoding="utf-8"), APP.read_text(encoding="utf-8")
    for file_text in (core, app):
        if "</script" in file_text.lower():
            raise ValueError("Inline script tag terminator is forbidden")
    output = template.replace("/* __BERNSTEIN_CORE__ */", core).replace("/* __BERNSTEIN_APP__ */", app)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(output, encoding="utf-8")
    print(f"Built {out} ({len(output.encode('utf-8'))} bytes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT)
    build(parser.parse_args().out)

#!/usr/bin/env python3
"""Merge per-process frozen-output certification benchmark parts (identical builds required)."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    parts = [json.loads(p.read_text()) for p in a.parts]
    base = {k: v for k, v in parts[0].items() if k not in ("rows", "affinity")}
    for p in parts[1:]:
        # Compare build IDENTITY (source/exports/binary SHA, compiler), not the command line, which
        # embeds each process's temporary output path (Dev40 finding, 2026-11-17).
        ident = lambda bs: [(b["source_sha256"], b["exports_map_sha256"], b["binary_sha256"], b["compiler"]) for b in bs]  # noqa: E731
        if ident(p["builds"]) != ident(parts[0]["builds"]) or p["methods"] != parts[0]["methods"] or p["scope"] != parts[0]["scope"]:
            raise SystemExit("MIXED_BUILDS_OR_METHODS")
    rows, seen = [], set()
    for p in parts:
        for r in p["rows"]:
            key = (r["source_npz_sha256"], r["contract"], r["method"])
            if key in seen:
                raise SystemExit("DUPLICATE_ROW")
            seen.add(key)
            rows.append(r)
    out = {**base, "affinity_by_part": [p["affinity"] for p in parts], "parts": [str(p) for p in a.parts], "rows": rows}
    a.out.write_text(json.dumps(out) + "\n")
    print(f"merged {len(rows)} rows from {len(parts)} parts")


if __name__ == "__main__":
    main()

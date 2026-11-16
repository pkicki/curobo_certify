#!/usr/bin/env python3
"""Build a self-contained offline margin-vs-certifiability report.

By default this is an EMPTY interactive viewer with JSON import, not a report
claiming any new measurements. Embed only explicitly provided analysis JSON.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"web/margin_dashboard_template.html"
DEFAULT=ROOT/"reports/margin_certifiability_dashboard.html"


def build(out: Path, analysis: Path | None = None) -> None:
    template=SOURCE.read_text(encoding="utf-8")
    if template.count("/*__EMBED_DATA__*/")!=1:
        raise ValueError("Expected exactly one JSON data marker")
    if analysis is not None:
        obj=json.loads(analysis.read_text(encoding="utf-8"))
        if obj.get("schema_version")!=1 or not str(obj.get("scope","")).startswith(
            "EXPLORATORY_MATCHED_MARGIN_"
        ):
            raise ValueError("refusing incompatible benchmark scope")
        contents=json.dumps(obj,sort_keys=True,allow_nan=False,separators=(",",":"))
        # JSON may contain arbitrary user-controlled source IDs. Do not let
        # "</script>" close a script element and execute injected HTML.
        contents=(contents.replace("&","\\u0026").replace("<","\\u003c")
                  .replace(">","\\u003e").replace("\u2028","\\u2028")
                  .replace("\u2029","\\u2029"))
    else:
        contents="null"
    html=template.replace("/*__EMBED_DATA__*/","window.MARGIN_ANALYSIS = "+contents+";")
    if html.count("<script>")!=2 or html.count("</script>")!=2:
        raise ValueError("bad standalone script assembly")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(html,encoding="utf-8")
    print(f"Built offline margin dashboard: {out} ({len(html.encode('utf-8'))} bytes)")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis",type=Path,help="real analysis JSON, optional")
    ap.add_argument("--out",type=Path,default=DEFAULT)
    args=ap.parse_args()
    build(args.out,args.analysis)

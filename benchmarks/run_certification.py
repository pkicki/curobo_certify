#!/usr/bin/env python3
"""Fail-closed benchmark entry point; requires a frozen adapter-backed corpus."""
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if not a.manifest.is_file(): raise SystemExit('BLOCKED_BENCHMARK: frozen manifest is unavailable')
    m=json.loads(a.manifest.read_text())
    if not m.get('frozen') or not m.get('records'): raise SystemExit('BLOCKED_BENCHMARK: manifest is not frozen/nonempty')
    raise SystemExit('BLOCKED_BENCHMARK: no compatible frozen Panda/UR5 TimeOpt cohort or complete measurement driver; synthetic adapter/proof results are not campaign inputs; no timing rows written')
if __name__=='__main__': main()

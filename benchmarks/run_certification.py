#!/usr/bin/env python3
"""Fail-closed benchmark entry point; requires a frozen adapter-backed corpus."""
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if not a.manifest.is_file(): raise SystemExit('BLOCKED_BENCHMARK: frozen manifest is unavailable')
    m=json.loads(a.manifest.read_text())
    if not m.get('frozen') or not m.get('records'): raise SystemExit('BLOCKED_BENCHMARK: manifest is not frozen/nonempty')
    raise SystemExit('BLOCKED_BENCHMARK: pinned cuRobo adapter parity gate and complete multi-span proof contract have not passed; no timing rows written')
if __name__=='__main__': main()

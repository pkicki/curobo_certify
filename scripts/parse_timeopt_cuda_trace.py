#!/usr/bin/env python3
"""Decode TIMEOPT_TRACE records captured from an instrumented CUDA build.

Preserves raw float32 bits and corresponding exact float values; this tool is
not a proof of arithmetic equivalence or an error bound.
"""
import argparse
import json
import re
import struct
from pathlib import Path

TRACE = re.compile(r'\bTIMEOPT_(TRACE|TERMS)\s+((?:[a-z0-9_]+=(?:[0-9a-fA-F]+)\s*)+)', re.I)
KEY_VALUE = re.compile(r'([a-z0-9_]+)=([0-9a-fA-F]+)', re.I)
INT_KEYS = {'idx', 'j', 'seg'}


def decode_lines(text: str) -> list[dict]:
    records = {}
    for line in text.splitlines():
        match = TRACE.search(line)
        if not match:
            continue
        kind, fields = match.groups()
        raw = dict(KEY_VALUE.findall(fields))
        if not {'idx', 'j'}.issubset(raw):
            continue
        # idx/j/seg are printed as decimal, everything else as hex bits.
        idx, joint = int(raw.pop('idx'), 10), int(raw.pop('j'), 10)
        item = records.setdefault((idx, joint), {'idx': idx, 'joint': joint, 'source': 'CUDA_PRINTF_DIAGNOSTIC'})
        for key, val in raw.items():
            if key in INT_KEYS:
                item[key] = int(val, 10)
            else:
                bits = int(val, 16)
                item[key + '_bits'] = val.lower().zfill(8)
                item[key] = struct.unpack('!f', bits.to_bytes(4, 'big'))[0]
        item[kind.lower() + '_present'] = True
    return [records[k] for k in sorted(records)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {'scope': 'NATIVE_CUDA_OPERAND_DIAGNOSTIC_ONLY', 'records': decode_lines(args.log.read_text())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'Parsed {len(result["records"])} CUDA trace records')


if __name__ == '__main__':
    main()

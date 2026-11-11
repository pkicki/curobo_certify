#!/usr/bin/env python3
"""Build the strictly compiled binary64 K3 hull *component* (not a full certifier)."""
from pathlib import Path
import argparse
import json
import hashlib
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ('-std=c++17', '-O3', '-shared', '-fPIC', '-fno-fast-math',
         '-ffp-contract=off', '-fexcess-precision=standard', '-Wall', '-Wextra', '-Werror')


def build(output: Path, compiler: str = 'g++') -> dict:
    source = ROOT / 'src/curobo_certify/native/kinematic_hull.cpp'
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, *FLAGS, str(source), '-o', str(output)]
    subprocess.run(command, check=True)
    compiler_version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    return {'scope': 'COMPILED_BINARY64_INITIAL_HULL_ONLY',
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'binary_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'compiler': compiler_version, 'command': command,
            'platform': platform.platform(), 'full_k3_proof': False,
            'arithmetic_assumption': 'IEEE binary64 FLT_EVAL_METHOD=0, no fast math/FMA contraction; nextafter enclosure for every primitive, tested but not formal cross-platform audit'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    result = build(args.out, args.compiler)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2) + '\n')
    print('Built compiled K3 hull-only component:', args.out)


if __name__ == '__main__':
    main()

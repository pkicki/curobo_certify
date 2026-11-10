"""Canonical JSON proof-tree serialization for one-span certificates."""
import json
from dataclasses import asdict

def dumps(result): return json.dumps(asdict(result),sort_keys=True,separators=(",",":"),allow_nan=False)
def dump(result,path):
    with open(path,"w",encoding="utf-8") as f: f.write(dumps(result)+"\n")

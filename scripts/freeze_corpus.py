#!/usr/bin/env python3
"""Validate and freeze caller-provided immutable path records; never synthesizes paths."""
import argparse, hashlib, json
from pathlib import Path

SPLITS={"Dev","Holdout","Confirm"}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def freeze(source: Path, out: Path):
    rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not rows: raise ValueError("empty source manifest: no source paths are available")
    required={"path_id","source_id","scene_id","robot","split","arrays","provenance"}
    seen=set(); group_split={}; frozen=[]
    base=source.resolve().parent
    for row in rows:
        missing=required-row.keys()
        if missing: raise ValueError(f"{row.get('path_id','<row>')}: missing fields {sorted(missing)}")
        if row["split"] not in SPLITS: raise ValueError(f"invalid split {row['split']!r}")
        if row["path_id"] in seen: raise ValueError(f"duplicate path_id {row['path_id']}")
        seen.add(row["path_id"])
        group=(row["robot"],row["source_id"])
        prev=group_split.setdefault(group,row["split"])
        if prev != row["split"]: raise ValueError(f"source/scene leakage across splits: {group}")
        if not isinstance(row["arrays"],dict) or not row["arrays"]: raise ValueError(f"{row['path_id']}: no immutable arrays")
        records={}
        for key,desc in row["arrays"].items():
            p=(base/desc["path"]).resolve()
            if not p.is_file(): raise ValueError(f"missing array {p}")
            digest=sha(p)
            if digest != desc.get("sha256"): raise ValueError(f"SHA256 mismatch for {p}")
            records[key]={"path":str(p),"sha256":digest,"bytes":p.stat().st_size}
        frozen.append({**row,"arrays":records})
    frozen.sort(key=lambda r:(r["split"],r["robot"],r["source_id"],r["path_id"]))
    out.mkdir(parents=True,exist_ok=False)
    manifest={"schema_version":1,"frozen":True,"source_manifest_sha256":sha(source),"path_count":len(frozen),"split_counts":{s:sum(r['split']==s for r in frozen) for s in sorted(SPLITS)},"records":frozen}
    tmp=out/"manifest.json.tmp"; tmp.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n"); tmp.replace(out/"manifest.json")
    return manifest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source-manifest",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    m=freeze(a.source_manifest,a.out); print(json.dumps({"frozen":m['path_count'],"splits":m['split_counts'],"manifest":str(a.out/'manifest.json')}))
if __name__=="__main__":
    try: main()
    except (ValueError,KeyError,TypeError,FileNotFoundError,FileExistsError) as exc: raise SystemExit(f"BLOCKED_CORPUS: {exc}")

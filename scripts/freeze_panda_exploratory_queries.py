#!/usr/bin/env python3
"""Freeze a separate, deterministic eight-query Panda exploratory population.

This is not Dev/Holdout/Confirm and must never be mixed with the original pilot.
Queries are serialized before any solver execution; solver margins stay pinned.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from curobo_certify.query_pipeline import pilot_queries,_canonical_hash

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20261111);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 rows=[r for r in pilot_queries(a.seed) if r['robot']=='panda']
 for i,row in enumerate(rows):
  row['source_id']=f'explore-panda-{a.seed}-{i:02d}'
  row['source_query_sha256']=_canonical_hash({k:v for k,v in row.items() if k!='source_query_sha256'})
 data=(json.dumps(rows,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_bytes(data)
 sidecar={'schema':'timeopt_panda_exploratory_queries_v1','scope':'SEPARATE_EXPLORATORY_POPULATION_NOT_FORMAL_COHORT',
  'seed':a.seed,'query_file_sha256':hashlib.sha256(data).hexdigest(),
  'solver_configuration':'pinned cuRobo c301cb00a87012c2808b1a60649c4065c9b3c95a; native default TimeOpt margins and 32 controls unchanged',
  'query_count':len(rows),'scene_counts':{s:sum(r['scene_id']==s for r in rows) for s in sorted({r['scene_id'] for r in rows})}}
 Path(str(a.out)+'.manifest.json').write_text(json.dumps(sidecar,sort_keys=True,indent=2)+'\n')
 print(json.dumps(sidecar|{'out':str(a.out)}))
if __name__=='__main__':main()

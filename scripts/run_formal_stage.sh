#!/usr/bin/env bash
# Run one frozen Panda-only formal cohort end to end (sequential stages, no GPU overlap).
# Usage: scripts/run_formal_stage.sh <Panda-Dev40|Panda-Holdout100|Panda-Confirm300> <manifest-sha256> [--sweeps]
# Requires the R1 freeze (docs/reports/formal_panda/FREEZE.json) and is refused otherwise.
set -euo pipefail
COHORT="$1"; SHA="$2"; SWEEPS="${3:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
Q="docs/reports/formal_panda/queries/${COHORT}.json"
OUT="runs/formal_panda/${COHORT}"
ENV="conda run --no-capture-output -n curobo-timeopt-c301cb"
GPU_RTX="CUDA_VISIBLE_DEVICES=1"
test -f docs/reports/formal_panda/FREEZE.json || { echo "NO_FREEZE"; exit 2; }
python scripts/check_formal_freeze.py --cohort "$COHORT" --queries "$Q" --expected-sha256 "$SHA"
mkdir -p "$OUT"
log() { echo "[$(date -Is)] $*" | tee -a "$OUT/stage.log"; }

log "1 generate (first-K, RTX 3090)"
env $GPU_RTX PYTHONPATH=src $ENV python scripts/generate_formal_cohort.py --cohort "$COHORT" --queries "$Q" \
    --expected-sha256 "$SHA" --out-dir "$OUT/native" 2>&1 | tee -a "$OUT/generate.log"
mapfile -t NPZ < <(python - "$OUT/native/generation_summary.json" "$OUT/native" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
for e in s["ledger"]:
    if (e["arm"] == "primary" and e["selected_eligible"]) or (e["arm"] == "secondary" and e["planner_success"]):
        print(f"{sys.argv[2]}/{e['source_id']}.npz")
EOF
)
mapfile -t PRIMARY_IDS < <(python -c "
import json,sys; s=json.load(open('$OUT/native/generation_summary.json'))
print('\n'.join(e['source_id'] for e in s['ledger'] if e['arm']=='primary'))")
log "eligible outputs: ${#NPZ[@]}; primary attempted: ${#PRIMARY_IDS[@]}"

log "2 T1 v2 exhaustive audit (RTX 3090, offline)"
env $GPU_RTX PYTHONPATH=src $ENV python scripts/native_time_audit_v2.py --npz "${NPZ[@]}" \
    --require-gpu "NVIDIA GeForce RTX 3090" --out "$OUT/t1_v2.json" 2>&1 | tee -a "$OUT/t1.log"

log "3 exact tightness (CPU, untimed)"
python scripts/analyze_hardcases.py --npz "${NPZ[@]}" --workers 12 --margin-queries "$Q" \
    --margin-records-dir "$OUT/native" --out "$OUT/tightness.json" > "$OUT/tightness.log" 2>&1

log "4 frozen-output certification benchmark (CPU, 3 pinned processes)"
N=${#NPZ[@]}; P=3; pids=()
for i in 0 1 2; do
  part=(); for ((k=i; k<N; k+=P)); do part+=("${NPZ[$k]}"); done
  core=$((6 + 2*i))
  taskset -c $core python scripts/bench_certification_methods.py --npz "${part[@]}" $SWEEPS \
      --margin-queries "$Q" --margin-records-dir "$OUT/native" --out "$OUT/cert_bench_part$i.json" \
      > "$OUT/cert_bench_part$i.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/merge_cert_bench.py --parts "$OUT"/cert_bench_part{0,1,2}.json --out "$OUT/cert_bench.json"

log "5 fresh paired latency (RTX 3090; native vs +cert 1 thread vs +cert 8 threads)"
python -c "
import json; rows=json.load(open('$Q')); ids=set('''${PRIMARY_IDS[*]}'''.split())
json.dump([r for r in rows if r['source_id'] in ids], open('$OUT/fresh_queries.json','w'), indent=1)"
env $GPU_RTX OMP_NUM_THREADS=8 PYTHONPATH=src $ENV python scripts/bench_fresh_requests.py --queries "$OUT/fresh_queries.json" \
    --limit 100000 --warmups 50 --repeats 10 --contract native --threads 1 8 --seed 20261118 \
    --out "$OUT/fresh_rtx3090.json" 2>&1 | tee -a "$OUT/fresh.log"

log "6 analysis"
python scripts/analyze_formal_cohort.py --cohort "$COHORT" --queries "$Q" --expected-sha256 "$SHA" \
    --records-dir "$OUT/native" --cert-bench "$OUT/cert_bench.json" --tightness "$OUT/tightness.json" \
    --t1 "$OUT/t1_v2.json" --fresh "$OUT/fresh_rtx3090.json" --out "$OUT/analysis.json" | tee -a "$OUT/stage.log"
log "done"

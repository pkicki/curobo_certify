#!/usr/bin/env bash
# Preregistered A100 replication subset (FREEZE.json a100_replication_preregistered), run AFTER the
# Panda-Confirm300 RTX 3090 stage so no timing overlaps. A100 claims are only what is measured here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
C=runs/formal_panda/Panda-Confirm300; OUT=runs/formal_panda/a100; mkdir -p "$OUT"
ENV="conda run --no-capture-output -n curobo-timeopt-c301cb"
python - "$C" "$OUT" <<'EOF'
import json, sys
c, out = sys.argv[1], sys.argv[2]
s = json.load(open(f"{c}/native/generation_summary.json"))
prim = sorted([e for e in s["ledger"] if e["arm"] == "primary" and e["selected_eligible"]], key=lambda e: e["pool_order"])[:40]
pairs = {e["pair_id"] for e in prim}
sec = [e for e in s["ledger"] if e["arm"] == "secondary" and e["pair_id"] in pairs and e["planner_success"]]
npz = [f"{c}/native/{e['source_id']}.npz" for e in prim + sec]
open(f"{out}/subset_npz.txt", "w").write("\n".join(npz) + "\n")
rows = json.load(open("docs/reports/formal_panda/queries/Panda-Confirm300.json"))
ids = {e["source_id"] for e in prim}
json.dump([r for r in rows if r["source_id"] in ids], open(f"{out}/fresh_queries_a100.json", "w"), indent=1)
json.dump({"primary": [e["source_id"] for e in prim], "secondary": [e["source_id"] for e in sec]},
          open(f"{out}/subset.json", "w"), indent=1)
print(len(prim), len(sec))
EOF
mapfile -t NPZ < "$OUT/subset_npz.txt"
echo "[$(date -Is)] A100 T1 v2 on ${#NPZ[@]} RTX-frozen outputs" | tee -a "$OUT/stage.log"
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $ENV python scripts/native_time_audit_v2.py --npz "${NPZ[@]}" \
    --require-gpu "NVIDIA A100-PCIE-40GB" --out "$OUT/t1_v2_a100.json" > "$OUT/t1.log" 2>&1
echo "[$(date -Is)] A100 fresh paired latency" | tee -a "$OUT/stage.log"
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONPATH=src $ENV python scripts/bench_fresh_requests.py \
    --queries "$OUT/fresh_queries_a100.json" --limit 100000 --warmups 50 --repeats 10 --contract native --threads 1 8 \
    --seed 20261118 --out "$OUT/fresh_a100.json" > "$OUT/fresh.log" 2>&1
echo "[$(date -Is)] compare" | tee -a "$OUT/stage.log"
python scripts/compare_a100_replication.py --rtx-t1 "$C/t1_v2.json" --a100-t1 "$OUT/t1_v2_a100.json" \
    --rtx-fresh "$C/fresh_rtx3090.json" --a100-fresh "$OUT/fresh_a100.json" --subset "$OUT/subset.json" \
    --binary-audit docs/reports/formal_panda/a100/t1_binary_audit_a100.json \
    --fp-audit docs/reports/formal_panda/a100/fp_unit_audit_a100.json --out "$OUT/a100_replication.json"
echo "[$(date -Is)] done" | tee -a "$OUT/stage.log"

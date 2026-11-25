#!/usr/bin/env bash
# Step 8: score the quantized GGUF models with the same test sets and metric code as evaluate.py. Safe to re-run.
#
#   ./eval_gguf.sh                     # after quantize.py; CPU only, like a laptop
#   THREADS=4 ./eval_gguf.sh
#
# FLEURS test (Urdu script + Roman Urdu) in full, plus a fixed 500-clip sample of Common Voice test, so CPU decoding
# stays under an hour per model. Writes results/gguf_<quant>*.json.
set -euo pipefail
cd "$(dirname "$0")"
G=${G:-runs/urdu-0.6b/gguf}
THREADS=${THREADS:-8}
PORT=8092
URL=http://127.0.0.1:$PORT

for Q in q8_0 q4_k_m; do
  tools/llama.cpp/build/bin/llama-server -m "$G/urdu-asr-0.6b-$Q.gguf" --mmproj "$G/mmproj-urdu-asr-0.6b-f16.gguf" \
    --port $PORT -t "$THREADS" -np 4 -c 8192 > "results/gguf_${Q}_server.log" 2>&1 &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null' EXIT
  until curl -sf $URL/health > /dev/null; do sleep 1; done
  ev() {  # ev <name> [evaluate.py args...]
    local name=$1; shift
    if [ -f "results/$name.json" ]; then echo "  $name: done"; return; fi
    echo "  $name ..."
    .venv/bin/python evaluate.py --name "$name" --server $URL "$@" > "results/$name.log" 2>&1
  }
  ev "gguf_$Q"       --language Urdu         --sets test_fleurs
  ev "gguf_${Q}_roman" --language "Roman urdu" --sets test_fleurs_roman
  ev "gguf_${Q}_cv500"       --language Urdu         --sets test_cv       --limit 500
  ev "gguf_${Q}_cv500_roman" --language "Roman urdu" --sets test_cv_roman --limit 500
  kill $SERVER; wait $SERVER 2>/dev/null || true
done

.venv/bin/python - <<'EOF'
import json
from pathlib import Path
for p in sorted(Path("results").glob("gguf_*.json")):
    for s, v in json.loads(p.read_text())["sets"].items():
        print(f"{p.stem:28} {s:18} WER {v['wer']:6.1%}  CER {v['cer']:6.1%}  ({v['clips']} clips)")
EOF

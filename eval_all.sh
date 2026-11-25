#!/usr/bin/env bash
# After training: score the best checkpoint on everything. Safe to re-run: finished results are skipped.
#
#   ./eval_all.sh                                # best checkpoint of runs/urdu-0.6b (lowest dev loss)
#   CK=runs/urdu-0.6b/checkpoint-1000 TAG=_ck1000 ./eval_all.sh
#
# Urdu script and Roman Urdu on Common Voice + FLEURS, and the forgetting check: English and Hindi (FLEURS), told
# the language and left to pick it. Base-model results are computed once and reused.
set -euo pipefail
cd "$(dirname "$0")"
RUN=${RUN:-runs/urdu-0.6b}
TAG=${TAG:-}
CK=${CK:-$(.venv/bin/python - "$RUN" <<'EOF'
import json, sys
from pathlib import Path
cks = {int(p.name.split("-")[1]): p for p in Path(sys.argv[1]).glob("checkpoint-*")}
log = json.loads((cks[max(cks)] / "trainer_state.json").read_text())["log_history"]
dev = {h["step"]: h["eval_loss"] for h in log if "eval_loss" in h and h["step"] in cks}
print(cks[min(dev, key=dev.get)] if dev else cks[max(cks)])
EOF
)}
echo "Scoring $CK"

ev() {  # ev <name> <model> [evaluate.py args...]; skipped if results/<name>.json exists
  local name=$1 model=$2; shift 2
  if [ -f "results/$name.json" ]; then echo "  $name: done"; return; fi
  echo "  $name ..."
  .venv/bin/python evaluate.py --name "$name" --model "$model" "$@" > "results/$name.log" 2>&1
}
B=Qwen/Qwen3-ASR-0.6B
mkdir -p results
ev "finetuned$TAG"        "$CK" --language Urdu
ev "finetuned_roman$TAG"  "$CK" --language "Roman urdu" --sets test_cv_roman test_fleurs_roman
ev base_en                "$B"  --language English --sets test_fleurs_en
ev "finetuned_en$TAG"     "$CK" --language English --sets test_fleurs_en
ev base_hi                "$B"  --language Hindi --sets test_fleurs_hi
ev "finetuned_hi$TAG"     "$CK" --language Hindi --sets test_fleurs_hi
ev base_auto_other        "$B"  --sets test_fleurs_en test_fleurs_hi
ev "finetuned_auto_other$TAG" "$CK" --sets test_fleurs_en test_fleurs_hi

.venv/bin/python - "$TAG" <<'EOF'
import json, sys
from pathlib import Path
tag = sys.argv[1]
for n in ["base", "base_forced", f"finetuned{tag}", f"finetuned_roman{tag}", "base_en", f"finetuned_en{tag}",
          "base_hi", f"finetuned_hi{tag}", "base_auto_other", f"finetuned_auto_other{tag}"]:
    p = Path(f"results/{n}.json")
    if p.exists():
        for s, v in json.loads(p.read_text())["sets"].items():
            langs = ", ".join(f"{k} {c / v['clips']:.0%}" for k, c in list(v["detected_language"].items())[:2])
            print(f"{n:26} {s:18} WER {v['wer']:6.1%}  CER {v['cer']:6.1%}  detected: {langs}")
EOF

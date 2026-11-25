#!/usr/bin/env bash
# Step 4: full fine-tune of Qwen3-ASR-0.6B on Urdu with Qwen's official script (qwen3_asr_sft.py, plus one fix for its loss scaling, marked FIX).
# Re-running the same command resumes from the newest checkpoint in $OUT.
#
#   ./train.sh                                                      # the real run
# Once data/train_roman.jsonl exists (stage 2), it is included automatically: one model, both scripts.
#   TRAIN=data/smoke.jsonl OUT=runs/smoke EPOCHS=1 SAVE_STEPS=5 ./train.sh   # check first (README step 4)
#
# Settings (environment variables, defaults below). Effective batch = BATCH x ACCUM clips per update.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-Qwen/Qwen3-ASR-0.6B}
roman() { [ -f "data/${1}_roman.jsonl" ] && echo " data/${1}_roman.jsonl" || true; }
export TRAIN=${TRAIN:-data/train.jsonl$(roman train)}   # one or more lists, space-separated
export DEV=${DEV:-data/dev.jsonl$(roman dev)}
export OUT=${OUT:-runs/urdu-0.6b}
BATCH=${BATCH:-2}          # clips per forward pass; 4+ runs out of memory on the longest (30 s) clips
ACCUM=${ACCUM:-32}         # forward passes per update, so 64 clips per update
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less wasted memory when clip lengths vary
LR=${LR:-2e-5}
EPOCHS=${EPOCHS:-2}
SAVE_STEPS=${SAVE_STEPS:-200}   # also how often dev loss is measured

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt 12000 ]; then
  echo "Only ${free} MiB GPU memory free; training needs ~12 GB. Pause the GPU services first." >&2
  exit 1
fi

mkdir -p "$OUT"
# Qwen's script reads only "audio" and "text"; our lists carry extra columns (speaker exists for Common Voice only),
# and its JSON loader fails when train and dev columns differ. So hand it exactly the two columns it uses.
echo "Training lists: $TRAIN | dev: $DEV"
.venv/bin/python -c '
import json, os
for var, dst in (("TRAIN", "sft_train"), ("DEV", "sft_dev")):
    with open(os.path.join(os.environ["OUT"], dst + ".jsonl"), "w") as g:
        for src in os.environ[var].split():
            for line in open(src):
                r = json.loads(line)
                g.write(json.dumps({"audio": r["audio"], "text": r["text"]}, ensure_ascii=False) + "\n")
'

.venv/bin/python qwen3_asr_sft.py \
  --model_path "$MODEL" --train_file "$OUT/sft_train.jsonl" --eval_file "$OUT/sft_dev.jsonl" --output_dir "$OUT" \
  --batch_size "$BATCH" --grad_acc "$ACCUM" --lr "$LR" --epochs "$EPOCHS" \
  --log_steps 10 --save_steps "$SAVE_STEPS" --save_total_limit 3 --resume 1 \
  2>&1 | tee -a "$OUT/train.log"

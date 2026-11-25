#!/usr/bin/env bash
# Assemble the laptop kit: scripts + 20 unseen FLEURS test clips + our GGUF files -> Deliverables/urdu-asr-laptop-kit.zip
# Copy the zip to the laptop and follow LAPTOP.md inside it. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
KIT=Deliverables/urdu-asr-laptop-kit
G=${G:-runs/urdu-0.6b/gguf}
rm -rf "$KIT" && mkdir -p "$KIT/samples" "$KIT/models"
cp Deliverables/laptop/{laptop.py,demo.html,LAPTOP.md} common.py llama_client.py "$KIT/"
cp "$G"/urdu-asr-0.6b-{q8_0,q4_k_m}.gguf "$G"/mmproj-urdu-asr-0.6b-f16.gguf "$KIT/models/"

.venv/bin/python - "$KIT" <<'EOF'
import json, shutil, sys
from pathlib import Path
from common import read_jsonl, sample
kit = Path(sys.argv[1])
roman = {r["id"].removesuffix("#roman"): r["ref"] for r in read_jsonl("data/test_fleurs_roman.jsonl")}
rows = [r for r in sample(read_jsonl("data/test_fleurs.jsonl"), 0, seed=7) if r["id"] in roman and r["seconds"] <= 20][:20]
with open(kit / "samples/samples.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        name = Path(r["audio"]).name
        shutil.copy(r["audio"], kit / "samples" / name)
        f.write(json.dumps({"file": name, "seconds": r["seconds"], "ref": r["ref"], "ref_roman": roman[r["id"]],
                            "source": "FLEURS ur_pk test (CC-BY-4.0)"}, ensure_ascii=False) + "\n")
print(f"{len(rows)} clips, {sum(r['seconds'] for r in rows):.0f} s")
EOF
(cd Deliverables && rm -f urdu-asr-laptop-kit.zip && zip -qr0 urdu-asr-laptop-kit.zip urdu-asr-laptop-kit)
du -sh Deliverables/urdu-asr-laptop-kit.zip

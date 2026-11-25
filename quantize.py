"""Step 7: shrink the fine-tuned model to GGUF for llama.cpp (CPU, laptops). Safe to re-run.

    python quantize.py                                    # best checkpoint of runs/urdu-0.6b
    python quantize.py --ck runs/urdu-0.6b/checkpoint-1800

Needs llama.cpp built in tools/llama.cpp (README, "Quantize"). Writes runs/<run>/gguf/:
  urdu-asr-0.6b-{f16,q8_0,q4_k_m}.gguf   the language model (decoder), three sizes
  mmproj-urdu-asr-0.6b-f16.gguf          the audio encoder; kept at 16 bit (small, and audio quality suffers first)

Converted from the qwen_asr checkpoint as is: llama.cpp reads that layout (thinker.*) directly.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluate import complete_checkpoint

LLAMA = Path("tools/llama.cpp")
NAME = "urdu-asr-0.6b"


def best_checkpoint(run):
    cks = {int(p.name.split("-")[1]): p for p in Path(run).glob("checkpoint-*")}
    log = json.loads((cks[max(cks)] / "trainer_state.json").read_text())["log_history"]
    dev = {h["step"]: h["eval_loss"] for h in log if "eval_loss" in h and h["step"] in cks}
    return cks[min(dev, key=dev.get)] if dev else cks[max(cks)]


def run(*cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/urdu-0.6b")
    ap.add_argument("--ck", help="checkpoint folder (default: lowest dev loss)")
    a = ap.parse_args()
    ck = Path(a.ck) if a.ck else best_checkpoint(a.run)
    out = Path(a.run) / "gguf"
    out.mkdir(exist_ok=True)
    complete_checkpoint(ck)  # the converter needs preprocessor_config.json
    convert = [sys.executable, LLAMA / "convert_hf_to_gguf.py", ck, "--outtype", "f16"]
    f16, mmproj = out / f"{NAME}-f16.gguf", out / f"mmproj-{NAME}-f16.gguf"
    if not f16.exists():
        run(*convert, "--outfile", f16)
    if not mmproj.exists():
        run(*convert, "--mmproj", "--outfile", mmproj)
    for q in ("q8_0", "q4_k_m"):
        if not (dest := out / f"{NAME}-{q}.gguf").exists():
            run(LLAMA / "build/bin/llama-quantize", f16, dest, q.upper())
    (out / "SOURCE.txt").write_text(f"{ck}\n")
    for p in sorted(out.glob("*.gguf")):
        print(f"{p.stat().st_size / 1e6:8.0f} MB  {p}")


if __name__ == "__main__":
    main()

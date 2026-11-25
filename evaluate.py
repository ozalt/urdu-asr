"""Steps 3 and 5: transcribe the test sets with a model and score it. Same code for "before" and "after".

    python evaluate.py --name base                                        # untouched Qwen3-ASR-0.6B
    python evaluate.py --name finetuned --model runs/urdu-0.6b/checkpoint-XXXX --language Urdu
    python evaluate.py --name finetuned_roman --model runs/urdu-0.6b/checkpoint-XXXX \
        --language "Roman urdu" --sets test_cv_roman test_fleurs_roman       # stage 2: Roman Urdu output

A model trained on both scripts needs --language to pick one. The Roman Urdu references were written by the
transliterator (romanize/), not by people, so those scores measure agreement with it.

Quantized GGUF models run in llama.cpp; start llama-server (README, "Quantize"), then score through it:
    python evaluate.py --name gguf_q8 --server http://127.0.0.1:8080 --language Urdu --sets test_fleurs

Writes results/<name>.json (scores) and results/<name>_<set>.jsonl (every prediction, for reading errors).
Decoding is greedy (no randomness), so re-running gives the same numbers.
"""

import argparse
import json
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jiwer
import torch
from huggingface_hub import hf_hub_download
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference import utils as qwen_utils

import llama_client
from common import LANG, LANG_ROMAN, for_scoring, read_jsonl, sample, script_of, write_jsonl

for lang in (LANG, LANG_ROMAN):  # lets --language force our new languages on a fine-tuned model
    if lang not in qwen_utils.SUPPORTED_LANGUAGES:
        qwen_utils.SUPPORTED_LANGUAGES.append(lang)


def complete_checkpoint(path, base="Qwen/Qwen3-ASR-0.6B"):
    """Qwen's training script saves the weights and tokenizer but not the audio preprocessor config and chat
    template, which training doesn't change. Copy them from the base model so the checkpoint loads on its own."""
    if Path(path).is_dir():
        for f in ("preprocessor_config.json", "chat_template.json"):
            if not (Path(path) / f).exists():
                shutil.copy(hf_hub_download(base, f), Path(path) / f)


def score(rows, preds):
    refs = [for_scoring(r["ref"]) for r in rows]
    hyps = [for_scoring(p.text) for p in preds]
    return {"clips": len(rows),
            "wer": round(jiwer.wer(refs, hyps), 4),   # word error rate: share of words wrong (lower is better)
            "cer": round(jiwer.cer(refs, hyps), 4),   # character error rate: kinder to spelling variants
            "output_script": dict(Counter(script_of(p.text) for p in preds).most_common()),
            "detected_language": dict(Counter(p.language or "none" for p in preds).most_common())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="label for the result files")
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B", help="HF model ID or a checkpoint folder")
    ap.add_argument("--sets", nargs="+", default=["test_cv", "test_fleurs"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--language", default=None, help="force the output language (default: model decides)")
    ap.add_argument("--limit", type=int, default=0, help="score a fixed random sample of N clips per set")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--server", help="score a GGUF model through this llama-server URL instead of --model")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.server:  # llama-server decodes greedily too; a few requests at once keep its parallel slots busy
        pool = ThreadPoolExecutor(4)

        def transcribe(audio, language):
            return list(pool.map(lambda w: llama_client.transcribe(w, language, a.server), audio))
        a.model = llama_client.server_model(a.server)
    else:
        complete_checkpoint(a.model)
        model = Qwen3ASRModel.from_pretrained(a.model, dtype=torch.bfloat16, device_map=a.device,
                                              max_inference_batch_size=a.batch, max_new_tokens=256)
        model.model.generation_config.do_sample = False  # greedy: reproducible
        transcribe = model.transcribe

    result = {"name": a.name, "model": a.model, "language": a.language, "sets": {}}
    for name in a.sets:
        path = Path(a.data) / f"{name}.jsonl"
        if not path.exists():
            print(f"{name}: {path} missing, skipped")
            continue
        rows = sample(read_jsonl(path), a.limit)
        preds, t0 = [], time.time()
        for i in range(0, len(rows), a.batch):
            preds += transcribe(audio=[r["audio"] for r in rows[i:i + a.batch]], language=a.language)
            left = (time.time() - t0) / len(preds) * (len(rows) - len(preds)) / 60
            print(f"  {name}: {len(preds)}/{len(rows)} clips, ~{left:.1f} min left", flush=True)
        result["sets"][name] = score(rows, preds)
        write_jsonl(Path("results") / f"{a.name}_{name}.jsonl",
                    [{"id": r["id"], "ref": r["ref"], "pred": p.text, "language": p.language}
                     for r, p in zip(rows, preds)])

    Path("results").mkdir(exist_ok=True)
    Path(f"results/{a.name}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

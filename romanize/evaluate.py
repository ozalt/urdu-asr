"""Stage 2, steps 2 and 4: score a model at writing Urdu in Roman letters (the "before" and "after" numbers).

    romanize/.venv/bin/python romanize/evaluate.py --name base
    romanize/.venv/bin/python romanize/evaluate.py --name finetuned --model runs/romanize/final

Test set: Dakshina, 4,880 Urdu sentences romanized by native speakers. Roman Urdu has no standard spelling
("nahi" / "nahin"), so no model can score perfectly against a single human reference; compare models, not to 0.
Scores, after lowercasing and removing punctuation:
  cer          character error rate, lower is better
  wer          word error rate, lower is better
  char_bleu    BLEU over characters, higher is better; the metric of the published results on this test set
  chrf         character n-gram F-score, higher is better
Writes romanize/results/<name>.json and romanize/results/<name>_<set>.jsonl (every output next to its reference).
"""

import argparse
import json
import sys
from pathlib import Path

import jiwer
import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import for_scoring, read_jsonl, sample, write_jsonl  # noqa: E402

# The same instruction for the base model and the fine-tuned one, so the comparison is fair.
PROMPT = ("Write this Urdu text in Roman Urdu (Urdu in Latin letters, the way Pakistanis type it). "
          "Reply with the Roman Urdu only.\n\n{}")


def prompt_text(tok, ur):
    return tok.apply_chat_template([{"role": "user", "content": PROMPT.format(ur)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def load(model):
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16,
                                             device_map="cuda:0" if torch.cuda.is_available() else "cpu")
    return tok, m.eval()


def stop_ids(tok, model):
    """Stop at the end of the answer. The chat template ends a turn with <|im_end|> (the tokenizer's eos), but the
    model config's eos is <|endoftext|>; without both, generation runs on ("...\\nassistant\\n...") to the limit."""
    ids = model.generation_config.eos_token_id
    return sorted({tok.eos_token_id, *(ids if isinstance(ids, list) else [ids] if ids is not None else [])})


@torch.inference_mode()
def romanize(tok, model, texts, batch=64):
    """Greedy decoding. Sorted by length so each batch has little padding; results come back in input order."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([prompt_text(tok, texts[i]) for i in idx], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=2 * enc.input_ids.shape[1] + 16,
                             eos_token_id=stop_ids(tok, model), pad_token_id=tok.pad_token_id)
        for i, g in zip(idx, gen[:, enc.input_ids.shape[1]:]):
            out[i] = tok.decode(g, skip_special_tokens=True).strip()
        print(f"  {min(s + batch, len(order))}/{len(order)}", end="\r", flush=True)
    return out


def scores(refs, preds):
    r, p = [for_scoring(x) or "-" for x in refs], [for_scoring(x) or "-" for x in preds]
    return {"cer": round(jiwer.cer(r, p), 4), "wer": round(jiwer.wer(r, p), 4),
            "char_bleu": round(sacrebleu.corpus_bleu(p, [r], tokenize="char").score, 2),
            "chrf": round(sacrebleu.corpus_chrf(p, [r]).score, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--set", default="test_dakshina")
    ap.add_argument("--limit", type=int, default=0, help="score a fixed random sample of this many sentences")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    rows = sample(read_jsonl(f"romanize/data/{a.set}.jsonl"), a.limit)
    tok, model = load(a.model)
    preds = romanize(tok, model, [r["ur"] for r in rows], a.batch)
    res = {"name": a.name, "model": a.model, "set": a.set, "sentences": len(rows),
           **scores([r["roman"] for r in rows], preds)}
    Path("romanize/results").mkdir(exist_ok=True)
    Path(f"romanize/results/{a.name}.json").write_text(json.dumps(res, indent=1))
    write_jsonl(f"romanize/results/{a.name}_{a.set}.jsonl",
                [{"ur": r["ur"], "ref": r["roman"], "pred": p} for r, p in zip(rows, preds)])
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

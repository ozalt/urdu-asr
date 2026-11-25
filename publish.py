"""Step 9: publish the fine-tuned model to Hugging Face (needs HF_TOKEN=... in .env). Safe to re-run.

    python publish.py --ck runs/urdu-0.6b/checkpoint-1800          # weights + GGUF + model card
    python publish.py --ck ... --card-only                          # write runs/.../publish/README.md, upload nothing

Also mirrors to ModelScope when MODELSCOPE_TOKEN is in .env (international accounts: MODELSCOPE_ENDPOINT=https://www.modelscope.ai).

Uploads only what inference needs (no optimizer state), the GGUF files from quantize.py, and a model card built from
results/*.json. Never uploads training data: Common Voice terms forbid re-hosting it.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

from evaluate import complete_checkpoint
from romanize.publish import token

REPO = "qwen3-asr-0.6b-urdu"
GITHUB = "https://github.com/ozalt/urdu-asr"
KEEP = ["model.safetensors", "config.json", "generation_config.json", "preprocessor_config.json", "chat_template.json",
        "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json",
        "special_tokens_map.json"]


def res(name, s):
    p = Path(f"results/{name}.json")
    return json.loads(p.read_text())["sets"].get(s) if p.exists() else None


def cell(r):
    return f"{100 * r['wer']:.1f}% / {100 * r['cer']:.1f}%" if r else "not run"


def card(user, gguf):
    rows = [("Qwen3-ASR-0.6B, auto language (writes Hindi script)", "base", "test_cv", "test_fleurs"),
            ("Qwen3-ASR-0.6B, told Urdu", "base_forced", "test_cv", "test_fleurs"),
            ("**This model, Urdu script**", "finetuned", "test_cv", "test_fleurs"),
            ("**This model, Roman Urdu**", "finetuned_roman", "test_cv_roman", "test_fleurs_roman")]
    table = "\n".join(f"| {label} | {cell(res(n, cv))} | {cell(res(n, fl))} |" for label, n, cv, fl in rows)
    quant = "\n".join(
        f"| {q} | {sizes.get(q, '?')} MB | {cell(res(f'gguf_{q}', 'test_fleurs'))} | "
        f"{cell(res(f'gguf_{q}_roman', 'test_fleurs_roman'))} |"
        for q, sizes in [(q, {q: round((gguf / f"urdu-asr-0.6b-{q}.gguf").stat().st_size / 1e6)})
                         for q in ("q8_0", "q4_k_m") if (gguf / f"urdu-asr-0.6b-{q}.gguf").exists()])
    forget = "\n".join(f"| {label} | {cell(res(b, s))} | {cell(res(f, s))} |" for label, b, f, s in [
        ("English, told English", "base_en", "finetuned_en", "test_fleurs_en"),
        ("Hindi, told Hindi", "base_hi", "finetuned_hi", "test_fleurs_hi")])
    return f"""---
license: apache-2.0
language:
- ur
base_model: Qwen/Qwen3-ASR-0.6B
pipeline_tag: automatic-speech-recognition
tags:
- urdu
- roman-urdu
- speech-recognition
- qwen3-asr
- gguf
- llama.cpp
datasets:
- google/fleurs
- {user}/urdu-speech-transcripts-roman
---

# Qwen3-ASR-0.6B Urdu (Urdu script and Roman Urdu)

Urdu speech to text that runs offline on a laptop CPU. One model writes either **Urdu script** or **Roman Urdu**
(Urdu in Latin letters, the way most Pakistanis type), chosen by a language tag.

A full fine-tune of [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B). Left to itself, the base model
writes Urdu speech in Hindi script; told the language is Urdu, it still gets about half the words wrong. This model
halves that error and adds Roman Urdu.

## Results

Word error rate / character error rate on test sets never seen in training, after removing punctuation (lower is
better). Common Voice test: 5,071 clips from speakers not in training. FLEURS test: 299 clips, different sentences,
speakers and microphones.

| Model | Common Voice | FLEURS |
| --- | --- | --- |
{table}

Roman Urdu references are the human Urdu transcripts romanized by our
[romanizer](https://huggingface.co/{user}/qwen3.5-0.8b-urdu-romanizer); Roman Urdu has no fixed spelling, so its
scores overstate real mistakes.

### Quantized (GGUF, llama.cpp)

Decoder sizes below; all use the same 379 MB audio encoder (`mmproj-urdu-asr-0.6b-f16.gguf`). FLEURS test, CPU.

| File | Decoder size | Urdu | Roman Urdu |
| --- | --- | --- | --- |
{quant}

## Use

### llama.cpp (CPU, laptops)

```bash
llama-server -m urdu-asr-0.6b-q8_0.gguf --mmproj mmproj-urdu-asr-0.6b-f16.gguf --port 8080
```

Always set the language: send a raw `/completion` prompt ending in `language Urdu<asr_text>` or
`language Roman urdu<asr_text>`. `llama_client.py` (in this repo, standard library only) does that:

```bash
python llama_client.py clip.wav --language Urdu
python llama_client.py clip.wav --language "Roman urdu"
```

### Python (qwen-asr)

```python
import torch
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference import utils
utils.SUPPORTED_LANGUAGES += ["Urdu", "Roman urdu"]

model = Qwen3ASRModel.from_pretrained("{user}/{REPO}", dtype=torch.bfloat16, device_map="cuda:0")
print(model.transcribe(audio=["clip.wav"], language="Urdu")[0].text)
print(model.transcribe(audio=["clip.wav"], language="Roman urdu")[0].text)
```

For a one-command laptop setup (benchmark + browser demo with microphone) see the
[laptop kit]({GITHUB}/tree/main/Deliverables/laptop). Code, training scripts and the full write-up: [{GITHUB}]({GITHUB}).

## Training

- Data: Mozilla Common Voice Urdu (33,046 clips) + FLEURS Urdu train (2,103 clips), 43.5 hours. Each clip twice: with
  its Urdu transcript and with a Roman Urdu transcript from our romanizer (70,293 examples).
- Full fine-tune with the official Qwen3-ASR SFT script, 2 epochs, learning rate 2e-5, effective batch 64, bf16.
  Kept the checkpoint with the lowest dev loss (update 1,800 of 2,198).
- Hardware: one NVIDIA RTX 5060 Ti 16 GB, about 2.2 hours.

## Limitations

- **Set the language.** Trained only on Urdu tags, the model now labels most speech as Urdu when left to detect it.
- Other languages degrade (catastrophic forgetting). FLEURS word error rate, base vs this model:

| Language | Base | This model |
| --- | --- | --- |
{forget}

  Hindi suffers most: spoken Hindi and Urdu are nearly the same language, and about half the Hindi outputs come out in
  Urdu script.
- English loanwords and names are often misspelled, especially in Roman Urdu.
- Read speech (Common Voice, FLEURS) dominates training; noisy, fast or conversational speech will be harder.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--card-only", action="store_true")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    ck, gguf = Path(a.ck), Path(a.ck).parent / "gguf"
    api = HfApi(token=token())
    user = api.whoami()["name"]
    out = ck.parent / "publish"
    out.mkdir(exist_ok=True)
    complete_checkpoint(ck)
    for f in KEEP:
        if (ck / f).exists() and not (out / f).exists():
            (out / f).hardlink_to(ck / f)
    for f in gguf.glob("*.gguf"):
        if "f16.gguf" in f.name and "mmproj" not in f.name:
            continue  # the 16-bit decoder is just the safetensors again
        if not (out / f.name).exists():
            (out / f.name).hardlink_to(f)
    shutil.copy("llama_client.py", out / "llama_client.py")
    (out / "README.md").write_text(card(user, gguf))
    print(f"{out}: " + ", ".join(sorted(p.name for p in out.iterdir())))
    if a.card_only:
        return
    repo = f"{user}/{REPO}"
    api.create_repo(repo, private=a.private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo)
    print(f"https://huggingface.co/{repo}")
    if ms_token := token("MODELSCOPE_TOKEN"):
        from modelscope.hub.api import HubApi
        ms = HubApi()
        ms.login(ms_token)
        ms.create_repo(repo, repo_type="model", visibility="private" if a.private else "public",
                       license="apache-2.0", exist_ok=True)
        ms.upload_folder(repo_id=repo, folder_path=str(out), repo_type="model")
        print(f"{os.environ.get('MODELSCOPE_ENDPOINT', 'https://modelscope.cn')}/models/{repo}")


if __name__ == "__main__":
    main()

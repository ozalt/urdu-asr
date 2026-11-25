"""Publish to Hugging Face (needs HF_TOKEN=... in .env). Safe to re-run: it updates the same repos.

    romanize/.venv/bin/python romanize/publish.py dataset    # romanize/data_full -> <user>/roman-urdu-parl-clean
    romanize/.venv/bin/python romanize/publish.py model      # runs/romanize/final -> <user>/qwen3.5-0.8b-urdu-romanizer
    romanize/.venv/bin/python romanize/publish.py labels     # data/roman_labels.jsonl -> <user>/urdu-speech-transcripts-roman
                                                              # (Hugging Face, plus ModelScope if MODELSCOPE_TOKEN is in .env;
                                                              #  international accounts: MODELSCOPE_ENDPOINT=https://www.modelscope.ai)

Add --private to keep a repo private. The dataset needs `prepare.py --out romanize/data_full --train-size 0` first;
the model needs train.py and `evaluate.py --name base` / `--name finetuned` first (their scores go on the card).
Never uploaded: Dakshina (CC-BY-SA, Google hosts it) and anything derived from Common Voice (no re-hosting).
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

DATASET, MODEL, LABELS = "roman-urdu-parl-clean", "qwen3.5-0.8b-urdu-romanizer", "urdu-speech-transcripts-roman"
# Common Voice's sentence list, published separately under CC0. The Common Voice *dataset* may not be re-shared,
# so only transcripts that appear in this public list are uploaded.
CV_SENTENCES = "https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/ur/sentence-collector.txt"
RUP_CITATION = ("Mehreen Alam and Sibt ul Hussain. 2022. Roman-Urdu-Parl: Roman-Urdu and Urdu Parallel Corpus for "
                "Urdu Language Understanding. ACM Transactions on Asian and Low-Resource Language Information "
                "Processing 21(1).")


def token(name="HF_TOKEN"):
    for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return os.environ.get(name)


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dataset_card(train, dev):
    ex = "\n".join(f"| {r['ur']} | {r['roman']} |" for r in random.Random(3).sample(train, 6))
    return f"""---
license: apache-2.0
language:
- ur
task_categories:
- translation
tags:
- transliteration
- roman-urdu
- urdu
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
---

# Roman-Urdu-Parl, cleaned

{len(train):,} Urdu ↔ Roman Urdu sentence pairs (plus {len(dev):,} for validation), filtered from
[Roman-Urdu-Parl](https://huggingface.co/datasets/Mavkif/Roman-Urdu-Parl-split) (6.3M rows). Roman Urdu is Urdu
typed in Latin letters, the way most Pakistanis write on their phones.

| Urdu | Roman Urdu |
| --- | --- |
{ex}

## Why a cleaned version

Roman-Urdu-Parl was enlarged by machine-substituting word variants, and that broke a large share of rows. For many
common words the *wrong* romanization is the majority: یہ (yeh) appears as "ki" in 70% of its rows, ہاتھ (haath) as
"ke" in 85%, کہوں (kahoon) as "kaisay" in 95%, رانا (Rana) as "pakistani" in 63%. Frequency-based filtering cannot
fix errors that are the majority, so this version checks *sound* instead.

## How it was cleaned

1. Exact duplicates removed; text normalized (Urdu letter variants, invisible characters, whitespace).
2. Both sides in their own script, the same number of words (so words line up one-to-one), and no ALL-CAPS words
   (in this corpus they mark machine-inserted variants: "KA", "HAI").
3. **Every word pair must sound alike**:
   - its consonants match closely (vowels, h, و/ی ignored, since Roman Urdu spells those freely; ش "sh" kept apart
     from س "s"): یہ/"yeh" passes; رانا/"pakistani", دنیا/"ne", سکتے/"satke", خوشی/"khusi" fail;
   - its first sound matches: ہی/"hi" passes, ہی/"yee" fails; جاتے/"jatay" passes, جاتے/"atay" fails;
   - a و inside the Urdu word appears as o/u/w/v: کوئی/"koi" passes, کوئی/"ki" fails;
   - no machine artifacts such as "kkh" or "fh" (خدا/"kkhuda", محفل/"mehfhil");
   - ملک is "mulk" (country) or the name "Malik"; lowercase "malik", the corpus's substitution in 84% of its rows,
     is rejected.

   These rules were found by auditing the 300 most frequent words: before them, the *wrong* form was the majority
   for دنیا, سکتے, جاتے, ملک, خوشی and خدا; after them, the correct form is (duniya 100%, khushi 96%, mulk 79%).
4. Sentences from the [Dakshina](https://github.com/google-research-datasets/dakshina) Urdu dev/test sets were
   removed, so Dakshina stays a clean test set for models trained on this data.

About 45% of the unique pairs pass. The code is in `scripts/prepare.py` (run with `--train-size 0`).

## Quality

Common Urdu words are now romanized correctly (see the audit above). A spot check of 40 random rows found the
remaining flaws in about 1 row in 5, almost all of one kind: **English loanwords and foreign names spelled letter by
letter** (کینٹکی "kintki" for Kentucky, ایڈیٹوریل "ayditoril" for editorial, ممبروں "mmbron" for members), plus a few
odd short-word spellings (یہ as "yy"). Spelling also varies by writer ("hamaray" / "hmary"), which is normal for
Roman Urdu. Text comes from the original corpus: news, Wikipedia, poetry, blog and forum comments.

## Use

```python
from datasets import load_dataset
ds = load_dataset("{{repo}}")
```

Rows are in random order. Used to train
[qwen3.5-0.8b-urdu-romanizer](https://huggingface.co/{{user}}/{MODEL}) on the first rows of `train`.

## Source and license

Derived from Roman-Urdu-Parl via the
[Mavkif/Roman-Urdu-Parl-split](https://huggingface.co/datasets/Mavkif/Roman-Urdu-Parl-split) re-split
(Apache-2.0), whose train/validation separation is kept. Please cite the original corpus:

> {RUP_CITATION}
"""


def model_card(info, base, ft):
    def row(name, r):
        return f"| {name} | {100 * r['cer']:.1f}% | {100 * r['wer']:.1f}% | {r['char_bleu']:.1f} | {r['chrf']:.1f} |"
    return f"""---
license: apache-2.0
language:
- ur
base_model: Qwen/Qwen3.5-0.8B
pipeline_tag: text-generation
library_name: transformers
tags:
- transliteration
- roman-urdu
- urdu
- lora
datasets:
- {{user}}/{DATASET}
---

# Qwen3.5-0.8B Urdu → Roman Urdu romanizer

Writes Urdu (Perso-Arabic script) in Roman Urdu, the Latin-letter Urdu most Pakistanis type:

> اپنا ملک چھوڑ کر پاکستان آ گیا ہو → apna mulk chhor kar pakistan aa gaya ho

[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) fine-tuned with LoRA (rank {info['rank']}, all linear
layers, merged into the weights) on {info['train_pairs']:,} pairs of
[{{user}}/{DATASET}](https://huggingface.co/datasets/{{user}}/{DATASET}), {info['epochs']:g} epoch,
learning rate {info['lr']}, 64 sentences per update. Trained on one RTX 5060 Ti (16 GB).

## Results

Test set: [Dakshina](https://github.com/google-research-datasets/dakshina) Urdu, {ft['sentences']:,} sentences
romanized by native speakers (sentences up to 80 words), never seen in training. Scores after lowercasing and
removing punctuation.

| Model | CER ↓ | WER ↓ | char-BLEU ↑ | chrF ↑ |
| --- | --- | --- | --- | --- |
{row("Qwen3.5-0.8B, untouched (same prompt)", base)}
| GPT-4o mini, zero-shot \\* | — | — | 75.1 | 68.0 |
| m2m100, trained on Roman-Urdu-Parl \\* | — | — | 68.7 | 61.8 |
| m2m100, trained on Roman-Urdu-Parl + Dakshina data \\* | — | — | 78.5 | — |
{row("**This model** (0.8B, 50k training pairs)", ft)}

\\* Reported in [arXiv 2503.21530](https://arxiv.org/abs/2503.21530) (m2m100 rows: the paper's models without MLM
pre-training, except the last, its best Dakshina result), not re-run by us. That paper's Dakshina test split and
text normalization may differ from ours (it lists a 500-sentence test set), so treat those rows as indicative.
"—" = not reported.

Roman Urdu has no standard spelling ("nahi" / "nahin"), so no system matches a single human reference exactly:
compare systems, don't expect 0% error.

## Use

Needs `transformers` 5.x (Qwen3.5 support). Use exactly this prompt; it is the one the model was trained with.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{{user}}/{MODEL}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")

PROMPT = ("Write this Urdu text in Roman Urdu (Urdu in Latin letters, the way Pakistanis type it). "
          "Reply with the Roman Urdu only.\\n\\n{{}}")
text = tok.apply_chat_template([{{"role": "user", "content": PROMPT.format("یہ ایک سوالیہ نشان بنا ہوا ہے")}}],
                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
```

## Limitations

- Trained on text from news, Wikipedia, poetry and forums: its spelling follows that mix, not one standard.
- Rare names and English loanwords are the most common errors: they come out spelled letter by letter
  (کالج "kalij" instead of "college", فریڈ "free" instead of "Fred").
- Urdu script usually omits short vowels, so some words are ambiguous without context (چین: Cheen/China or chain/peace).
- Built as a helper: it writes Roman Urdu training labels for an Urdu speech-recognition model.
"""


def label_rows():
    """Roman Urdu labels of the speech transcripts, split by how the ASR model used each sentence."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common import ROMAN_CHECKS, for_scoring, read_jsonl
    with urllib.request.urlopen(CV_SENTENCES) as r:
        public = {for_scoring(line) for line in r.read().decode().splitlines()}
    use = {}  # sentence -> (source, split); a sentence used in any test set counts as test
    rank = {"train": 0, "validation": 1, "test": 2}
    for f, split in (("train", "train"), ("dev", "validation"), ("test_cv", "test"), ("test_fleurs", "test")):
        for r in read_jsonl(f"data/{f}.jsonl"):
            source, old = r["source"], use.get(r["ref"], (None, "train"))[1]
            use[r["ref"]] = (source, split if rank[split] >= rank[old] else old)
    out, dropped = {"train": [], "validation": [], "test": []}, 0
    for x in read_jsonl("data/roman_labels.jsonl"):
        source, split = use.get(x["ur"], (None, None))
        if source is None or (source == "common_voice" and for_scoring(x["ur"]) not in public):
            dropped += 1
            continue
        out[split].append({"ur": x["ur"], "roman": x["roman"], "source": source,
                           "passed_checks": not any(bad(x["ur"], x["roman"]) for bad in ROMAN_CHECKS.values())})
    return out, dropped


def labels_card(rows, dropped):
    allrows = [r for v in rows.values() for r in v]
    by = {s: sum(r["source"] == s for r in allrows) for s in ("common_voice", "fleurs")}
    ex = "\n".join(f"| {r['ur']} | {r['roman']} |" for r in random.Random(5).sample(rows["train"], 6))
    counts = " · ".join(f"{k}: {len(v):,}" for k, v in rows.items())
    return f"""---
license: cc-by-4.0
language:
- ur
task_categories:
- translation
- automatic-speech-recognition
tags:
- roman-urdu
- urdu
- transliteration
- synthetic
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
---

# Urdu speech transcripts in Roman Urdu (synthetic)

{len(allrows):,} Urdu speech transcripts with a **Roman Urdu** version (Urdu in Latin letters, the way most Pakistanis
type). No public Urdu speech corpus has Roman Urdu transcripts, so these were written by our romanizer,
[{{user}}/{MODEL}](https://huggingface.co/{{user}}/{MODEL}), to train a speech recognizer that can write either script.

| Urdu (human) | Roman Urdu (model) |
| --- | --- |
{ex}

## Contents

Each row: `ur` (human transcript), `roman` (romanizer output), `source` (`common_voice` or `fleurs`),
`passed_checks` (see below). Text only: no audio.

Splits follow how our speech model used each sentence ({counts}). A sentence in any of our test sets is in `test`;
leave it out of training if you want to compare with our results.

| Source | Sentences | License |
| --- | --- | --- |
| [Common Voice](https://commonvoice.mozilla.org) Urdu sentences | {by['common_voice']:,} | CC0 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) Urdu (ur_pk) transcripts | {by['fleurs']:,} | CC-BY-4.0 |

Only Common Voice sentences that Mozilla publishes in its CC0 sentence list
([`server/data/ur`](https://github.com/common-voice/common-voice/tree/main/server/data/ur)) are included; the Common
Voice dataset itself may not be re-shared, so {dropped:,} transcripts not in that list were left out.

## How it was made

1. Distinct transcripts from our Common Voice and FLEURS training, validation and test lists.
2. Romanized by [{{user}}/{MODEL}](https://huggingface.co/{{user}}/{MODEL}) (Qwen3.5-0.8B fine-tuned on
   [{{user}}/{DATASET}](https://huggingface.co/datasets/{{user}}/{DATASET})), greedy decoding.
3. Automatic checks (`passed_checks` is false if any fails): Urdu letters left in the output, a length ratio outside
   0.7 to 2.5, a word count far from the Urdu one, or a phrase repeated three times. Rows that fail were not used
   for training. Script: `scripts/common.py` (`ROMAN_CHECKS`).

## Quality

Synthetic, not human-checked row by row. The romanizer scores char-BLEU 76.2 / chrF 69.2 on the human-romanized
Dakshina test set. Common words are reliable; **English loanwords and names are often spelled letter by letter**, and
spelling follows the romanizer's habits ("nahi", "hai", "mein"), not one standard: Roman Urdu has none.

## Pairing with audio

Audio is not included. Get it from the sources (Common Voice via the
[Mozilla Data Collective](https://mozilladatacollective.com), FLEURS from Hugging Face) and join on the Urdu text.
Our speech model, trained on these pairs: [{{user}}/qwen3-asr-0.6b-urdu](https://huggingface.co/{{user}}/qwen3-asr-0.6b-urdu).

## Attribution

FLEURS: Conneau et al., 2022, "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech",
CC-BY-4.0. Common Voice sentences: Mozilla, CC0.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["dataset", "model", "labels"])
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="labels: build the files and card, upload nothing")
    ap.add_argument("--modelscope-user", help="ModelScope account (default: same as Hugging Face)")
    a = ap.parse_args()
    api = HfApi(token=token())
    user = api.whoami()["name"]

    if a.what == "dataset":
        src, repo = Path("romanize/data_full"), f"{user}/{DATASET}"
        train, dev = rows(src / "train.jsonl"), rows(src / "dev.jsonl")
        api.create_repo(repo, repo_type="dataset", private=a.private, exist_ok=True)
        card = dataset_card(train, dev).replace("{repo}", repo).replace("{user}", user)
        for data, path in ((card.encode(), "README.md"), (str(src / "train.jsonl"), "train.jsonl"),
                           (str(src / "dev.jsonl"), "validation.jsonl"),
                           ("romanize/prepare.py", "scripts/prepare.py"), ("common.py", "scripts/common.py")):
            print("uploading", path, flush=True)
            api.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=repo, repo_type="dataset")
        print(f"https://huggingface.co/datasets/{repo}")
    elif a.what == "labels":
        rows, dropped = label_rows()
        out, repo = Path("runs/labels_publish"), f"{user}/{LABELS}"
        out.mkdir(parents=True, exist_ok=True)
        for split, data in rows.items():
            with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
        (out / "README.md").write_text(labels_card(rows, dropped).replace("{user}", user))
        (out / "scripts").mkdir(exist_ok=True)
        for f in ("common.py", "romanize/label.py", "romanize/publish.py"):
            (out / "scripts" / Path(f).name).write_bytes(Path(f).read_bytes())
        print({k: len(v) for k, v in rows.items()}, f"left out: {dropped}")
        if a.dry_run:
            return print(f"dry run: see {out}")
        api.create_repo(repo, repo_type="dataset", private=a.private, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=repo, repo_type="dataset")
        print(f"https://huggingface.co/datasets/{repo}")
        if ms_token := token("MODELSCOPE_TOKEN"):
            from modelscope.hub.api import HubApi
            ms = HubApi()
            ms.login(ms_token)
            ms_repo = f"{a.modelscope_user or user}/{LABELS}"
            ms.create_repo(ms_repo, repo_type="dataset", visibility="private" if a.private else "public",
                           license="cc-by-4.0", exist_ok=True)
            ms.upload_folder(repo_id=ms_repo, folder_path=str(out), repo_type="dataset")
            print(f"{os.environ.get('MODELSCOPE_ENDPOINT', 'https://modelscope.cn')}/datasets/{ms_repo}")
    else:
        src, repo = Path("runs/romanize/final"), f"{user}/{MODEL}"
        info = json.loads((src / "training_info.json").read_text())
        res = {n: json.loads(Path(f"romanize/results/{n}.json").read_text()) for n in ("base", "finetuned")}
        api.create_repo(repo, private=a.private, exist_ok=True)
        (src / "README.md").write_text(model_card(info, res["base"], res["finetuned"]).replace("{user}", user))
        api.upload_folder(folder_path=str(src), repo_id=repo, ignore_patterns=["training_info.json"])
        print(f"https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()

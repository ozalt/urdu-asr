"""Stage 2, step 5: write a Roman Urdu version of every speech transcript, using the trained transliterator.

    romanize/.venv/bin/python romanize/label.py          (then: .venv/bin/python prepare.py)

Reads the human Urdu transcripts in data/{train,dev,test_cv,test_fleurs}.jsonl and appends {"ur", "roman"} for
each distinct sentence to data/roman_labels.jsonl. Safe to re-run: sentences already labeled are skipped.
prepare.py then turns these into the Roman Urdu training/test lists (data/*_roman.jsonl).

At the end it checks every label automatically and writes 50 random ones to data/roman_labels_sample.txt to read.
`--report` runs only the checks.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))
from common import ROMAN_CHECKS, read_jsonl  # noqa: E402
from evaluate import load, romanize  # noqa: E402


def report(path):
    labels = read_jsonl(path)
    print(f"\nChecked {len(labels):,} labels:")
    flagged = set()
    for name, bad in ROMAN_CHECKS.items():
        hits = [x for x in labels if bad(x["ur"], x["roman"])]
        flagged |= {x["ur"] for x in hits}
        print(f"  {len(hits):5,} ({len(hits) / len(labels):.1%})  {name}")
        for x in hits[:3]:
            print(f"           {x['ur'][:50]}  ->  {x['roman'][:60]}")
    print(f"  {len(flagged):5,} ({len(flagged) / len(labels):.1%})  flagged by any check")
    sample = Path(path).with_name("roman_labels_sample.txt")
    sample.write_text("\n\n".join(f"{x['ur']}\n{x['roman']}" for x in random.Random(0).sample(labels, min(50, len(labels)))))
    print(f"50 random labels to read: {sample}")
    print("Flagged labels are skipped by prepare.py: those clips get no Roman Urdu copy.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/romanize/final")
    ap.add_argument("--labels", default="data/roman_labels.jsonl")
    ap.add_argument("--chunk", type=int, default=2048, help="sentences per save, so an interruption loses little")
    ap.add_argument("--report", action="store_true", help="only check the existing labels")
    a = ap.parse_args()
    if a.report:
        return report(a.labels)

    done = {r["ur"] for r in read_jsonl(a.labels)} if Path(a.labels).exists() else set()
    todo = sorted({r["ref"] for s in ("train", "dev", "test_cv", "test_fleurs")
                   for r in read_jsonl(f"data/{s}.jsonl")} - done)
    print(f"{len(done):,} sentences already labeled, {len(todo):,} to go")
    if todo:
        tok, model = load(a.model)
    for i in range(0, len(todo), a.chunk):
        part = todo[i:i + a.chunk]
        with open(a.labels, "a", encoding="utf-8") as f:
            for ur, roman in zip(part, romanize(tok, model, part)):
                f.write(json.dumps({"ur": ur, "roman": roman}, ensure_ascii=False) + "\n")
        print(f"{min(i + a.chunk, len(todo)):,}/{len(todo):,} labeled", flush=True)
    report(a.labels)


if __name__ == "__main__":
    main()

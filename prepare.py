"""Step 2: turn the raw downloads into 16 kHz mono WAV files and JSONL lists for training and testing.
Safe to re-run: WAVs that already exist are not decoded again; the lists are rebuilt identically.

    python prepare.py

Outputs (data/):
  train.jsonl        Common Voice (validated, minus dev/test speakers) + FLEURS train
  dev.jsonl          500 random clips from Common Voice dev + FLEURS validation (watched during training)
  test_cv.jsonl      Common Voice official test split (same kind of audio as training)
  test_fleurs.jsonl  FLEURS test split (different sentences, speakers and microphones)
  stats.json         hours and clip counts per split

Each line: {"audio": "/abs/path.wav", "text": "language Urdu<asr_text>...", "ref": "...", ...}
"""

import argparse
import csv
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import pyarrow.parquet as pq
import soundfile as sf

from common import LANG, LANG_ROMAN, clean, for_scoring, read_jsonl, roman_label_ok, sample, script_of, write_jsonl

SR = 16000
MIN_SEC, MAX_SEC = 0.5, 30.0
MAX_CHARS_PER_SEC = 20  # 99.5% of clips are under 12.2; faster means the audio is cut off or the text is wrong


def to_wav(job):
    """Decode any audio file to 16 kHz mono 16-bit WAV. Returns (dst, seconds) or (dst, None) if unreadable."""
    src, dst = job
    if os.path.exists(dst):
        return dst, sf.info(dst).duration
    try:
        wav, _ = librosa.load(src, sr=SR, mono=True)
    except Exception as e:  # a few clips in crowd-sourced data are broken; skip them, don't crash
        print(f"  unreadable, skipped: {src} ({e})")
        return dst, None
    sf.write(f"{dst}.tmp.wav", wav, SR, subtype="PCM_16")
    os.replace(f"{dst}.tmp.wav", dst)
    return dst, len(wav) / SR


def row(source, uid, wav, seconds, sentence, **extra):
    text = clean(sentence)
    return {"id": f"{source}/{uid}", "source": source, "audio": str(Path(wav).resolve()),
            "text": f"language {LANG}<asr_text>{text}", "ref": text, "seconds": round(seconds, 2), **extra}


def keep(r):
    """Drop clips whose label can't match the audio: wrong length, empty, romanized, or impossibly fast speech."""
    text = for_scoring(r["ref"])
    return (MIN_SEC <= r["seconds"] <= MAX_SEC and text and script_of(text) == "urdu"
            and len(text.replace(" ", "")) / r["seconds"] <= MAX_CHARS_PER_SEC)


def fleurs(raw, out):
    splits = {}
    for split in ("train", "validation", "test"):
        f = raw / "fleurs" / f"{split}.parquet"
        dest = out / "fleurs"
        dest.mkdir(parents=True, exist_ok=True)
        rows = []
        cols = ["audio", "raw_transcription", "gender"]
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=cols):
            for ex in batch.to_pylist():
                uid = Path(ex["audio"]["path"]).stem  # one recording; the sentence id repeats across speakers
                wav = dest / f"{uid}.wav"
                if not wav.exists():
                    audio, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
                    if audio.ndim > 1 or sr != SR:
                        audio = librosa.resample(librosa.to_mono(audio.T), orig_sr=sr, target_sr=SR)
                    sf.write(f"{wav}.tmp.wav", audio, SR, subtype="PCM_16")
                    os.replace(f"{wav}.tmp.wav", wav)
                rows.append(row("fleurs", uid, wav, sf.info(wav).duration, ex["raw_transcription"]))
        splits[split] = rows
        print(f"FLEURS {split}: {len(rows)} clips")
    return splits


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def common_voice(raw, out, workers):
    found = sorted((raw / "common_voice").glob("**/validated.tsv"))
    if not found:
        print("Common Voice not found under data/raw/common_voice: continuing with FLEURS only.")
        return None
    root = found[-1].parent
    print("Common Voice:", root)
    tables = {name: read_tsv(root / f"{name}.tsv") for name in ("validated", "dev", "test")}

    # Training uses all validated clips, except speakers and sentences that appear in dev or test,
    # so the test score measures new voices and new sentences.
    held = tables["dev"] + tables["test"]
    held_speakers = {r["client_id"] for r in held}
    held_sentences = {for_scoring(r["sentence"]) for r in held}
    tables["train"] = [r for r in tables["validated"]
                       if r["client_id"] not in held_speakers and for_scoring(r["sentence"]) not in held_sentences]
    del tables["validated"]

    dest = out / "common_voice"
    dest.mkdir(parents=True, exist_ok=True)
    clips = {r["path"]: r for t in tables.values() for r in t}
    jobs = [(str(root / "clips" / p), str(dest / f"{Path(p).stem}.wav")) for p in sorted(clips)]
    print(f"Common Voice: {len(jobs)} clips to convert (already converted ones are skipped) ...", flush=True)
    seconds = {}
    with ProcessPoolExecutor(workers) as pool:
        for i, (dst, sec) in enumerate(pool.map(to_wav, jobs, chunksize=64), 1):
            seconds[dst] = sec
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    splits = {}
    for name, table in tables.items():
        rows = []
        for r in table:
            wav = str(dest / f"{Path(r['path']).stem}.wav")
            if seconds.get(wav):
                rows.append(row("common_voice", Path(r["path"]).stem, wav, seconds[wav], r["sentence"],
                                speaker=r["client_id"]))
        splits[name] = rows
        print(f"Common Voice {name}: {len(rows)} clips")
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--dev-size", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)

    fl = fleurs(raw, out / "wav")
    cv = common_voice(raw, out / "wav", a.workers) or {"train": [], "dev": [], "test": []}

    tests = {"test_cv": cv["test"], "test_fleurs": fl["test"]}
    dev = cv["dev"] + fl["validation"]
    # No sentence from any dev/test set may appear in training.
    blocked = {for_scoring(r["ref"]) for rows in [dev, *tests.values()] for r in rows}
    train = [r for r in cv["train"] + fl["train"] if keep(r) and for_scoring(r["ref"]) not in blocked]
    splits = {"train": sample(train, 0), "dev": sample([r for r in dev if keep(r)], a.dev_size),
              **{k: [r for r in v if keep(r)] for k, v in tests.items()}}

    stats = {}
    for name, rows in splits.items():
        if rows:
            write_jsonl(out / f"{name}.jsonl", rows)
        stats[name] = {"clips": len(rows), "hours": round(sum(r["seconds"] for r in rows) / 3600, 2),
                       "by_source": {s: sum(r["source"] == s for r in rows) for s in ("common_voice", "fleurs")}}

    # Stage 2 (after romanize/label.py): the same clips again, with Roman Urdu targets -> data/<split>_roman.jsonl.
    # Training on both lists teaches one model both scripts; the language tag picks which one it writes.
    labels = out / "roman_labels.jsonl"
    if labels.exists():
        roman = {r["ur"]: r["roman"] for r in read_jsonl(labels) if roman_label_ok(r["ur"], r["roman"])}
        for name, rows in splits.items():
            twins = [{**r, "id": r["id"] + "#roman", "ref": roman[r["ref"]],
                      "text": f"language {LANG_ROMAN}<asr_text>{roman[r['ref']]}"} for r in rows if r["ref"] in roman]
            write_jsonl(out / f"{name}_roman.jsonl", twins)
            stats[name]["roman_clips"] = len(twins)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

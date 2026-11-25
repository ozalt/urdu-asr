"""Forgetting check: did learning Urdu cost the model its other languages? Safe to re-run.

    python other_langs.py                      # downloads FLEURS English + Hindi test sets (~1,000 clips)

Writes data/test_fleurs_en.jsonl and data/test_fleurs_hi.jsonl. Then score the base and fine-tuned models the
same way (README, "Forgetting check"):

    python evaluate.py --name base_en      --sets test_fleurs_en --language English
    python evaluate.py --name finetuned_en --sets test_fleurs_en --language English --model <checkpoint>

A full fine-tune updates every weight to fit Urdu, and nothing in the loss rewards keeping other languages, so some
loss is expected ("catastrophic forgetting"). This measures how much.
"""

import io
import os
import urllib.request
from pathlib import Path

import librosa
import pyarrow.parquet as pq
import soundfile as sf

from common import write_jsonl

URL = "https://huggingface.co/api/datasets/google/fleurs/parquet/{code}/test/0.parquet"
LANGS = {"en": "en_us", "hi": "hi_in"}
SR = 16000


def main():
    for short, code in LANGS.items():
        raw = Path(f"data/raw/fleurs_{code}/test.parquet")
        if not raw.exists():
            print(f"FLEURS {code} test: downloading ...", flush=True)
            raw.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL.format(code=code), f"{raw}.part")
            os.replace(f"{raw}.part", raw)
        dest = Path(f"data/wav/fleurs_{code}")
        dest.mkdir(parents=True, exist_ok=True)
        rows = []
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=64, columns=["audio", "raw_transcription"]):
            for ex in batch.to_pylist():
                uid = Path(ex["audio"]["path"]).stem
                wav = dest / f"{uid}.wav"
                if not wav.exists():
                    audio, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
                    if audio.ndim > 1 or sr != SR:
                        audio = librosa.resample(librosa.to_mono(audio.T), orig_sr=sr, target_sr=SR)
                    sf.write(f"{wav}.tmp.wav", audio, SR, subtype="PCM_16")
                    os.replace(f"{wav}.tmp.wav", wav)
                rows.append({"id": f"fleurs_{code}/{uid}", "source": f"fleurs_{code}", "audio": str(wav.resolve()),
                             "ref": ex["raw_transcription"], "seconds": round(sf.info(wav).duration, 2)})
        write_jsonl(f"data/test_fleurs_{short}.jsonl", rows)
        print(f"FLEURS {code} test: {len(rows)} clips, {sum(r['seconds'] for r in rows) / 3600:.2f} h "
              f"-> data/test_fleurs_{short}.jsonl")


if __name__ == "__main__":
    main()

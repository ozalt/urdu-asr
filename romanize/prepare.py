"""Stage 2, step 1: download and clean the Urdu -> Roman Urdu text pairs. Safe to re-run.

    romanize/.venv/bin/python romanize/prepare.py          (run from the urdu-asr folder)

- Roman-Urdu-Parl (Apache-2.0): 6.3M sentence pairs, partly machine-made and noisy -> cleaned sample for training.
- Dakshina, Urdu (Google, CC-BY-SA-4.0): 4,880 sentences romanized by native speakers -> the test set.
  Used only for testing: share-alike terms could otherwise apply to the trained model.

Outputs (romanize/data/): train.jsonl, dev.jsonl, test_dakshina.jsonl, each line {"ur": ..., "roman": ...}
"""

import argparse
import io
import os
import re
import sys
import tarfile
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import clean, for_scoring, sample, script_of, write_jsonl  # noqa: E402

RUP = "https://huggingface.co/datasets/Mavkif/Roman-Urdu-Parl-split/resolve/refs%2Fconvert%2Fparquet/default/{}"
RUP_FILES = ["train/0000.parquet", "train/0001.parquet", "train/0002.parquet", "validation/0000.parquet"]
DAKSHINA = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
DAK_DIR = "dakshina_dataset_v1.0/ur/romanized/"


def download(raw):
    for f in RUP_FILES:
        dest = raw / "rup" / f.replace("/", "_")
        if not dest.exists():
            print("Roman-Urdu-Parl", f, "downloading ...", flush=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(RUP.format(f), f"{dest}.part")
            os.replace(f"{dest}.part", dest)
    dak = raw / "dakshina" / DAK_DIR
    if not (dak / "ur.romanized.rejoined.test.roman.txt").exists():
        print("Dakshina: streaming the 2 GB archive, keeping only the Urdu romanized files ...", flush=True)
        with urllib.request.urlopen(DAKSHINA) as r, tarfile.open(fileobj=r, mode="r|") as tar:
            for m in tar:
                if m.name.startswith(DAK_DIR) and m.isfile():
                    (raw / "dakshina" / m.name).parent.mkdir(parents=True, exist_ok=True)
                    (raw / "dakshina" / m.name).write_bytes(tar.extractfile(m).read())


def read_rup(path):
    t = pq.read_table(path)
    return list(zip(t.column("Urdu text").to_pylist(), t.column("Roman-Urdu text").to_pylist()))


def normalize(ur, roman):
    return clean(ur or ""), " ".join((roman or "").split())


def plausible(ur, roman):
    """Same number of words, and each side in its own script. Equal word counts also let us line words up.
    No ALL-CAPS words: in Roman-Urdu-Parl they mark machine-inserted variants ("KA", "HAI", "si TV" for سی ٹی)."""
    return (ur and roman and script_of(ur) == "urdu" and script_of(roman) == "latin"
            and len(ur.split()) == len(roman.split()) and not re.search(r"\b[A-Z]{2,}\b", roman))


# Roman-Urdu-Parl was "augmented" by machine-substituting word variants, which broke millions of rows, often the
# majority for a word: یہ -> "ki" in 70% of its rows, ہاتھ (haath) -> "ke" in 85%. So frequency can't tell right
# from wrong; sound can. Each word is reduced to its consonants ("skeleton") on both sides and the two must match.
# Ignored: vowels, و and ی (vowel or consonant), and h (Roman Urdu drops and adds it freely).
_UR = str.maketrans({"ب": "b", "پ": "p", **dict.fromkeys("تٹط", "t"), **dict.fromkeys("ثسص", "s"), "ش": "S", "ج": "j",
                     "چ": "c", "د": "d", "ڈ": "d", **dict.fromkeys("ذزضظژ", "z"), "ر": "r", "ڑ": "r", "غ": "g",
                     "گ": "g", "ف": "f", "ق": "k", "ک": "k", "خ": "k", "ل": "l", "م": "m", "ن": "n", "ں": "n"})
_KEEP = set("bptsSjcdzrgfklmn0123456789")  # S = sh (ش), kept apart from s so خوشی/"khusi" fails


def _skeleton(word):
    return re.sub(r"(.)\1+", r"\1", "".join(c for c in word if c in _KEEP))  # "kitaab" = "kitab"


def skeleton_ur(word):
    return _skeleton(word.translate(_UR))


def skeleton_roman(word):
    w = word.lower()
    for digraph, sound in (("ch", "C"), ("sh", "S"), ("kh", "K"), ("gh", "G"), ("zh", "Z")):
        w = w.replace(digraph, sound)
    w = w.replace("q", "k").replace("x", "ks").replace("c", "k")
    return _skeleton(w.replace("C", "c").replace("K", "k").replace("G", "g").replace("Z", "z"))


# Short words have almost no consonants (ہی -> "", کوئی -> "k"), so two more checks: the first sound matches,
# and a و in the Urdu shows up as o/u/w/v. This catches ہی -> "yee" and کوئی -> "ki".
_FIRST = {**dict.fromkeys("ہحھ", "h"), "ی": "yie", **dict.fromkeys("اآعأ", "aeiou"), "و": "wvou"}


def word_ok(u, r, min_ratio=0.75):
    """0.75 rejects the corrupted forms that keep exactly 2 of 3 consonant matches (دنیا "ne", سکتے "satke")."""
    su, sr = skeleton_ur(u), skeleton_roman(r)
    if SequenceMatcher(None, su, sr).ratio() < min_ratio:
        return False
    if u[0] in _FIRST and r[0] not in _FIRST[u[0]]:
        return False
    if "kkh" in r or "fh" in r:  # machine artifacts, never real Roman Urdu: خدا "kkhuda", محفل "mehfhil"
        return False
    if u[0].translate(_UR) in _KEEP and su and not sr.startswith(su[0]):  # starts with a consonant: same one
        return False  # جاتے "atay"
    return "و" not in u[1:] or any(c in r for c in "ouwv")


def sounds_right(ur, roman):
    """Every word pair sounds alike: یہ/"yeh" passes; یہ/"ki", رانا/"pakistani", ہی/"yee" fail."""
    # ملک is mulk (country) or the name Malik; lowercase "malik" is Roman-Urdu-Parl's substitution (84% of its rows).
    # Checked on the original text, since for_scoring lowercases.
    if "ملک" in ur.split() and re.search(r"\bmalik\b", roman):
        return False
    return all(word_ok(u, r) for u, r in zip(for_scoring(ur).split(), for_scoring(roman).split()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="romanize/data")
    ap.add_argument("--train-size", type=int, default=200_000)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    download(raw)

    dak = raw / "dakshina" / DAK_DIR
    # A few Dakshina "sentences" are whole paragraphs (up to ~2,000 words). Our speech transcripts are at most
    # 74 words, so test on sentences up to 80 words (drops 36 of 4,880).
    test = [{"ur": clean(u), "roman": " ".join(r.split())} for u, r in zip(
        (dak / "ur.romanized.rejoined.test.native.txt").read_text().splitlines(),
        (dak / "ur.romanized.rejoined.test.roman.txt").read_text().splitlines()) if len(u.split()) <= 80]
    # Dakshina is Wikipedia text and so is part of Roman-Urdu-Parl: block its sentences from training.
    blocked = {for_scoring(u) for f in ("dev", "test")
               for u in (dak / f"ur.romanized.rejoined.{f}.native.txt").read_text().splitlines()}

    def cleaned(rows, n):
        pairs = {normalize(u, r) for u, r in rows}  # also removes exact duplicates
        pairs = sample(sorted(p for p in pairs if plausible(*p) and for_scoring(p[0]) not in blocked), 0)
        kept, checked = [], 0
        for p in pairs:  # random order; stop once there are enough
            checked += 1
            if sounds_right(*p):
                kept.append({"ur": p[0], "roman": p[1]})
                if len(kept) == n:
                    break
        print(f"  {len(rows):,} rows -> {len(pairs):,} unique plausible pairs; {len(kept) / checked:.0%} of the "
              f"{checked:,} checked sound right -> kept {len(kept):,}")
        return kept

    print("Roman-Urdu-Parl train:")
    train = cleaned([p for f in RUP_FILES[:3] for p in read_rup(raw / "rup" / f.replace("/", "_"))], a.train_size)
    print("Roman-Urdu-Parl validation:")
    dev = cleaned(read_rup(raw / "rup" / RUP_FILES[3].replace("/", "_")), 1000)

    for name, rows in (("train", train), ("dev", dev), ("test_dakshina", test)):
        write_jsonl(out / f"{name}.jsonl", rows)
        print(f"{name}: {len(rows):,} pairs  e.g. {rows[0]['ur'][:40]} -> {rows[0]['roman'][:40]}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        good = [("یہ", "yeh"), ("ہاتھ", "haath"), ("کہوں", "kahoon"), ("نہیں", "nahin"), ("سپورٹ", "support"),
                ("چھوڑ", "chhor"), ("بہت", "bohat"), ("خیال", "khayal"), ("ضرور", "zaroor"), ("کتنے", "kitney"),
                ("ہی", "hi"), ("ہے", "hai"), ("کوئی", "koi"), ("اور", "aur"), ("وہ", "woh"), ("عمر", "umar"),
                ("خواب", "khwab"), ("آپ", "aap"), ("ایک", "aik"), ("تو", "to"), ("یہاں", "yahan"),
                ("دنیا", "duniya"), ("سکتے", "sakte"), ("جاتے", "jaate"), ("ملک", "mulk"), ("ملک", "Malik"),
                ("خوشی", "khushi"), ("خدا", "khuda"), ("محفل", "mehfil"), ("شکریہ", "shukriya"), ("بس", "bas")]
        bad = [("یہ", "ki"), ("ہاتھ", "ke"), ("کہوں", "kaisay"), ("آتے", "se"), ("رانا", "pakistani"),
               ("سپورٹ", "ki"), ("رحمانی", "airej"), ("ہی", "yee"), ("ہے", "ye"), ("کوئی", "ki"),
               ("دنیا", "ne"), ("سکتے", "satke"), ("جاتے", "atay"), ("خوشی", "khusi"), ("خدا", "kkhuda"),
               ("محفل", "mehfhil")]
        assert all(sounds_right(u, r) for u, r in good) and not any(sounds_right(u, r) for u, r in bad)
        print("prepare.py self-check OK")
    else:
        main()

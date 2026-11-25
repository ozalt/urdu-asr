"""Shared helpers: Urdu text cleanup (for training targets and for scoring) and JSONL I/O."""

import json
import random
import re
import unicodedata
from pathlib import Path

LANG = "Urdu"  # the language tag the model learns: "language Urdu<asr_text>..."
LANG_ROMAN = "Roman urdu"  # second output: Urdu in Latin letters. Spelled this way because qwen_asr rewrites
                           # language names to "First letter capital, rest lowercase" before using them.

# Arabic code points that look like Urdu letters; Urdu text online mixes both.
_LETTERS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ", "ۀ": "ۂ"})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIACRITICS = re.compile("[ً-ٰٟ۪-ۭ]")  # zabar/zer/pesh etc.: rarely written, so not scored
_INVISIBLE = re.compile("[​-‏‪-‮⁦-⁩﻿]")


def clean(text):
    """Light cleanup for training targets: consistent letters, no invisible marks, single spaces.
    Punctuation is kept so the model learns to write readable sentences."""
    text = unicodedata.normalize("NFKC", text).translate(_LETTERS)
    text = _INVISIBLE.sub("", text)
    return " ".join(text.split())


def for_scoring(text):
    """Normalization applied to both reference and prediction before WER/CER (standard practice):
    no punctuation, no diacritics, lowercase (for English words), Western digits."""
    text = _DIACRITICS.sub("", clean(text)).translate(_DIGITS).lower()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def script_of(text):
    """Which writing system most letters are in: 'urdu' (Arabic script), 'devanagari', 'latin' or 'other'."""
    counts = {"urdu": 0, "devanagari": 0, "latin": 0, "other": 0}
    for c in text:
        if not c.isalpha():
            continue
        cp = ord(c)
        counts["urdu" if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F else
               "devanagari" if 0x0900 <= cp <= 0x097F else "latin" if c.isascii() else "other"] += 1
    return max(counts, key=counts.get) if any(counts.values()) else "none"


# Checks for machine-written Roman Urdu labels (romanize/label.py): each flags a label that can't be a faithful
# romanization of its sentence. Used by label.py to report and by prepare.py to skip.
ROMAN_CHECKS = {
    "still has Urdu letters": lambda u, r: script_of(r) != "latin",
    "length far off (Roman chars / Urdu chars outside 0.7-2.5)":
        lambda u, r: not 0.7 <= len(r.replace(" ", "")) / max(1, len(u.replace(" ", ""))) <= 2.5,
    "word count far off": lambda u, r: abs(len(r.split()) - len(u.split())) > max(2, 0.3 * len(u.split())),
    "repeats itself": lambda u, r: re.search(r"(.{8,}?)\s*\1\s*\1", r) is not None,
}


def roman_label_ok(ur, roman):
    return not any(bad(ur, roman) for bad in ROMAN_CHECKS.values())


def sample(rows, n, seed=0):
    """A fixed random sample of n rows (all rows if n is 0)."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n or None]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":  # self-check
    assert clean("  كيا  ہے‌ ") == "کیا ہے"
    assert for_scoring("آپ کیسے ہیں؟ Meeting ۲۳ کو!") == "آپ کیسے ہیں meeting 23 کو"
    assert for_scoring("بَچّہ") == for_scoring("بچہ")
    assert script_of("آپ کیسے ہیں") == "urdu" and script_of("आप कैसे हैं") == "devanagari"
    assert script_of("aap kaise hain") == "latin" and script_of("123") == "none"
    assert roman_label_ok("میں ٹھیک ہوں", "mein theek hoon")
    assert not roman_label_ok("دنیا", "ne") and not roman_label_ok("آپ", "آپ")
    print("common.py self-check OK")

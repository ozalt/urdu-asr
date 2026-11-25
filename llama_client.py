"""Transcribe audio with the GGUF model through a running llama-server. Standard library only (runs on any laptop).

    llama-server -m urdu-asr-0.6b-q8_0.gguf --mmproj mmproj-urdu-asr-0.6b-f16.gguf --port 8080
    python llama_client.py clip.wav                     # Urdu script
    python llama_client.py clip.wav --language "Roman urdu"
    python llama_client.py clip.wav --language auto     # the model picks (and names) the language

Same prompt as qwen_asr: forcing a language appends "language X<asr_text>" after the assistant turn starts.
The model was trained on clips of 30 s or less, so longer WAV audio (voice notes, interviews) is cut into pieces at
quiet moments, transcribed piece by piece and joined.
"""

import argparse
import base64
import io
import json
import urllib.request
import wave
from array import array
from collections import namedtuple
from functools import cache

PROMPT = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<__media__><|im_end|>\n<|im_start|>assistant\n"
Result = namedtuple("Result", "text language")
MAX_SECONDS, MIN_SECONDS = 28, 15  # long audio: cut at the quietest 0.1 s between 15 and 28 s into each piece


@cache
def media_marker(url):
    """llama-server picks a random audio placeholder per run and reports it in /props."""
    with urllib.request.urlopen(f"{url}/props", timeout=10) as r:
        return json.loads(r.read())["media_marker"]


def parse(raw, language):
    """'language Urdu<asr_text>...' -> Result('...', 'Urdu'). With a forced language the tag is in the prompt."""
    if language:
        return Result(raw.strip(), language)
    head, sep, text = raw.partition("<asr_text>")
    if not sep:
        return Result(raw.strip(), None)
    lang = head.replace("language", "", 1).strip()
    return Result("" if lang == "None" else text.strip(), lang or None)


def cut_points(samples, rate, max_s=MAX_SECONDS, min_s=MIN_SECONDS):
    """(start, end) sample ranges: the whole clip if it fits in 30 s, else pieces of at most max_s seconds, each cut
    at the quietest 0.1 s window between min_s and max_s, so cuts land in pauses rather than mid-word."""
    if len(samples) <= 30 * rate:
        return [(0, len(samples))]
    hop, start, cuts = rate // 10, 0, []
    while len(samples) - start > max_s * rate:
        windows = range(start + min_s * rate, start + max_s * rate - hop, hop)
        quietest = min(windows, key=lambda p: sum(x * x for x in samples[p:p + hop:4]))
        cuts.append((start, quietest + hop // 2))
        start = quietest + hop // 2
    return cuts + [(start, len(samples))]


def pieces(wav):
    """WAV path or bytes -> list of WAV bytes, each short enough for the model. Other formats pass through whole."""
    data = wav if isinstance(wav, bytes) else open(wav, "rb").read()
    if not data.startswith(b"RIFF"):
        return [data]
    with wave.open(io.BytesIO(data)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        return [data]
    samples = array("h", frames)[::channels]  # first channel is enough for speech
    out = []
    for a, b in cut_points(samples, rate):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
            w.writeframes(samples[a:b].tobytes())
        out.append(buf.getvalue())
    return out


def transcribe_piece(audio, language, url, timeout=600):
    """One request: audio bytes (30 s or less) -> Result."""
    prompt = PROMPT.replace("<__media__>", media_marker(url)) + (f"language {language}<asr_text>" if language else "")
    body = {"prompt": {"prompt_string": prompt, "multimodal_data": [base64.b64encode(audio).decode()]},
            "n_predict": 512, "temperature": 0, "cache_prompt": False}
    req = urllib.request.Request(f"{url}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return parse(json.loads(r.read())["content"], language)


def transcribe(wav, language="Urdu", url="http://127.0.0.1:8080", timeout=600):
    """WAV path or bytes, any length -> Result. language: "Urdu", "Roman urdu", or None/"auto"."""
    language = None if language in (None, "", "auto") else language
    results = [transcribe_piece(p, language, url, timeout) for p in pieces(wav)]
    return Result(" ".join(r.text for r in results if r.text), results[0].language)


def server_model(url="http://127.0.0.1:8080"):
    """The model file the server has loaded (recorded in the results)."""
    with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as r:
        return json.loads(r.read())["data"][0]["id"]


if __name__ == "__main__":
    assert parse("language Urdu<asr_text>سلام", None) == ("سلام", "Urdu")
    assert parse("language None<asr_text>", None) == ("", "None")
    assert parse(" salam ", "Roman urdu") == ("salam", "Roman urdu")
    tone = [3000, -3000] * 8000 * 20  # 20 s of sound
    assert cut_points(array("h", tone), 16000) == [(0, len(tone))]  # 20 s: one piece
    long_ = array("h", tone + [0] * 1600 + tone)  # 20 s, 0.1 s pause, 20 s: one cut, inside the pause
    (a, b), (c, d) = cut_points(long_, 16000)
    assert a == 0 and 20 * 16000 <= b <= 20.1 * 16000 and c == b and d == len(long_)
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--language", default="Urdu", help='"Urdu", "Roman urdu", or "auto"')
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    a = ap.parse_args()
    for w in a.wav:
        print(transcribe(w, a.language, a.url).text)

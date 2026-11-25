"""Transcribe audio with the GGUF model through a running llama-server. Standard library only (runs on any laptop).

    llama-server -m urdu-asr-0.6b-q8_0.gguf --mmproj mmproj-urdu-asr-0.6b-f16.gguf --port 8080
    python llama_client.py clip.wav                     # Urdu script
    python llama_client.py clip.wav --language "Roman urdu"
    python llama_client.py clip.wav --language auto     # the model picks (and names) the language

Same prompt as qwen_asr: forcing a language appends "language X<asr_text>" after the assistant turn starts.
"""

import argparse
import base64
import json
import urllib.request
from collections import namedtuple
from functools import cache

PROMPT = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<__media__><|im_end|>\n<|im_start|>assistant\n"
Result = namedtuple("Result", "text language")


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


def transcribe(wav, language="Urdu", url="http://127.0.0.1:8080", timeout=600):
    language = None if language in (None, "", "auto") else language
    with open(wav, "rb") as f:
        audio = base64.b64encode(f.read()).decode()
    prompt = PROMPT.replace("<__media__>", media_marker(url)) + (f"language {language}<asr_text>" if language else "")
    body = {"prompt": {"prompt_string": prompt,
                       "multimodal_data": [audio]},
            "n_predict": 256, "temperature": 0, "cache_prompt": False}
    req = urllib.request.Request(f"{url}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return parse(json.loads(r.read())["content"], language)


def server_model(url="http://127.0.0.1:8080"):
    """The model file the server has loaded (recorded in the results)."""
    with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as r:
        return json.loads(r.read())["data"][0]["id"]


if __name__ == "__main__":
    assert parse("language Urdu<asr_text>سلام", None) == ("سلام", "Urdu")
    assert parse("language None<asr_text>", None) == ("", "None")
    assert parse(" salam ", "Roman urdu") == ("salam", "Roman urdu")
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--language", default="Urdu", help='"Urdu", "Roman urdu", or "auto"')
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    a = ap.parse_args()
    for w in a.wav:
        print(transcribe(w, a.language, a.url).text)

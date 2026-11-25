"""Laptop kit: run the quantized Urdu speech model on this computer's CPU, measure it, and pack the evidence.
Linux x86-64, Python 3.9+, standard library only. See LAPTOP.md.

    python3 laptop.py demo                  # http://127.0.0.1:8000 : record or upload Urdu speech (any length)
    python3 laptop.py transcribe notes/     # every recording in a folder -> transcripts/<name>.txt (Urdu + Roman)
    python3 laptop.py bench                 # ~10 min: base vs fine-tuned on 20 unseen clips -> laptop-results.zip

First run downloads llama.cpp (prebuilt, CPU) and any model file missing from models/.
"""

import argparse
import io
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
import wave
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import llama_client
from common import for_scoring

HERE = Path(__file__).resolve().parent
MODELS, RESULTS, LLAMA = HERE / "models", HERE / "results", HERE / "llama.cpp"
LLAMA_TAG = "b12675"  # llama.cpp release the model was converted and tested with
LLAMA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-ubuntu-x64.tar.gz"
OURS = "https://huggingface.co/zaltahir/qwen3-asr-0.6b-urdu/resolve/main/"
BASE = "https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF/resolve/main/"
# name: (decoder file, audio encoder file, download base URL)
MODEL_FILES = {
    "base": ("Qwen3-ASR-0.6B-Q8_0.gguf", "mmproj-Qwen3-ASR-0.6B-BF16.gguf", BASE),  # 16-bit encoder, like ours
    "q8_0": ("urdu-asr-0.6b-q8_0.gguf", "mmproj-urdu-asr-0.6b-f16.gguf", OURS),
    "q4_k_m": ("urdu-asr-0.6b-q4_k_m.gguf", "mmproj-urdu-asr-0.6b-f16.gguf", OURS),
}
PORTS = {"base": 8081, "q8_0": 8082, "q4_k_m": 8083}


def download(url, dest):
    if dest.exists():
        return
    print(f"downloading {dest.name} ...", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(f"{dest}.part", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(f"{dest}.part", dest)


def llama_server():
    found = list(LLAMA.rglob("llama-server"))
    if not found:
        archive = HERE / f"llama-{LLAMA_TAG}.tar.gz"
        download(LLAMA_URL, archive)
        with tarfile.open(archive) as t:
            t.extractall(LLAMA)
        archive.unlink()
        found = list(LLAMA.rglob("llama-server"))
    return found[0]


def physical_cores():
    try:
        pairs = set()
        for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
            f = dict(l.split(":", 1) for l in block.splitlines() if ":" in l)
            f = {k.strip(): v.strip() for k, v in f.items()}
            if "core id" in f:
                pairs.add((f.get("physical id"), f["core id"]))
        return len(pairs) or os.cpu_count()
    except OSError:
        return os.cpu_count()


def hardware():
    info = {"os": platform.platform(), "cpu_threads": os.cpu_count(), "cpu_cores": physical_cores(),
            "llama_cpp": f"{LLAMA_TAG} (prebuilt CPU build, no GPU)"}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            info["cpu"] = line.split(":", 1)[1].strip()
            break
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal"):
            info["ram_gb"] = round(int(line.split()[1]) / 1024 ** 2, 1)
    osr = Path("/etc/os-release")
    if osr.exists():
        info["distro"] = dict(l.split("=", 1) for l in osr.read_text().splitlines() if "=" in l).get(
            "PRETTY_NAME", "").strip('"')
    return info


class Server:
    """One llama-server per model, CPU only."""

    def __init__(self, name, threads):
        model, mmproj, url = MODEL_FILES[name]
        for f in (model, mmproj):
            download(url + f, MODELS / f)
        self.name, self.url = name, f"http://127.0.0.1:{PORTS[name]}"
        self.files = [MODELS / model, MODELS / mmproj]
        RESULTS.mkdir(exist_ok=True)
        self.log = open(RESULTS / f"server_{name}.log", "w")
        t0 = time.time()
        self.proc = subprocess.Popen([str(llama_server()), "-m", self.files[0], "--mmproj", self.files[1],
                                      "--port", str(PORTS[name]), "-t", str(threads), "-np", "1", "-c", "4096",
                                      "-ngl", "0"], stdout=self.log, stderr=subprocess.STDOUT)
        while True:
            if self.proc.poll() is not None:
                sys.exit(f"llama-server for {name} stopped; see {self.log.name}")
            try:
                urllib.request.urlopen(f"{self.url}/health", timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        self.load_seconds = round(time.time() - t0, 1)

    def peak_ram_mb(self):  # Linux: high-water mark of resident memory
        for line in Path(f"/proc/{self.proc.pid}/status").read_text().splitlines():
            if line.startswith("VmHWM"):
                return round(int(line.split()[1]) / 1024)

    def transcribe(self, wav, language):
        t0 = time.time()
        r = llama_client.transcribe(wav if isinstance(wav, bytes) else str(wav), language, self.url)
        return r.text, r.language, time.time() - t0

    def stop(self):
        self.proc.terminate()
        self.proc.wait()
        self.log.close()


def edits(ref, hyp):
    """Levenshtein distance between two sequences (words or characters)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def error_rates(pairs):
    """Corpus WER and CER over (reference, prediction) pairs, after the same normalization as the paper numbers."""
    pairs = [(for_scoring(r), for_scoring(h)) for r, h in pairs]
    wer = sum(edits(r.split(), h.split()) for r, h in pairs) / sum(len(r.split()) for r, _ in pairs)
    cer = sum(edits(r, h) for r, h in pairs) / sum(len(r) for r, _ in pairs)
    return round(wer, 4), round(cer, 4)


def samples():
    rows = [json.loads(line) for line in open(HERE / "samples/samples.jsonl", encoding="utf-8")]
    for r in rows:
        r["path"] = HERE / "samples" / r["file"]
    return rows


# (model, language forced or None, reference field, label)
RUNS = [("base", None, "ref", "Base, auto language"), ("base", "Urdu", "ref", "Base, told Urdu"),
        ("q8_0", "Urdu", "ref", "Fine-tuned Q8_0, Urdu"), ("q8_0", "Roman urdu", "ref_roman", "Fine-tuned Q8_0, Roman"),
        ("q4_k_m", "Urdu", "ref", "Fine-tuned Q4_K_M, Urdu"),
        ("q4_k_m", "Roman urdu", "ref_roman", "Fine-tuned Q4_K_M, Roman")]


def bench(threads):
    clips, hw = samples(), hardware()
    audio_s = sum(c["seconds"] for c in clips)
    print(f"{hw.get('cpu')}, {hw['cpu_cores']} cores, {hw.get('ram_gb')} GB RAM; {len(clips)} clips, "
          f"{audio_s:.0f} s of audio; {threads} threads", flush=True)
    report, outputs = {"hardware": hw, "threads": threads, "clips": len(clips), "audio_seconds": audio_s,
                       "models": {}, "runs": []}, []
    for name in MODEL_FILES:
        server = Server(name, threads)
        server.transcribe(clips[0]["path"], "Urdu")  # warm-up, not timed
        for model, lang, ref_key, label in (r for r in RUNS if r[0] == name):
            preds, busy = [], 0.0
            for i, c in enumerate(clips, 1):
                text, detected, sec = server.transcribe(c["path"], lang)
                busy += sec
                preds.append(text)
                outputs.append({"run": label, "file": c["file"], "seconds_audio": c["seconds"],
                                "seconds_taken": round(sec, 2), "detected": detected, "pred": text,
                                "ref": c[ref_key]})
                print(f"\r  {label}: {i}/{len(clips)}", end="", flush=True)
            wer, cer = error_rates([(c[ref_key], p) for c, p in zip(clips, preds)])
            run = {"run": label, "wer": wer, "cer": cer, "seconds_taken": round(busy, 1),
                   "real_time_factor": round(busy / audio_s, 3)}
            report["runs"].append(run)
            print(f"\r  {label}: WER {wer:.1%}  CER {cer:.1%}  {busy:.0f} s for {audio_s:.0f} s of audio "
                  f"(real-time factor {busy / audio_s:.2f})", flush=True)
        report["models"][name] = {"files": {f.name: round(f.stat().st_size / 1e6) for f in server.files},
                                  "load_seconds": server.load_seconds, "peak_ram_mb": server.peak_ram_mb()}
        server.stop()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "bench.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    with open(RESULTS / "outputs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    lines = [f"Hardware: {hw.get('cpu')} ({hw['cpu_cores']} cores, {hw['cpu_threads']} threads), "
             f"{hw.get('ram_gb')} GB RAM, {hw.get('distro', hw['os'])}. CPU only, {threads} threads, "
             f"llama.cpp {LLAMA_TAG}.", f"{len(clips)} FLEURS test clips never seen in training, {audio_s:.0f} s "
             "of audio. Real-time factor = processing time / audio length (below 1 = faster than real time).", "",
             "| Run | WER | CER | Real-time factor |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['run']} | {r['wer']:.1%} | {r['cer']:.1%} | {r['real_time_factor']:.2f} |"
              for r in report["runs"]]
    lines += ["", "| Model | Files (MB) | Peak RAM (MB) | Load (s) |", "| --- | --- | --- | --- |"]
    lines += [f"| {n} | {sum(m['files'].values())} | {m['peak_ram_mb']} | {m['load_seconds']} |"
              for n, m in report["models"].items()]
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    zipped = HERE / "laptop-results.zip"
    with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as z:
        for p in RESULTS.iterdir():
            z.write(p, f"results/{p.name}")
    print(f"\nPacked {zipped}")


DEMO_RUNS = [("base", None, "Base Qwen3-ASR, auto language"), ("base", "Urdu", "Base Qwen3-ASR, told Urdu"),
             ("q8_0", "Urdu", "Fine-tuned: Urdu"), ("q8_0", "Roman urdu", "Fine-tuned: Roman Urdu")]


def demo(threads, port=8000):
    clips = samples()
    servers = {}

    class Handler(BaseHTTPRequestHandler):
        def send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send((HERE / "demo.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/samples":
                self.send(json.dumps([{k: c[k] for k in ("file", "seconds", "ref", "ref_roman")} for c in clips],
                                     ensure_ascii=False).encode(), "application/json")
            elif self.path.startswith("/samples/") and (HERE / "samples" / Path(self.path).name).is_file():
                self.send((HERE / "samples" / Path(self.path).name).read_bytes(), "audio/wav")
            else:
                self.send(b"not found", "text/plain", 404)

        def do_POST(self):  # body: a 16 kHz mono WAV made by the page
            audio = self.rfile.read(int(self.headers["Content-Length"]))
            with wave.open(io.BytesIO(audio)) as w:
                seconds = round(w.getnframes() / w.getframerate(), 2)
            if self.path == "/compare":  # all four outputs, base model first
                out = []
                for name, lang, label in DEMO_RUNS:
                    text, detected, sec = servers[name].transcribe(audio, lang)
                    out.append({"label": label, "text": text, "detected": detected, "seconds": round(sec, 2)})
                return self.send(json.dumps({"audio_seconds": seconds, "results": out}, ensure_ascii=False).encode(),
                                 "application/json")
            # /transcribe: our model, both scripts, one JSON line per piece so long recordings show progress
            parts = llama_client.pieces(audio)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for i, piece in enumerate(parts, 1):
                t0 = time.time()
                urdu, roman = (servers["q8_0"].transcribe(piece, lang)[0] for lang in ("Urdu", "Roman urdu"))
                self.wfile.write((json.dumps({"piece": i, "pieces": len(parts), "audio_seconds": seconds,
                                              "urdu": urdu, "roman": roman, "seconds": round(time.time() - t0, 2)},
                                             ensure_ascii=False) + "\n").encode())
                self.wfile.flush()

        def log_message(self, *args):
            pass

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # closing the terminal also stops the model servers
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)  # claim the port before starting the models
    except OSError:
        sys.exit(f"Port {port} is in use by another program. Try: python3 laptop.py demo --port 8010")
    try:
        for n in ("base", "q8_0"):  # one at a time, so a failed start still stops the one already running
            servers[n] = Server(n, threads)
        print(f"Demo running at http://127.0.0.1:{port}  (Ctrl+C to stop)", flush=True)
        webbrowser.open(f"http://127.0.0.1:{port}")
        httpd.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        for s in servers.values():
            s.stop()


AUDIO = {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".flac", ".webm", ".mp4", ".amr", ".3gp"}


def to_wav(path):
    """Any audio file -> 16 kHz mono WAV bytes (ffmpeg); plain WAV files work without it."""
    if shutil.which("ffmpeg"):  # raw samples, then a proper header (a piped WAV from ffmpeg has no length in it)
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                             capture_output=True, check=True).stdout
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
            w.writeframes(pcm)
        return buf.getvalue()
    if path.suffix.lower() == ".wav":
        return path.read_bytes()
    sys.exit(f"{path.name}: converting {path.suffix} files needs ffmpeg. Install it with: sudo apt install ffmpeg")


def transcribe_files(paths, threads):
    """Every audio file under the given files/folders -> transcripts/<name>.txt with Urdu, then Roman Urdu."""
    files = sorted({f for p in map(Path, paths) for f in ([p] if p.is_file() else p.rglob("*"))
                    if f.suffix.lower() in AUDIO})
    if not files:
        sys.exit("No audio files found (" + ", ".join(sorted(AUDIO)) + ")")
    out = HERE / "transcripts"
    out.mkdir(exist_ok=True)
    server = Server("q8_0", threads)
    try:
        for f in files:
            audio = to_wav(f)
            with wave.open(io.BytesIO(audio)) as w:
                seconds = w.getnframes() / w.getframerate()
            t0 = time.time()
            urdu, roman = (server.transcribe(audio, lang)[0] for lang in ("Urdu", "Roman urdu"))
            dest = out / f"{f.stem}.txt"
            dest.write_text(f"{urdu}\n\n{roman}\n", encoding="utf-8")
            print(f"{f.name}: {seconds:.0f} s of audio in {time.time() - t0:.0f} s -> {dest.relative_to(HERE)}",
                  flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    assert edits("abc", "abd") == 1 and edits([], ["x"]) == 1
    assert error_rates([("ایک دو تین", "ایک دو")]) == (0.3333, 0.4)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["demo", "transcribe", "bench"])
    ap.add_argument("paths", nargs="*", help="transcribe: audio files or folders")
    ap.add_argument("--threads", type=int, default=physical_cores(), help="CPU threads (default: physical cores)")
    ap.add_argument("--port", type=int, default=8000, help="demo: web page port")
    a = ap.parse_args()
    if a.what == "transcribe":
        transcribe_files(a.paths or ["."], a.threads)
    else:
        bench(a.threads) if a.what == "bench" else demo(a.threads, a.port)

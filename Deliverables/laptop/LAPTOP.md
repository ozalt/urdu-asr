# Laptop kit: run, measure, record

Everything here runs on the laptop's CPU, offline after the first run. Tested target: Kubuntu, Intel i5-1145G7,
12 GB RAM, Python 3. Nothing to `pip install`.

You bring back two things: **`laptop-results.zip`** (made by step 2) and **the demo video** (step 4).

## 1. Unpack (2 min)

Copy `urdu-asr-laptop-kit.zip` to the laptop, then:

```bash
cd ~ && unzip urdu-asr-laptop-kit.zip && cd urdu-asr-laptop-kit
python3 --version            # 3.9 or newer
```

`models/` already holds our model. The first run also downloads, once:
- llama.cpp (prebuilt for Ubuntu, 23 MB), into `llama.cpp/`
- the original Qwen3-ASR-0.6B in the same format (1 GB, from `ggml-org`), for the before/after comparison

So be online for the first run. After that, no internet is needed.

## 2. Benchmark (about 10 to 15 min)

Plug in the charger (on battery the CPU slows down), close the browser and other heavy apps, then:

```bash
python3 laptop.py bench
```

It runs 20 FLEURS test clips the model never saw through:
1. the original model, left to pick the language (it writes Hindi script);
2. the original model, told the language is Urdu;
3. our model in two sizes, Q8_0 and Q4_K_M, writing Urdu script and then Roman Urdu.

For each it prints the word error rate, character error rate and **real-time factor**: seconds of work per second of
audio, so 0.25 means a 10-second clip is transcribed in 2.5 seconds. At the end it shows two tables, then packs
`results/` into **`laptop-results.zip`**. Bring that zip back; it also records the CPU, RAM and OS automatically.

If it is very slow, try `python3 laptop.py bench --threads 8`. The default is one thread per physical core (4 on an
i5-1145G7), usually the fastest.

## 3. Try the demo (5 min)

```bash
python3 laptop.py demo
```

The browser opens `http://127.0.0.1:8000`. Allow the microphone. Then try all three inputs:
- **Record**: speak a sentence of Urdu, press Stop.
- **Upload audio**: any audio file, such as a WhatsApp voice note.
- **Unseen test clip**: a FLEURS clip, with the human transcript shown above the results.

Each input gets four answers: the original model (auto, then told Urdu) and our model (Urdu script, Roman Urdu), each
with the time it took. Stop with Ctrl+C in the terminal.

## 4. Record the video (2 to 3 minutes long)

Install a screen recorder once: `sudo apt install obs-studio` (or use Spectacle, Kubuntu's screenshot tool, if your
version has "Record screen"). Record the full screen at 1080p, with the microphone on.

Suggested script:

| Time | Show | Say |
| --- | --- | --- |
| 0:00 to 0:20 | Title page or the demo page | Problem: most Pakistanis type Urdu in Roman letters; voice-to-text for Urdu is poor and usually goes to the cloud |
| 0:20 to 0:40 | Terminal: `lscpu \| head -15`, `free -h`, then **turn Wi-Fi off** | "This is a normal laptop: i5, 12 GB, no GPU, and now no internet." |
| 0:40 to 0:50 | `python3 laptop.py demo` | One command starts everything |
| 0:50 to 1:30 | Pick an unseen test clip | Original model writes Hindi script; told Urdu it gets about half the words wrong; ours matches the human transcript, in both scripts. Point at the seconds taken |
| 1:30 to 2:10 | Record yourself speaking Urdu, then upload a voice note | It works on your own voice, in real time on a CPU |
| 2:10 to 2:40 | `results/summary.md`, plus the results table from the write-up | Numbers: word error rate 51% to 25% (Urdu), on test sets never seen in training; model size; RAM |
| 2:40 to 3:00 | Hugging Face page | Where to get it; what still fails (English loanwords, other languages degrade) |

Keep it under 3 minutes. Upload it to YouTube as **Unlisted** (public link, not listed in search) and send me the link.

## Troubleshooting

- **`llama-server for ... stopped`**: open `results/server_<name>.log`; the last lines say why (usually out of memory:
  close apps and re-run).
- **Microphone gives nothing**: check the input device in the system settings (Plasma: Audio volume, Input devices);
  Firefox and Chrome both allow the microphone on `127.0.0.1`.
- **Port already in use**: something else uses port 8000 or 8081 to 8083. Reboot, or stop the other program.
- **Download fails**: check the internet once; files that finished downloading are kept, so re-running continues.

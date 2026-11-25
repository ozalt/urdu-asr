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

## 3. Use it (5 min)

```bash
python3 laptop.py demo
```

The browser opens `http://127.0.0.1:8000` (if that port is busy: `python3 laptop.py demo --port 8010`). Allow the
microphone. The page has two tabs:

**Transcribe** (opens first): the everyday tool.
- **Record**: speak Urdu, press Stop.
- **Upload audio or voice note**: any length. For a WhatsApp voice note, save it from the phone (or WhatsApp Web:
  download) and upload the `.opus`/`.ogg` file.
- You get Urdu script and Roman Urdu side by side, with **Copy** and **Save .txt** buttons. Recordings longer than
  30 seconds are cut at pauses and the text appears part by part, with a progress bar.

**Compare with the original model**: the same inputs, but four answers: the original Qwen3-ASR (left to pick the
language, then told Urdu) next to ours (Urdu, Roman Urdu), each with the time it took. Pick an **Unseen test clip** to
also see the human transcript.

Stop with Ctrl+C in the terminal.

**A whole folder at once** (interviews, lectures, a voice-note archive):

```bash
sudo apt install ffmpeg                  # once; needed for .opus, .ogg, .m4a, .mp3 (plain .wav works without it)
python3 laptop.py transcribe ~/voice-notes/
```

Every recording becomes `transcripts/<name>.txt`: Urdu first, then Roman Urdu.

## 4. Record the video (2 to 3 minutes long)

Before recording, put one or two **real Urdu voice notes** (yours, or a relative's with permission) in a folder, ideally
30 seconds to 2 minutes long. Install a screen recorder once: `sudo apt install obs-studio` (or Spectacle, if your
version has "Record screen"). Record the full screen at 1080p, with the microphone on.

Suggested script:

| Time | Show | Say |
| --- | --- | --- |
| 0:00 to 0:20 | The demo page | The problem: Pakistanis send Urdu voice notes all day and type in Roman Urdu; Urdu speech-to-text is poor and usually goes to the cloud |
| 0:20 to 0:35 | Terminal: `lscpu \| head -15`, `free -h`, then **turn Wi-Fi off** | "A normal laptop: i5, 12 GB, no GPU, and now no internet." |
| 0:35 to 0:45 | `python3 laptop.py demo` | One command |
| 0:45 to 1:25 | **Transcribe** tab: upload a real voice note | Text appears part by part in Urdu and Roman Urdu; press Copy and paste it into a chat or a document |
| 1:25 to 1:45 | Record yourself saying a sentence | Works on a live voice too |
| 1:45 to 2:15 | **Compare** tab, an unseen test clip | Proof the training worked: the original model writes Hindi script even when told Urdu; ours matches the human transcript |
| 2:15 to 2:40 | Terminal: `python3 laptop.py transcribe ~/voice-notes/`, then open a `.txt` | A whole folder at once, for interviews or lectures |
| 2:40 to 3:00 | `results/summary.md` and the Hugging Face page | Numbers (word errors 51% to 25%, model under 1 GB, about 2 GB RAM); what still fails (loanwords, other languages degrade); where to get it |

Keep it under 3 minutes. Upload it to YouTube as **Unlisted** (public link, not listed in search) and send me the link.

## Troubleshooting

- **`llama-server for ... stopped`**: open `results/server_<name>.log`; the last lines say why (usually out of memory:
  close apps and re-run).
- **Microphone gives nothing**: check the input device in the system settings (Plasma: Audio volume, Input devices);
  Firefox and Chrome both allow the microphone on `127.0.0.1`.
- **Port already in use**: for the page, `python3 laptop.py demo --port 8010`. The models use ports 8081 to 8083;
  if those are busy, stop the other program or reboot.
- **Download fails**: check the internet once; files that finished downloading are kept, so re-running continues.

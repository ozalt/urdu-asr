# Urdu Speech to Text (Urdu script and Roman Urdu)

Model: https://huggingface.co/zaltahir/qwen3-asr-0.6b-urdu · Code and full write-up: https://github.com/ozalt/urdu-asr
· Demo video: https://www.linkedin.com/posts/ozal-tahir_qwenlocalhackathon-activity-7531971240006414336-sRm4

## What it does

You speak Urdu, or drop in a WhatsApp voice note; it writes what was said, offline, on a CPU. One model, two outputs:

- **Urdu script** (اردو), for documents and search.
- **Roman Urdu** (Urdu in Latin letters), the way most Pakistanis actually type on their phones.

Audio never leaves the machine, which matters for voice notes, interviews and medical or legal dictation. The app has
copy and save buttons, handles recordings of any length (long audio is cut at pauses), and a batch command turns a
folder of recordings into text files.

## Why we built it

Qwen3-ASR-0.6B does not list Urdu among its 30 languages. Left to detect the language, it writes Urdu speech in Hindi
(Devanagari) script almost every time: 100.8% word error rate (WER). Even when told the language is Urdu it gets
about half the words wrong (51.3% WER). Roman Urdu it cannot produce at all. Fine-tuning on Urdu speech fixes the
script and roughly halves the errors, and a language tag lets the same model write either script.

## Dataset

| Part | Source and license | Size |
| --- | --- | --- |
| Train | Mozilla Common Voice Urdu (CC0) 33,046 clips + Google FLEURS Urdu train (CC-BY-4.0) 2,103 clips | 43.5 h |
| Dev (watched during training) | Common Voice dev + FLEURS validation | 500 clips |
| Test A | Common Voice official test split (speakers never in training) | 5,071 clips, 6.5 h |
| Test B | FLEURS test (different sentences, speakers and microphones) | 299 clips, 0.8 h |

Cleaning: clips outside 0.5 to 30 seconds, transcripts not in Urdu script, and clips faster than 20 characters per
second (mismatched audio) were dropped.

**Roman Urdu labels.** No public Urdu speech set has Roman Urdu transcripts, so we made them. We fine-tuned
Qwen3.5-0.8B (LoRA) as an Urdu to Roman Urdu romanizer on a cleaned version of Roman-Urdu-Parl, both published:
https://huggingface.co/zaltahir/qwen3.5-0.8b-urdu-romanizer and
https://huggingface.co/datasets/zaltahir/roman-urdu-parl-clean. On the Dakshina test set (native-speaker romanizations, never trained on) it
scores char-BLEU 76.2, against 75.1 reported for GPT-4o mini. It then wrote a Roman Urdu transcript for every
training clip; labels that failed automatic checks (leftover Urdu letters, wrong length, repetition loops) were
dropped (5 of 35,149). Each clip is trained twice, once per script: 70,293 examples. The labels are published:
https://huggingface.co/datasets/zaltahir/urdu-speech-transcripts-roman.

## Training

- Base: `Qwen/Qwen3-ASR-0.6B` (Apache-2.0) at revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, audio
  encoder plus Qwen3 decoder, 0.94B parameters.
- Method: **full fine-tune** (all weights), the official Qwen3-ASR SFT script on Hugging Face Transformers, with one
  fix (gradient accumulation loss scaling).
- Settings: 2 epochs, learning rate 2e-5, effective batch 64 (2 x 32 accumulation), bf16.
- Output format: `language Urdu<asr_text>...` or `language Roman urdu<asr_text>...`; the tag picks the script.
- Hardware: one RTX 5060 Ti 16 GB. Training time: about 2.2 hours (2,198 updates).
- Checkpoint: lowest dev loss (update 1,800 of 2,198).
- Software: Python 3.12.3, PyTorch 2.14.0 (CUDA 13.0), transformers 4.57.6, accelerate 1.12.0, qwen-asr 0.0.6;
  GGUF conversion and quantization with llama.cpp b12675.

## Run it locally

Runtime: llama.cpp (`llama-server`), CPU only. Quantization: GGUF Q8_0 decoder (639 MB) plus the audio encoder kept
at 16 bit (379 MB); a Q4_K_M decoder (397 MB) is also published.
Context 4096 tokens; audio goes in as pieces of at most 30 s (longer recordings are cut at pauses); empty
system prompt; greedy decoding (temperature 0), up to 512 output tokens per piece.

Easiest: the app (Linux, Python 3, nothing to install). The first run downloads a prebuilt llama.cpp and the model
files from Hugging Face.

```bash
git clone https://github.com/ozalt/urdu-asr && cd urdu-asr/Deliverables/laptop
python3 laptop.py demo                    # browser app: record or upload, Urdu + Roman Urdu
python3 laptop.py transcribe ~/notes/     # every recording in a folder -> text files
```

Or with llama.cpp directly; always set the language, since the client sends the prompt ending in
`language Urdu<asr_text>`:

```bash
llama-server -m urdu-asr-0.6b-q8_0.gguf --mmproj mmproj-urdu-asr-0.6b-f16.gguf --port 8080 &
python3 llama_client.py clip.wav --language Urdu          # or --language "Roman urdu"
```

Measured with `laptop.py bench`, CPU only: AMD Ryzen 5 7600 limited to 4 cores and 8 GB RAM, Linux Mint 22.1,
4 threads, 20 unseen FLEURS clips (205 s of audio):

| Model | WER / CER | Seconds of work per second of audio | Peak RAM |
| --- | --- | --- | --- |
| Base Qwen3-ASR Q8_0, told Urdu | 74.9% / 55.8% | 0.30 | 2.5 GB |
| **Fine-tuned Q8_0, Urdu** | **22.1% / 9.3%** | **0.28** | **2.3 GB** |
| Fine-tuned Q8_0, Roman Urdu | 30.6% / 12.0% | 0.21 | 2.3 GB |
| Fine-tuned Q4_K_M, Urdu | 23.5% / 9.9% | 0.21 | 2.2 GB |

In the app, writing both scripts on 4 cores, a 14-second voice note took 8 seconds and a 1-minute one 34 seconds.

## Before and after

Scores on held-out test sets (never seen in training), after removing punctuation. Lower is better.

| Model | Common Voice WER / CER | FLEURS WER / CER |
| --- | --- | --- |
| Base, auto language (writes Hindi script) | 100.8% / 84.3% | 100.6% / 86.3% |
| Base, told "Urdu" | 51.3% / 27.0% | 58.6% / 38.6% |
| **Fine-tuned, Urdu script** | **24.6% / 9.4%** | **26.1% / 10.6%** |
| **Fine-tuned, Roman Urdu** | 31.2% / 12.3% | 35.5% / 14.0% |
| Fine-tuned Q8_0 GGUF, Urdu (CPU) | 27.3% / 10.8% (500-clip sample; bf16 on it: 25.4%) | 25.8% / 10.4% |
| Fine-tuned Q4_K_M GGUF, Urdu (CPU) | 27.8% / 10.4% (same sample) | 29.6% / 12.8% |

Roman Urdu is scored against our romanizer's text of the reference transcript, and has no single correct spelling
("nahi" / "nahin"), so its error rate overstates real mistakes.

One FLEURS test clip (13.8 s), never seen in training:

- Human transcript: اس [ویلز] نے در اصل شروع سے ہی ہم سے جھوٹ بولا۔ سب سے پہلے تو اس طرح کے اقدام کرکے جیسے یہ قانونی وجوہات سے ہو۔ دوسرے نمبر پر، یہ ظاہر کرکے کہ وہ ہماری بات سن رہا ہے، بالکل اپنے حذف کرنے کے فن کی طرح۔
- Base, told Urdu (WER 100%): इसने دراصل शुरू से ही हमसे झूठ बोला, सबसे पहले तो इस तरह के आक्दाम करके जैसा ये कानूनी वजह से हो, दूसरे नंबर पर ये जाहिर करके वो हमारी बात सुन रहा है, बल्कि अपने हजफ करने की फन की तरह.
- Fine-tuned, Urdu (WER 23%): اس نے دراصل شروع سے ہی ہم سے جھوٹ بولا۔ سب سے پہلے تو اس طرح کے اقدام کر کے جیسا یہ قانونی وجوہات سے ہو، دوسرا نمبر پر یہ ظاہر کر کے وہ ہماری بات سن رہا ہے، بلکہ اپنے حذف کرنے کی فن کی طرح۔
- Fine-tuned, Roman Urdu: is ne darasal shuru se hi hum se jhoot bola. sab se pehlay to is terhan ke ikhdam kar ke jaisay yeh qanooni wajah se ho, dosray number par yeh zahir kar ke woh hamari baat sun rahay, balkay apne hizab karne ki fun ki terhan.

Judged by word and character error rate against the human transcript, with the same normalization for every model.

**What still fails**
- Other languages degrade, as expected from a full fine-tune on one language ("catastrophic forgetting"). On FLEURS,
  English told "English" went from 4.9% to 8.0% WER. Hindi told "Hindi" went from 19.3% to 62.5%: spoken Hindi and
  Urdu are nearly the same language, and about half the Hindi outputs now come out in Urdu script. Left to detect the
  language itself, the model now labels English and Hindi audio as Urdu, so the app always sets the language.
  Mixing some English and Hindi back into training ("replay") is the known fix; we did not do it.
- English loanwords and names in Roman Urdu are often spelled letter by letter.
- Typical Urdu errors: word spacing in compounds (کر کے vs کرکے counts as two wrong words, which is why CER is the
  fairer number), similar-sounding words (بلکہ for بالکل), and agreement endings (دوسرا for دوسرے).
- Read speech dominates training (Common Voice, FLEURS); fast, noisy or conversational speech is harder.

## What we learned

- Measure the honest baseline. "Base gets 100% WER" is mostly the wrong script; telling it the language halves that.
  We report both.
- Public Roman Urdu data is badly corrupted (for many common words the wrong romanization is the majority). Filtering
  by sound, not frequency, made the romanizer usable.
- Dev loss flattened after one epoch; the second epoch gained under 1 point of WER.

## Team

Ozal Tahir (https://github.com/ozalt), solo.

Credits: Qwen team (Qwen3-ASR, Qwen3.5), Mozilla Common Voice, Google FLEURS and Dakshina, Roman-Urdu-Parl
(Alam and Hussain, 2022), llama.cpp.

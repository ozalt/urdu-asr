# Urdu Speech to Text (Urdu script and Roman Urdu)

<!-- DRAFT. Lines marked TODO are filled once quantization, upload and video are done. No em dash characters
     anywhere in this file: the starter repo lint rejects them. -->

## What it does

You speak Urdu; it writes what you said, offline, on a normal laptop. One model, two outputs:

- **Urdu script** (اردو), for documents and search.
- **Roman Urdu** (Urdu in Latin letters), the way most Pakistanis actually type on their phones.

Audio never leaves the machine, which matters for voice notes, interviews and medical or legal dictation.

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
TODO link model, TODO link dataset. On the Dakshina test set (native-speaker romanizations, never trained on) it
scores char-BLEU 76.2, against 75.1 reported for GPT-4o mini. It then wrote a Roman Urdu transcript for every
training clip; labels that failed automatic checks (leftover Urdu letters, wrong length, repetition loops) were
dropped (5 of 35,149). Each clip is trained twice, once per script: 70,293 examples.

## Training

- Base: `Qwen/Qwen3-ASR-0.6B` (Apache-2.0), audio encoder plus Qwen3 decoder, 0.94B parameters.
- Method: **full fine-tune** (all weights), the official Qwen3-ASR SFT script on Hugging Face Transformers, with one
  fix (gradient accumulation loss scaling).
- Settings: 2 epochs, learning rate 2e-5, effective batch 64 (2 x 32 accumulation), bf16.
- Output format: `language Urdu<asr_text>...` or `language Roman urdu<asr_text>...`; the tag picks the script.
- Hardware: one RTX 5060 Ti 16 GB. Training time: TODO hours.
- Checkpoint: lowest dev loss (update 1,800 of 2,198).

## Run it locally

TODO after quantization: download GGUF + mmproj from the model repo, run with `llama-mtmd-cli`, state RAM, file
sizes, CPU, threads and real-time factor.

## Before and after

Scores on held-out test sets (never seen in training), after removing punctuation. Lower is better.

| Model | Common Voice WER / CER | FLEURS WER / CER |
| --- | --- | --- |
| Base, auto language (writes Hindi script) | 100.8% / 84.3% | 100.6% / 86.3% |
| Base, told "Urdu" | 51.3% / 27.0% | 58.6% / 38.6% |
| **Fine-tuned, Urdu script** | **24.6% / 9.4%** | **26.1% / 10.6%** |
| **Fine-tuned, Roman Urdu** | 31.2% / 12.3% | 35.5% / 14.0% |
| Fine-tuned, quantized TODO | TODO | TODO |

Roman Urdu is scored against our romanizer's text of the reference transcript, and has no single correct spelling
("nahi" / "nahin"), so its error rate overstates real mistakes.

TODO: one FLEURS clip, reference, base output, fine-tuned output (both scripts).

**What still fails**
- Other languages degrade, as expected from a full fine-tune on one language ("catastrophic forgetting"). On FLEURS,
  English told "English" went from 4.9% to 8.0% WER. Hindi told "Hindi" went from 19.3% to 62.5%: spoken Hindi and
  Urdu are nearly the same language, and about half the Hindi outputs now come out in Urdu script. Left to detect the
  language itself, the model now labels English and Hindi audio as Urdu, so the app always sets the language.
  Mixing some English and Hindi back into training ("replay") is the known fix; we did not do it.
- English loanwords and names in Roman Urdu are often spelled letter by letter.
- TODO: typical Urdu errors from the test outputs.

## What we learned

- Measure the honest baseline. "Base gets 100% WER" is mostly the wrong script; telling it the language halves that.
  We report both.
- Public Roman Urdu data is badly corrupted (for many common words the wrong romanization is the majority). Filtering
  by sound, not frequency, made the romanizer usable.
- Dev loss flattened after one epoch; the second epoch gained under 1 point of WER.

## Team

TODO names and GitHub profiles.

Credits: Qwen team (Qwen3-ASR, Qwen3.5), Mozilla Common Voice, Google FLEURS and Dakshina, Roman-Urdu-Parl
(Alam and Hussain, 2022), llama.cpp.

# Urdu Speech to Text: Qwen3-ASR-0.6B in Urdu script and Roman Urdu

Speak Urdu, get text, offline on a laptop CPU. One small model writes either **Urdu script** (اردو) or **Roman
Urdu** (Urdu in Latin letters, the way most Pakistanis actually type), chosen by a language tag.

Qwen3-ASR-0.6B does not list Urdu among its 30 languages. It understands spoken Urdu (spoken Urdu and Hindi are
nearly the same), but left to itself it writes Hindi script, and even when told the language is Urdu it gets about
half the words wrong. A full fine-tune on 43.5 hours of Urdu speech halves that error and adds Roman Urdu. Quantized
to 8 bits, it keeps its accuracy and runs faster than real time on an ordinary CPU, no GPU needed.

| | |
| --- | --- |
| Model (weights + GGUF) | [zaltahir/qwen3-asr-0.6b-urdu](https://huggingface.co/zaltahir/qwen3-asr-0.6b-urdu) |
| Roman Urdu training labels | [zaltahir/urdu-speech-transcripts-roman](https://huggingface.co/datasets/zaltahir/urdu-speech-transcripts-roman) |
| Romanizer used to make them | [zaltahir/qwen3.5-0.8b-urdu-romanizer](https://huggingface.co/zaltahir/qwen3.5-0.8b-urdu-romanizer) |
| Run it on a laptop | [Deliverables/laptop/LAPTOP.md](Deliverables/laptop/LAPTOP.md) |
| Demo video | TODO |

## Results

All scores are on test clips **never seen in training** (unseen speakers and sentences), after removing
punctuation. WER = word error rate, CER = character error rate; lower is better.

| Model | Common Voice test (5,071 clips) | FLEURS test (299 clips) |
| --- | --- | --- |
| Qwen3-ASR-0.6B, left to pick the language | 100.8% / 84.3% (writes Hindi script) | 100.6% / 86.3% |
| Qwen3-ASR-0.6B, told "Urdu" (the fair baseline) | 51.3% / 27.0% | 58.6% / 38.6% |
| **Fine-tuned, Urdu script** | **24.6% / 9.4%** | **26.1% / 10.6%** |
| **Fine-tuned, Roman Urdu** | **31.2% / 12.3%** | **35.5% / 14.0%** |

FLEURS is the harder, independent test: different sentences, speakers and microphones from the Common Voice data
the model mostly trained on. Roman Urdu is scored against the human transcripts romanized by our romanizer (there is
no human Roman Urdu for these clips), and Roman Urdu has no fixed spelling ("nahi" / "nahin"), so its scores overstate
real mistakes.

### Quantized, on CPU (llama.cpp)

| Version | Size | FLEURS Urdu | FLEURS Roman | Common Voice (500 clips) Urdu |
| --- | --- | --- | --- | --- |
| Full precision (bf16) | 1.9 GB | 26.1% / 10.6% | 35.5% / 14.0% | 25.4% / 9.1% |
| **Q8_0 GGUF** | 639 MB + 379 MB audio encoder | **25.8% / 10.4%** | **35.1% / 13.9%** | 27.3% / 10.8% |
| Q4_K_M GGUF | 397 MB + 379 MB audio encoder | 29.6% / 12.8% | 38.3% / 15.3% | 27.8% / 10.4% |

8-bit costs essentially nothing; 4-bit costs 2 to 4 points of WER. The audio encoder stays at 16 bit.

**On a laptop** (Intel i5-1145G7, 4 cores, 12 GB RAM, Kubuntu, CPU only): TODO from `laptop.py bench`.
On a desktop CPU (Ryzen 5 7600, 4 threads), Q8_0 transcribes a 10-second clip in about 3 to 5 seconds with about
2.2 GB of RAM.

### One clip, before and after

FLEURS test clip (13.8 s), never seen in training. Human transcript:

> اس [ویلز] نے در اصل شروع سے ہی ہم سے جھوٹ بولا۔ سب سے پہلے تو اس طرح کے اقدام کرکے جیسے یہ قانونی وجوہات سے ہو۔ دوسرے نمبر پر، یہ ظاہر کرکے کہ وہ ہماری بات سن رہا ہے، بالکل اپنے حذف کرنے کے فن کی طرح۔

| Model | Output | WER |
| --- | --- | --- |
| Base, auto | इसने दरअसल शुरू से ही हमसे झूठ बोला। सबसे पहले तो इस तरह के आक्दाम करके जैसा ये कानूनी वजह से हो, ... | 100% |
| Base, told Urdu | इसने دراصل शुरू से ही हमसे झूठ बोला, सबसे पहले तो इस तरह के आक्दाम करके ... (still Hindi script) | 100% |
| **Fine-tuned, Urdu** | اس نے دراصل شروع سے ہی ہم سے جھوٹ بولا۔ سب سے پہلے تو اس طرح کے اقدام کر کے جیسا یہ قانونی وجوہات سے ہو، دوسرا نمبر پر یہ ظاہر کر کے وہ ہماری بات سن رہا ہے، بلکہ اپنے حذف کرنے کی فن کی طرح۔ | 23% |
| **Fine-tuned, Roman** | is ne darasal shuru se hi hum se jhoot bola. sab se pehlay to is terhan ke ikhdam kar ke jaisay yeh qanooni wajah se ho, dosray number par yeh zahir kar ke woh hamari baat sun rahay, balkay apne hizab karne ki fun ki terhan. | 27% |

Several of the fine-tuned model's "errors" are only spacing (کر کے vs کرکے counts as two wrong words), which is why
CER is the fairer number for Urdu. Real errors remain too: بلکہ for بالکل, دوسرا for دوسرے. Every prediction for
FLEURS is in `results/*fleurs*.jsonl`.

### What still fails

- **Other languages degrade** (catastrophic forgetting). A full fine-tune on one language moves every weight toward it:

  | FLEURS, base vs fine-tuned | Base | Fine-tuned |
  | --- | --- | --- |
  | English, told "English" | 4.9% WER | 8.0% |
  | Hindi, told "Hindi" | 19.3% | 62.5% |

  Hindi suffers most because spoken Hindi and Urdu are nearly the same; about half the Hindi outputs now come out in
  Urdu script. Left to detect the language, the model now labels English and Hindi audio as Urdu, so **always set
  the language tag** (the demo and `llama_client.py` do). Mixing some English and Hindi back into training ("replay")
  is the known fix; we did not do it.
- English loanwords and names are often misspelled, most visibly in Roman Urdu.
- Training is read speech (Common Voice, FLEURS); fast, noisy or conversational speech will be harder.

## How it works

**Data.** Mozilla Common Voice 27.0 Urdu (CC0) and Google FLEURS Urdu (CC-BY-4.0). Common Voice's official dev/test
split is kept; training excludes every speaker and sentence that appears in any dev or test set. Clips shorter than
0.5 s, longer than 30 s, with Latin-letter transcripts, or faster than 20 characters per second (cut-off audio) are
dropped.

| Split | Clips | Hours |
| --- | --- | --- |
| Train (Common Voice 33,046 + FLEURS 2,103) | 35,149 | 43.5 |
| Dev, watched during training | 500 | 0.7 |
| Test: Common Voice | 5,071 | 6.5 |
| Test: FLEURS | 299 | 0.8 |

**Roman Urdu labels.** No Urdu speech corpus has Roman Urdu transcripts, so we made them in three steps:
1. Cleaned Roman-Urdu-Parl, a 6.3M-pair Urdu/Roman Urdu corpus in which machine "augmentation" made the *wrong*
   romanization the majority for many common words (یہ as "ki" in 70% of its rows). A sound-based filter (consonant
   skeletons must match, plus first-sound rules) keeps 1.6M pairs
   ([dataset](https://huggingface.co/datasets/zaltahir/roman-urdu-parl-clean)).
2. LoRA-fine-tuned Qwen3.5-0.8B on 50k of them as a romanizer. On Dakshina (native-speaker romanizations, never
   trained on) it scores char-BLEU 76.2, against 2.3 untouched and 75.1 reported for GPT-4o mini.
3. Romanized every human Urdu transcript with it; 5 of 35,149 failed automatic checks and were dropped.

Each training clip then appears twice, once per script: 70,293 examples. The target is
`language Urdu<asr_text>...` or `language Roman urdu<asr_text>...`; the tag in the prompt picks the script.

**Training.** Full fine-tune (all 938M weights) with Qwen's official SFT script (`qwen3_asr_sft.py`, plus one marked
fix: gradient-accumulation loss scaling). 2 epochs, learning rate 2e-5, 64 clips per update (2 x 32 accumulation),
bf16, on one RTX 5060 Ti 16 GB in about 2.2 hours. Dev loss flattened after about one epoch (0.266 at update 1,000,
0.263 at 1,800); we kept update 1,800, the lowest. The second epoch gained under 1 point of WER.

**Quantization.** llama.cpp converts the checkpoint directly (`quantize.py`): the language model to Q8_0 and
Q4_K_M, the audio encoder kept at 16 bit. `llama_client.py` forces the language by sending llama-server the same raw
prompt the Python package uses, ending in `language Urdu<asr_text>`.

## Run it

**On a laptop, one command** (Linux, Python 3, nothing to install): see
[Deliverables/laptop/LAPTOP.md](Deliverables/laptop/LAPTOP.md). `laptop.py demo` opens a browser page with
microphone, upload and test-clip inputs, showing the base model and ours side by side; `laptop.py bench` measures
speed, RAM and accuracy on 20 unseen clips.

**With llama.cpp directly:**

```bash
hf download zaltahir/qwen3-asr-0.6b-urdu urdu-asr-0.6b-q8_0.gguf mmproj-urdu-asr-0.6b-f16.gguf llama_client.py \
    --local-dir urdu-asr
cd urdu-asr
llama-server -m urdu-asr-0.6b-q8_0.gguf --mmproj mmproj-urdu-asr-0.6b-f16.gguf --port 8080 &
python3 llama_client.py my_clip.wav --language Urdu
python3 llama_client.py my_clip.wav --language "Roman urdu"
```

**In Python** (`pip install qwen-asr`):

```python
import torch
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference import utils
utils.SUPPORTED_LANGUAGES += ["Urdu", "Roman urdu"]

model = Qwen3ASRModel.from_pretrained("zaltahir/qwen3-asr-0.6b-urdu", dtype=torch.bfloat16, device_map="cuda:0")
print(model.transcribe(audio=["my_clip.wav"], language="Urdu")[0].text)
```

## Reproduce from scratch

Every step is a separate script, safe to re-run: finished work is skipped, splits and samples use fixed seeds,
decoding is greedy, and training resumes from its last checkpoint. `notebook.ipynb` walks through the same steps
with data exploration and charts.

| File | Step |
| --- | --- |
| `download.py` | 1. Download FLEURS Urdu and Common Voice Urdu into `data/raw/` |
| `prepare.py` | 2. 16 kHz WAV + train/dev/test lists in `data/` (and the Roman Urdu twins once labels exist) |
| `evaluate.py` | 3, 5. Score a model on the test sets, the same code for before and after |
| `train.sh` | 4. Full fine-tune with `qwen3_asr_sft.py` |
| `romanize/` | Stage 2: clean the text pairs, train and score the romanizer, label the transcripts |
| `other_langs.py`, `eval_all.sh` | 6. English/Hindi forgetting check; scores everything for the best checkpoint |
| `quantize.py`, `eval_gguf.sh` | 7. GGUF conversion and quantized scores (CPU) |
| `publish.py` | 8. Upload weights, GGUF and model card |
| `llama_client.py` | Transcribe through llama-server (standard library only) |
| `common.py` | Text cleanup, scoring normalization, label checks (`python common.py` runs its self-check) |

### 0. Setup

```bash
git clone https://github.com/ozalt/urdu-asr && cd urdu-asr
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -p .venv/bin/python -r requirements.txt
source .venv/bin/activate
python common.py            # prints: common.py self-check OK
```

### 1. Download

Common Voice needs a free Mozilla Data Collective account: sign up at https://mozilladatacollective.com, accept the
terms on the [Common Voice Urdu page](https://mozilladatacollective.com/datasets/cmu5x1w3400cyo107st8cgsoq) (a
one-time step on the website), and create an API key under Account -> Credentials. Its terms forbid re-sharing the
dataset, so nothing derived from it beyond Mozilla's public CC0 sentence list is published here.

```bash
MDC_API_KEY=your_key python download.py      # FLEURS ~2 GB, Common Voice 27.0 Urdu ~6.2 GB; re-run resumes
```

If Mozilla replaces the release, `python download.py --find-cv` lists the current one; pass its ID with
`--cv-dataset <id>`.

### 2. Prepare

```bash
python prepare.py           # 10 to 20 GB of WAV, up to an hour; prints data/stats.json
```

### 3. Baseline

```bash
python evaluate.py --name base                         # the model picks the language
python evaluate.py --name base_forced --language Urdu  # told it is Urdu: the fair "before"
```

Writes `results/<name>.json` (scores, output script, detected language) and `results/<name>_<set>.jsonl` (every
prediction). `--limit 1000` scores a fixed random sample. Both texts are normalized first (no punctuation or
diacritics, consistent letters), the standard practice.

### 4. Train

Check memory first on the 640 longest clips (memory depends on the longest clip in a batch):

```bash
python -c "import json; r = sorted(open('data/train.jsonl'), key=lambda l: -json.loads(l)['seconds']); open('data/smoke.jsonl', 'w').writelines(r[:640])"
TRAIN=data/smoke.jsonl OUT=runs/smoke EPOCHS=1 SAVE_STEPS=5 ./train.sh
rm -rf runs/smoke data/smoke.jsonl
./train.sh                  # log in runs/urdu-0.6b/train.log; re-run to resume
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `BATCH` | 2 | Clips per forward pass (4 runs out of memory on 30 s clips with 16 GB) |
| `ACCUM` | 32 | Forward passes per update: 64 clips per update |
| `LR` | 2e-5 | Learning rate (Qwen's default) |
| `EPOCHS` | 2 | Passes over the training data |

Watch `eval_loss` (every 200 updates, on dev clips): if it rises while `loss` falls, the model is memorizing; use an
earlier checkpoint.

### 5. Stage 2: Roman Urdu

The romanizer needs `transformers` 5.x, so it has its own environment:

```bash
uv venv -p 3.12 romanize/.venv
uv pip install -p romanize/.venv/bin/python torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -p romanize/.venv/bin/python -r romanize/requirements.txt
R=romanize/.venv/bin/python

$R romanize/prepare.py                       # download + clean the text pairs (CPU, ~5 min)
$R romanize/evaluate.py --name base          # before: untouched Qwen3.5-0.8B on Dakshina
$R romanize/train.py                         # LoRA fine-tune, 30 to 60 min
$R romanize/evaluate.py --name finetuned --model runs/romanize/final
$R romanize/label.py                         # romanize every transcript -> data/roman_labels.jsonl
python prepare.py                            # adds data/*_roman.jsonl
./train.sh                                   # trains on both scripts
```

### 6. Score everything, including forgetting

```bash
python other_langs.py       # FLEURS English + Hindi test sets
./eval_all.sh               # best checkpoint by dev loss: Urdu, Roman Urdu, English, Hindi, auto-detect
```

### 7. Quantize and score on CPU

Build llama.cpp once (CPU is enough):

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp tools/llama.cpp
cmake -S tools/llama.cpp -B tools/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build tools/llama.cpp/build -j 8 --target llama-server llama-quantize
python quantize.py          # runs/urdu-0.6b/gguf/: f16, q8_0, q4_k_m + audio encoder
./eval_gguf.sh              # FLEURS + 500 Common Voice clips per quantized model
```

### 8. Publish

```bash
python publish.py --ck runs/urdu-0.6b/checkpoint-1800 --card-only    # check the card first
python publish.py --ck runs/urdu-0.6b/checkpoint-1800                # needs HF_TOKEN in .env
Deliverables/build_kit.sh                                            # laptop kit zip
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `train.sh` says only N MiB free | Free the GPU (training needs about 15 GB) |
| `CUDA out of memory` during training | `BATCH=1 ACCUM=64 ./train.sh` |
| Loss stuck at a huge value (about 80) | Gradient-accumulation scaling: keep the marked fix in `qwen3_asr_sft.py` |
| `download.py` gives a 403 error | Accept the dataset's terms on the Mozilla website first |
| Fine-tuned model writes Roman Urdu or Urdu for other languages | Expected; always pass the language |

## Credits and licenses

Code: Apache-2.0. Base models: [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) by the Qwen team (Apache-2.0). Data:
[Mozilla Common Voice](https://commonvoice.mozilla.org) (CC0), [FLEURS](https://huggingface.co/datasets/google/fleurs)
(Conneau et al., 2022, CC-BY-4.0), [Roman-Urdu-Parl](https://huggingface.co/datasets/Mavkif/Roman-Urdu-Parl-split)
(Alam and Hussain, 2022), [Dakshina](https://github.com/google-research-datasets/dakshina) (test only). Runtime:
[llama.cpp](https://github.com/ggml-org/llama.cpp). Built for the Qwen Local Hackathon 2026 (Qwen Pakistan).

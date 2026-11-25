"""Stage 2, step 3: teach Qwen3.5-0.8B to write Urdu in Roman letters (LoRA). Re-run to resume.

    romanize/.venv/bin/python romanize/train.py          (GPU services paused; about 30-60 minutes)

LoRA: the 0.8B base weights stay frozen; small add-on matrices (rank 64, on every linear layer) are trained in
full precision. Romanizing is a narrow skill, so this is enough, and it avoids the bf16 rounding problem of
updating bf16 weights directly. At the end the add-ons are merged into the model, saved to <out>/final.
The model learns only the answer: the prompt tokens are masked out of the loss.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import prompt_text, stop_ids  # noqa: E402  (same prompt and stop tokens as the evaluation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--train", default="romanize/data/train.jsonl")
    ap.add_argument("--dev", default="romanize/data/dev.jsonl")
    ap.add_argument("--out", default="runs/romanize")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)  # usual LoRA rate; full fine-tunes use ~10x less
    # Qwen3.5 scores all 248k vocabulary entries at every position: 16 sentences per pass needed >15 GB.
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=16)  # 64 sentences per update
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--save-steps", type=int, default=500)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def encode(ex):
        prompt = tok(prompt_text(tok, ex["ur"]), add_special_tokens=False).input_ids
        answer = tok(ex["roman"] + tok.eos_token, add_special_tokens=False).input_ids
        return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}

    ds = load_dataset("json", data_files={"train": a.train, "dev": a.dev})
    ds = ds.map(encode, remove_columns=["ur", "roman"])

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda key, v: torch.tensor([b[key] + [v] * (n - len(b[key])) for b in batch])  # noqa: E731
        return {"input_ids": pad("input_ids", tok.pad_token_id), "labels": pad("labels", -100),
                "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"]))
                                                for b in batch])}

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=0.03, per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.accum, bf16=True, logging_steps=25, eval_strategy="steps",
        eval_steps=a.save_steps, save_steps=a.save_steps, save_total_limit=2,
        train_sampling_strategy="group_by_length",  # batches of similar length: less padding, faster
        report_to="none", dataloader_num_workers=2, seed=0, use_cpu=not torch.cuda.is_available())
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["dev"],
                      data_collator=collate)
    trainer.train(resume_from_checkpoint=any(Path(a.out).glob("checkpoint-*")) or None)

    info = {**vars(a), "train_pairs": len(ds["train"]), "dev_loss": trainer.evaluate()["eval_loss"]}
    final = Path(a.out) / "final"
    merged = model.merge_and_unload()
    merged.generation_config.eos_token_id = stop_ids(tok, merged)  # so anyone loading it stops at <|im_end|>
    merged.save_pretrained(final)
    tok.save_pretrained(final)
    (final / "training_info.json").write_text(json.dumps(info, indent=1))  # read by publish.py for the model card
    print("Merged model saved to", final)


if __name__ == "__main__":
    main()

# =============================================================================
# BART Fine-tuning for Multi-Dataset Dialogue Generation
# =============================================================================
# This script is designed to run on Google Colab with a T4 GPU.
#
# SETUP INSTRUCTIONS:
# 1. Open Google Colab → New Notebook → Runtime → Change runtime type → T4 GPU
# 2. Upload your model-ready JSONL files to Google Drive:
#       My Drive/NLP_Project/data/train.jsonl
#       My Drive/NLP_Project/data/validation.jsonl
#       My Drive/NLP_Project/data/test.jsonl
# 3. Copy each section below into a separate Colab cell and run sequentially.
# =============================================================================


# %%  =======================================================================
# CELL 1: Install Dependencies
# ============================================================================

# !pip install transformers==4.44.0 datasets evaluate rouge-score nltk accelerate -q
# import nltk
# nltk.download('punkt_tab', quiet=True)


# %%  =======================================================================
# CELL 2: Mount Google Drive & Configuration
# ============================================================================

import os
import json
import torch
import numpy as np
from pathlib import Path

# --- Mount Google Drive ---
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================================
# CONFIGURATION — edit these values as needed
# ============================================================================
CONFIG = {
    # --- Paths ---
    # For Colab, use Drive paths:
    # "data_dir":       "/content/drive/MyDrive/NLP_Project/data",
    # "output_dir":     "/content/bart_dialogue_output",
    # "cache_dir":      "/content/hf_cache",

    # For local testing, use project paths:
    "data_dir":       str(Path(__file__).resolve().parent / "data" / "model_ready"),
    "output_dir":     str(Path(__file__).resolve().parent / "output"),
    "cache_dir":      str(Path(__file__).resolve().parent / "hf_cache"),

    # --- Model ---
    "model_name":     "facebook/bart-base",

    # --- Tokenisation ---
    "max_input_length":   512,   # covers 99%+ of inputs (p99=310 words)
    "max_target_length":  128,   # covers 99%+ of targets (p99=41 words)

    # --- Training ---
    "per_device_train_batch_size":  8,
    "per_device_eval_batch_size":   16,
    "gradient_accumulation_steps":  4,      # effective batch = 32
    "learning_rate":                5e-5,
    "num_train_epochs":             3,
    "warmup_ratio":                 0.05,
    "weight_decay":                 0.01,
    "fp16":                         True,   # mixed precision for T4

    # --- Evaluation & Saving ---
    "eval_steps":                   2000,
    "save_steps":                   2000,
    "save_total_limit":             3,
    "logging_steps":                200,

    # --- Generation (during evaluation) ---
    "generation_max_length":        128,
    "generation_num_beams":         4,

    # --- Pilot mode: set to a small number (e.g. 5000) for a quick test,
    #     or None for the full dataset ---
    "max_train_samples":            5000,
    "max_eval_samples":             500,
}

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# %%  =======================================================================
# CELL 3: Load Data
# ============================================================================

from datasets import Dataset

def load_model_ready_jsonl(split: str) -> Dataset:
    """Load a model-ready JSONL file into a HuggingFace Dataset."""
    path = os.path.join(CONFIG["data_dir"], f"{split}.jsonl")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


print("Loading datasets...")
train_dataset = load_model_ready_jsonl("train")
val_dataset   = load_model_ready_jsonl("validation")
test_dataset  = load_model_ready_jsonl("test")

# Apply sample limits for pilot mode
if CONFIG["max_train_samples"]:
    train_dataset = train_dataset.select(range(min(CONFIG["max_train_samples"], len(train_dataset))))
if CONFIG["max_eval_samples"]:
    val_dataset  = val_dataset.select(range(min(CONFIG["max_eval_samples"], len(val_dataset))))
    test_dataset = test_dataset.select(range(min(CONFIG["max_eval_samples"], len(test_dataset))))

print(f"  Train:      {len(train_dataset):>8,} rows")
print(f"  Validation: {len(val_dataset):>8,} rows")
print(f"  Test:       {len(test_dataset):>8,} rows")

# Preview samples
print("\n--- Sample 1 ---")
print("INPUT:")
print(train_dataset[0]["input_text"])
print("\nTARGET:")
print(train_dataset[0]["target_text"])

print("\n--- Sample 2 ---")
print("INPUT:")
idx_100 = min(100, len(train_dataset) - 1)
print(train_dataset[idx_100]["input_text"])
print("\nTARGET:")
print(train_dataset[idx_100]["target_text"])


# %%  =======================================================================
# CELL 4: Tokenization
# ============================================================================

from transformers import AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["model_name"],
    cache_dir=CONFIG["cache_dir"],
)

def tokenize_function(examples):
    """Tokenize input_text and target_text for BART seq2seq training."""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=CONFIG["max_input_length"],
        truncation=True,
        padding=False,   # DataCollator will handle padding
    )

    # Tokenize targets
    labels = tokenizer(
        examples["target_text"],
        max_length=CONFIG["max_target_length"],
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Tokenizing train set...")
train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["input_text", "target_text"],
    desc="Tokenizing train",
)

print("Tokenizing validation set...")
val_tokenized = val_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["input_text", "target_text"],
    desc="Tokenizing validation",
)

print("Tokenizing test set...")
test_tokenized = test_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["input_text", "target_text"],
    desc="Tokenizing test",
)

# Show token length stats
train_input_lens = [len(x["input_ids"]) for x in train_tokenized]
train_label_lens = [len(x["labels"]) for x in train_tokenized]
print(f"\nToken length stats (train):")
print(f"  Input  — mean: {np.mean(train_input_lens):.0f}, "
      f"p95: {np.percentile(train_input_lens, 95):.0f}, "
      f"max: {max(train_input_lens)}")
print(f"  Target — mean: {np.mean(train_label_lens):.0f}, "
      f"p95: {np.percentile(train_label_lens, 95):.0f}, "
      f"max: {max(train_label_lens)}")


# %%  =======================================================================
# CELL 5: Model & Training Setup
# ============================================================================

import evaluate
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# --- Load model ---
print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    CONFIG["model_name"],
    cache_dir=CONFIG["cache_dir"],
)

# Enable gradient checkpointing to save memory
model.config.use_cache = False  # required when gradient_checkpointing is on

print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable:        {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# --- Data collator ---
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,  # ignore padding tokens in loss
)

# --- Metrics ---
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    """Compute ROUGE scores on generated predictions."""
    preds, labels = eval_preds

    # Decode predictions
    # Replace -100 in labels (padding) with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds  = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute ROUGE
    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    return {
        "rouge1":  round(result["rouge1"], 4),
        "rouge2":  round(result["rouge2"], 4),
        "rougeL":  round(result["rougeL"], 4),
    }


# --- Training arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir=CONFIG["output_dir"],

    # Training
    num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_ratio=CONFIG["warmup_ratio"],
    weight_decay=CONFIG["weight_decay"],
    fp16=CONFIG["fp16"],
    gradient_checkpointing=True,

    # Evaluation
    eval_strategy="steps",
    eval_steps=CONFIG["eval_steps"],
    predict_with_generate=True,
    generation_max_length=CONFIG["generation_max_length"],
    generation_num_beams=CONFIG["generation_num_beams"],

    # Saving
    save_strategy="steps",
    save_steps=CONFIG["save_steps"],
    save_total_limit=CONFIG["save_total_limit"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_rougeL",
    greater_is_better=True,

    # Logging
    logging_steps=CONFIG["logging_steps"],
    logging_first_step=True,
    report_to="none",   # set to "tensorboard" if you want TB logging

    # Performance
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)

# --- Trainer ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

total_steps = (len(train_tokenized) // (CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"])) * CONFIG["num_train_epochs"]
print(f"\nTraining plan:")
print(f"  Effective batch size: {CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps']}")
print(f"  Steps per epoch:     {len(train_tokenized) // (CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps']):,}")
print(f"  Total steps:         {total_steps:,}")
print(f"  Eval every:          {CONFIG['eval_steps']} steps")
print(f"\nReady to train!")


# %%  =======================================================================
# CELL 6: Train
# ============================================================================

print("Starting training...")
train_result = trainer.train()

# Print training summary
print("\n--- Training complete ---")
metrics = train_result.metrics
print(f"  Total steps:    {metrics.get('total_flos', 'N/A')}")
print(f"  Training loss:  {metrics.get('train_loss', 'N/A'):.4f}")
print(f"  Runtime:        {metrics.get('train_runtime', 0) / 3600:.1f} hours")

# Save the final model
trainer.save_model(os.path.join(CONFIG["output_dir"], "final_model"))
tokenizer.save_pretrained(os.path.join(CONFIG["output_dir"], "final_model"))
print(f"  Model saved to: {CONFIG['output_dir']}/final_model")


# %%  =======================================================================
# CELL 7: Evaluate on Test Set
# ============================================================================

print("Evaluating on test set...")
test_results = trainer.predict(test_tokenized)

print("\n--- Test Results ---")
for key, value in test_results.metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Show sample predictions
print("\n--- Sample Predictions ---\n")
preds = test_results.predictions
labels = test_results.label_ids

# Replace -100 with pad_token_id for decoding
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

for i in range(min(10, len(decoded_preds))):
    # Also get the original input text
    original_input = test_dataset[i]["input_text"]
    print(f"--- Example {i+1} ---")
    print(f"INPUT:\n{original_input}\n")
    print(f"EXPECTED:  {decoded_labels[i].strip()}")
    print(f"GENERATED: {decoded_preds[i].strip()}")
    print()


# %%  =======================================================================
# CELL 8: Interactive Demo
# ============================================================================

def generate_response(
    context: str = "",
    persona: str = "",
    emotion: str = "",
    history: str = "",
    player_input: str = "",
    do_sample: bool = True,
    num_beams: int = 1,
    max_length: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
):
    """Generate a response given dialogue context fields.

    Handles sampling mode (do_sample=True, top_p, temperature, num_beams=1)
    and beam search mode (do_sample=False, num_beams=4, early_stopping=True)
    correctly by excluding mismatched options.
    """

    # Build input text in the same format as training data
    parts = []
    if context.strip():
        parts.append(f"Context: {context.strip()}")
    if persona.strip():
        parts.append(f"Persona: {persona.strip()}")
    if emotion.strip():
        parts.append(f"Emotion: {emotion.strip()}")
    if history.strip():
        parts.append(f"History:\n{history.strip()}")
    parts.append(f"Player: {player_input.strip()}")
    parts.append("Generate Response:")

    input_text = "\n".join(parts)

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=CONFIG["max_input_length"],
        truncation=True,
    ).to(model.device)

    # Setup generation arguments dynamically to avoid conflicts
    gen_kwargs = {
        "max_length": max_length,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }

    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["num_beams"] = num_beams
    else:
        gen_kwargs["do_sample"] = False
        gen_kwargs["num_beams"] = num_beams

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_kwargs
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# --- Demo examples ---
print("=" * 60)
print("Interactive Demo")
print("=" * 60)

# Example 1: LIGHT-style
print("\n--- Example 1: Fantasy RPG ---")
resp = generate_response(
    context="Dark Forest: A dense, ancient forest where shadows move between the trees.",
    persona="guardian: I have protected these woods for centuries. I trust no outsider.",
    player_input="Who are you? Why do you block our path?",
)
print(f"Response: {resp}")

# Example 2: PersonaChat-style
print("\n--- Example 2: Casual Chat ---")
resp = generate_response(
    persona="I love cooking Italian food. | I have two cats. | I work as a teacher.",
    player_input="What do you like to do on weekends?",
)
print(f"Response: {resp}")

# Example 3: Empathetic-style
print("\n--- Example 3: Empathetic Response ---")
resp = generate_response(
    context="My dog passed away last week. He was with me for 12 years.",
    emotion="sad",
    player_input="I just feel so empty without him around.",
)
print(f"Response: {resp}")

# Example 4: DailyDialog-style
print("\n--- Example 4: Daily Conversation ---")
resp = generate_response(
    history="Hi, how's your day going?\nPretty good, just got back from the gym.",
    player_input="Do you go to the gym often?",
)
print(f"Response: {resp}")


# %%  =======================================================================
# CELL 9: Save to Google Drive (Colab only)
# ============================================================================

# Uncomment these lines when running on Google Colab:

# import shutil
# drive_output = "/content/drive/MyDrive/NLP_Project/bart_trained_model"
# os.makedirs(drive_output, exist_ok=True)
#
# model_dir = os.path.join(CONFIG["output_dir"], "final_model")
# print(f"Copying model from {model_dir} to {drive_output}...")
# shutil.copytree(model_dir, drive_output, dirs_exist_ok=True)
# print(f"Model saved to Google Drive: {drive_output}")
# print("You can now safely close this Colab session.")

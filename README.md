# Game Dialogue Generator

Context-aware dialogue generation pipeline for text-based games and interactive storytelling.

This repository is a working-stage NLP project focused on preparing dialogue data and fine-tuning a transformer model to generate game-style responses. It is not a finished game yet. The current goal is to build a reliable dialogue generation foundation that can later be plugged into a larger RPG, story engine, or text adventure.

## Project Status

Current stage: data pipeline + model training prototype.

What is working right now:

- Dataset download scripts for the source corpora.
- Dataset conversion into one unified dialogue schema.
- Model-ready preprocessing for BART fine-tuning.
- A Colab-friendly training script for `facebook/bart-base`.

What is not finished yet:

- No playable game UI.
- No production API or backend service.
- No live inference loop integrated into a game engine.
- No polished deployment pipeline.

## What This Project Is

The aim is to generate dialogue that feels like it belongs inside a game world, not a generic chatbot reply.

The model is trained to take inputs such as:

- scene or context
- character persona
- emotion label
- conversation history
- player input

and predict the next response in a way that stays coherent, character-aware, and situation-aware.

## Why These Datasets

The project combines several dialogue datasets because each one teaches a different part of the behavior we want in a game NPC.

- DailyDialog teaches natural turn-to-turn conversation.
- PersonaChat teaches persona consistency and speaking in character.
- EmpatheticDialogues teaches emotional responses.
- LIGHT teaches fantasy-style roleplay and scene-driven dialogue.

That mix is important because no single dataset covers all of those behaviors well on its own.

## Repository Workflow

The pipeline is designed in three main stages:

1. Download the raw datasets.
2. Convert them into one shared master format.
3. Prepare model-ready input and target text pairs for BART training.

### 1) Download source data

Run `download_datasets.py` to fetch the source dialogue corpora into the `data/` folder.

It currently handles:

- DailyDialog
- PersonaChat
- EmpatheticDialogues
- LIGHT

### 2) Convert into a unified schema

Run `convert_datasets.py` to normalize all datasets into one structure.

The unified row format is:

```json
{
  "conversation_id": "...",
  "turn_id": 0,
  "source_dataset": "dailydialog",
  "context": "...",
  "persona": "...",
  "emotion": "...",
  "history": "...",
  "player_input": "...",
  "response": "..."
}
```

This is stored in:

- `data/unified/train.jsonl`
- `data/unified/validation.jsonl`
- `data/unified/test.jsonl`

### 3) Prepare model-ready training data

Run `prepare_model_data.py` to convert the unified rows into the final text pair format used for sequence-to-sequence training.

Each record becomes:

- `input_text`
- `target_text`

The input text is built in a structured prompt style, for example:

```text
Context: Dark cave
Persona: Suspicious guardian
Emotion: Tense
History:
Player: Who are you?
NPC: I guard this place.
Player: Why are you blocking the path?
Generate Response:
```

The model learns to generate the matching `target_text` as the response.

## Current Data Snapshot

The processed data in this working copy is already built and ready for training. In the GitHub repo, these generated datasets should stay local and be recreated from the scripts instead of being committed.

### Unified data

- Train: 321,549 rows
- Validation: 25,227 rows
- Test: 21,285 rows

### Dataset contribution in the current unified split files

- DailyDialog: 76,052 train / 7,069 validation / 6,740 test
- PersonaChat: 131,438 train / 7,801 validation / no test split in the source data used here
- EmpatheticDialogues: 36,629 train / 5,712 validation / 5,242 test
- LIGHT: 77,430 train / 4,645 validation / 9,303 test

### Model-ready data

The final `data/model_ready/` files contain the compact training format used by the BART script.

Observed stats from the current processed snapshot:

- Train: 321,549 examples
- Validation: 25,227 examples
- Test: 21,285 examples

Input lengths are intentionally kept within BART-friendly limits, and the prompts are built to preserve context without making the format too verbose.

## Training Setup

The training script is in `train_bart_colab.py`.

It is written in a Colab-friendly style, but it can also be adapted for a local environment. The current configuration uses:

- model: `facebook/bart-base`
- max input length: 512
- max target length: 128
- mixed precision: enabled for GPU runs
- pilot mode sample limits for faster experimentation

The script loads the model-ready JSONL files, tokenizes them, and prepares a seq2seq training loop with ROUGE-based evaluation.

## File Structure

```text
.
├── convert_datasets.py
├── download_datasets.py
├── prepare_model_data.py
├── train_bart_colab.py
├── data/
│   ├── dailydialog/
│   ├── empathetic_dialogues/
│   ├── light/
│   ├── personachat/
│   ├── unified/
│   └── model_ready/
├── Documentation.txt
├── Dataset descrption.txt
└── Ideation.txt
```

## How To Run

### Install dependencies

Create a virtual environment and install the packages used by the pipeline.

```bash
pip install datasets requests tqdm transformers evaluate rouge-score nltk accelerate torch
```

### Download the datasets

```bash
python download_datasets.py
```

### Build the unified dataset

```bash
python convert_datasets.py
```

### Prepare model-ready files

```bash
python prepare_model_data.py
```

### Train the model

Open `train_bart_colab.py` in Google Colab or adapt the config for local execution, then run the cells or script sections in order.

## Design Notes

This project is intentionally structured around dialogue context instead of plain question-answer pairs.

The prompt format keeps the following signals separate:

- environment or scene
- speaker persona
- emotion
- prior turns
- current player input

That structure makes it easier for the model to learn when a response should sound friendly, guarded, emotional, fantasy-themed, or strictly narrative-driven.

## Known Limitations

This is still a work in progress, so a few things are deliberately left open:

- The project does not yet have a front-end game interface.
- The training setup is a starting point, not a fully optimized final run.
- Dataset balance and sampling strategy may still be refined.
- Some source datasets have different split availability, so the unified output is not perfectly symmetric across all corpora.

## Next Steps

The most natural future additions are:

- a lightweight text-based game demo
- an inference API for NPC responses
- checkpoint saving and model comparison
- stronger evaluation for coherence, persona consistency, and emotion handling
- a small leaderboard of model experiments

## Notes

The `Dataset descrption.txt`, `Documentation.txt`, and `Ideation.txt` files are kept in the repo as working notes and project planning material. This README is the main entry point for the current state of the project.

The `data/` folder is treated as generated output for the published repo. Run the download and preprocessing scripts to rebuild it in a fresh clone.

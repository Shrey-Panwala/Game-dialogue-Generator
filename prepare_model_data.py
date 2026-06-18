"""
Prepare Model-Ready Data
=========================
Converts unified master-schema JSONL into compact model-ready files
containing only `input_text` and `target_text` fields for BART fine-tuning.

Runs locally (CPU only, ~1-2 min).  Output goes to  data/model_ready/

Usage:
    python prepare_model_data.py
"""

import json
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (portable: derived from this script's location)
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "unified"
OUT_DIR   = BASE_DIR / "data" / "model_ready"


# ---------------------------------------------------------------------------
# Core: build the structured input string from a master-schema row
# ---------------------------------------------------------------------------

def build_input_text(row: dict) -> str:
    """
    Convert a master-schema row into the structured input string that BART
    will see as encoder input.

    Format (empty fields are omitted, not left blank):
        Context: <context>
        Persona: <persona>
        Emotion: <emotion>
        History:
        <line1>
        <line2>
        Player: <player_input>
        Generate Response:
    """
    parts = []

    if row.get("context", "").strip():
        parts.append(f"Context: {row['context'].strip()}")

    if row.get("persona", "").strip():
        parts.append(f"Persona: {row['persona'].strip()}")

    if row.get("emotion", "").strip():
        parts.append(f"Emotion: {row['emotion'].strip()}")

    if row.get("history", "").strip():
        parts.append(f"History:\n{row['history'].strip()}")

    parts.append(f"Player: {row['player_input'].strip()}")
    parts.append("Generate Response:")

    return "\n".join(parts)


def build_target_text(row: dict) -> str:
    """Extract the raw response as the target text."""
    return row["response"].strip()


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_split(split: str) -> dict:
    """Process one split. Returns stats dict."""
    in_path = INPUT_DIR / f"{split}.jsonl"
    if not in_path.exists():
        print(f"  [SKIP] {in_path} not found")
        return {}

    out_path = OUT_DIR / f"{split}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_lengths  = []
    target_lengths = []
    ds_counts      = {}
    total          = 0
    skipped        = 0

    tmp = out_path.with_suffix(".tmp")
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(tmp, "w", encoding="utf-8") as fout:

        for line in fin:
            row = json.loads(line)

            input_text  = build_input_text(row)
            target_text = build_target_text(row)

            # Skip rows with empty input or target
            if not input_text.strip() or not target_text.strip():
                skipped += 1
                continue

            fout.write(json.dumps({
                "input_text":  input_text,
                "target_text": target_text,
            }, ensure_ascii=False) + "\n")

            # Track stats  (word count as rough proxy for tokens)
            input_lengths.append(len(input_text.split()))
            target_lengths.append(len(target_text.split()))

            ds = row.get("source_dataset", "unknown")
            ds_counts[ds] = ds_counts.get(ds, 0) + 1
            total += 1

    tmp.replace(out_path)

    stats = {
        "split":   split,
        "total":   total,
        "skipped": skipped,
        "per_dataset": ds_counts,
        "input_word_lengths": {
            "min":   min(input_lengths)   if input_lengths else 0,
            "max":   max(input_lengths)   if input_lengths else 0,
            "mean":  round(statistics.mean(input_lengths), 1)   if input_lengths else 0,
            "p50":   round(statistics.median(input_lengths), 1) if input_lengths else 0,
            "p95":   sorted(input_lengths)[int(len(input_lengths) * 0.95)] if input_lengths else 0,
            "p99":   sorted(input_lengths)[int(len(input_lengths) * 0.99)] if input_lengths else 0,
        },
        "target_word_lengths": {
            "min":   min(target_lengths)   if target_lengths else 0,
            "max":   max(target_lengths)   if target_lengths else 0,
            "mean":  round(statistics.mean(target_lengths), 1)   if target_lengths else 0,
            "p50":   round(statistics.median(target_lengths), 1) if target_lengths else 0,
            "p95":   sorted(target_lengths)[int(len(target_lengths) * 0.95)] if target_lengths else 0,
            "p99":   sorted(target_lengths)[int(len(target_lengths) * 0.99)] if target_lengths else 0,
        },
    }

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Prepare Model-Ready Data")
    print("=" * 60)

    all_stats = {}

    for split in ["train", "validation", "test"]:
        print(f"\n--- {split} ---")
        stats = process_split(split)
        if stats:
            all_stats[split] = stats
            print(f"  Total rows:   {stats['total']:>8,}")
            print(f"  Skipped:      {stats['skipped']:>8,}")
            for ds, cnt in sorted(stats["per_dataset"].items()):
                print(f"    {ds:>15}: {cnt:>7,}")

            il = stats["input_word_lengths"]
            tl = stats["target_word_lengths"]
            print(f"  Input  words: min={il['min']}, mean={il['mean']}, "
                  f"p50={il['p50']}, p95={il['p95']}, p99={il['p99']}, max={il['max']}")
            print(f"  Target words: min={tl['min']}, mean={tl['mean']}, "
                  f"p50={tl['p50']}, p95={tl['p95']}, p99={tl['p99']}, max={tl['max']}")

    # Write combined stats
    stats_path = OUT_DIR / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\nStats written to {stats_path}")

    # Print a sample
    sample_path = OUT_DIR / "train.jsonl"
    if sample_path.exists():
        print("\n--- Sample input/target pair ---\n")
        with open(sample_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 2:  # pick the 3rd sample for variety
                    row = json.loads(line)
                    print("INPUT TEXT:")
                    print(row["input_text"])
                    print("\nTARGET TEXT:")
                    print(row["target_text"])
                    break

    print("\n" + "=" * 60)
    total = sum(s["total"] for s in all_stats.values())
    print(f"Grand total: {total:,} model-ready rows")
    print(f"Output dir:  {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

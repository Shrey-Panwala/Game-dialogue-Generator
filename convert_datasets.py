"""
Dataset Conversion Pipeline
============================
Converts all four raw datasets into the unified master schema:

  conversation_id : str   – unique ID per conversation
  turn_id         : int   – turn number within conversation (0-indexed)
  source_dataset  : str   – which dataset this came from
  context         : str   – scene / situation description
  persona         : str   – character persona string(s)
  emotion         : str   – emotion label
  history         : str   – prior dialogue turns (newline-separated)
  player_input    : str   – the current user utterance
  response        : str   – the target response

Each converter reads raw JSONL files and yields rows in this schema.
Output is written to  data/unified/{split}.jsonl
"""

import json
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths  (portable: derived from this script's location)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = DATA_DIR / "unified"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: Path):
    """Write a list of dicts to a JSONL file (atomic via .tmp)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def clean_text(text: str) -> str:
    """Basic text clean-up shared across datasets."""
    if not text:
        return ""
    # Fix EmpatheticDialogues encoding artefacts
    text = text.replace("_comma_", ",")
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


# ---------------------------------------------------------------------------
# 1. DailyDialog converter
# ---------------------------------------------------------------------------
# Raw schema:  dialogue_id, domains, turns[]
# Each turn:   speaker (user/system), utterance, emotion, dialogue_acts
#
# Strategy: walk consecutive turn pairs.  For every turn at index i (i >= 1),
# the history is turns[0..i-1], player_input = turns[i-1], response = turns[i].
# We take the *response* turn's emotion as the emotion label for that row.
# ---------------------------------------------------------------------------

def convert_dailydialog(split: str) -> list[dict]:
    path = DATA_DIR / "dailydialog" / f"dailydialog_{split}.jsonl"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return []

    raw = read_jsonl(path)
    rows = []

    for conv in raw:
        conv_id = conv["dialogue_id"]
        turns = conv["turns"]

        if len(turns) < 2:
            continue  # need at least one pair

        for i in range(1, len(turns)):
            # Build history from all turns before the current pair
            history_parts = []
            for j in range(i - 1):
                utterance = clean_text(turns[j]["utterance"])
                role = "NPC" if (i - j) % 2 == 0 else "Player"
                history_parts.append(f"{role}: {utterance}")
            history_str = "\n".join(history_parts) if history_parts else ""

            player_input = clean_text(turns[i - 1]["utterance"])
            response     = clean_text(turns[i]["utterance"])

            # Use the response turn's emotion; normalise "no emotion" → ""
            raw_emotion = turns[i].get("emotion", "")
            emotion = "" if raw_emotion in ("no emotion", "no_emotion", "") else raw_emotion

            # DailyDialog has domain info but no scene/persona
            context_parts = conv.get("domains", [])
            context = ", ".join(context_parts) if context_parts else ""

            rows.append({
                "conversation_id": conv_id,
                "turn_id":         i - 1,          # 0-indexed pair index
                "source_dataset":  "dailydialog",
                "context":         context,
                "persona":         "",
                "emotion":         emotion,
                "history":         history_str,
                "player_input":    player_input,
                "response":        response,
            })

    return rows


# ---------------------------------------------------------------------------
# 2. PersonaChat converter
# ---------------------------------------------------------------------------
# Raw schema:  personality[], utterances[]
# Each utterance: history[] (growing list), candidates[] (last = gold)
#
# Strategy: each utterance entry is one training example.
# history[-1] is the player_input, history[:-1] is the prior context,
# and candidates[-1] is the gold response.
# ---------------------------------------------------------------------------

def convert_personachat(split: str) -> list[dict]:
    path = DATA_DIR / "personachat" / f"personachat_{split}.jsonl"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return []

    raw = read_jsonl(path)
    rows = []

    for idx, entry in enumerate(raw):
        # Persona: join all personality sentences
        persona_lines = entry.get("personality", [])
        persona_str = " | ".join(clean_text(p) for p in persona_lines)

        utterances = entry.get("utterances", [])
        conv_id = f"personachat-{split}-{idx}"

        for turn_id, utt in enumerate(utterances):
            history_list = utt.get("history", [])
            candidates   = utt.get("candidates", [])

            if not history_list or not candidates:
                continue

            # player_input = last history entry; prior = everything before it
            player_input = clean_text(history_list[-1])
            prior = []
            L = len(history_list)
            for k in range(L - 1):
                utterance = clean_text(history_list[k])
                dist = (L - 1) - k
                role = "NPC" if dist % 2 != 0 else "Player"
                prior.append(f"{role}: {utterance}")
            history_str = "\n".join(prior) if prior else ""

            # Gold response is the last candidate
            response = clean_text(candidates[-1])

            rows.append({
                "conversation_id": conv_id,
                "turn_id":         turn_id,
                "source_dataset":  "personachat",
                "context":         "",
                "persona":         persona_str,
                "emotion":         "",
                "history":         history_str,
                "player_input":    player_input,
                "response":        response,
            })

    return rows


# ---------------------------------------------------------------------------
# 3. EmpatheticDialogues converter
# ---------------------------------------------------------------------------
# Raw schema:  one row per utterance, grouped by conv_id.
# Fields:      conv_id, utterance_idx, context (=emotion label),
#              prompt (situation), speaker_idx, utterance
#
# Strategy: group rows by conv_id, sort by utterance_idx.
# For every turn at index i (i >= 1), treat history = turns[0..i-2],
# player_input = turns[i-1], response = turns[i].
# We generate a training sample only when the *listener* responds
# (even utterance_idx → speaker_idx for listener).
# ---------------------------------------------------------------------------

def convert_empathetic(split: str) -> list[dict]:
    fname = f"empathetic_{split}.jsonl"
    path = DATA_DIR / "empathetic_dialogues" / fname
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return []

    raw = read_jsonl(path)

    # Group by conversation
    convos = defaultdict(list)
    for row in raw:
        convos[row["conv_id"]].append(row)

    rows = []

    for conv_id, turns in convos.items():
        # Sort by utterance index
        turns.sort(key=lambda t: t["utterance_idx"])

        if len(turns) < 2:
            continue

        # Emotion and prompt (situation) are shared across the conversation
        emotion = clean_text(turns[0].get("context", ""))
        situation = clean_text(turns[0].get("prompt", ""))

        # Identify the two speakers in this conversation
        # The first speaker (utterance_idx=1) is the "speaker" (shares emotion)
        # The second speaker is the "listener" (responds empathetically)
        speaker_id = turns[0]["speaker_idx"]  # the emotion-sharer

        pair_idx = 0
        for i in range(1, len(turns)):
            # We only create training samples where the LISTENER responds
            # (i.e., the responder is NOT the emotion-sharer)
            if turns[i]["speaker_idx"] == speaker_id:
                continue  # skip: emotion-sharer speaking again

            # Build history from all previous turns (before the pair)
            history_parts = []
            for j in range(i - 1):
                utterance = clean_text(turns[j]["utterance"])
                role = "Player" if turns[j]["speaker_idx"] == speaker_id else "NPC"
                history_parts.append(f"{role}: {utterance}")
            history_str = "\n".join(history_parts) if history_parts else ""

            player_input = clean_text(turns[i - 1]["utterance"])
            response     = clean_text(turns[i]["utterance"])

            rows.append({
                "conversation_id": conv_id,
                "turn_id":         pair_idx,
                "source_dataset":  "empathetic",
                "context":         situation,   # the emotional situation prompt
                "persona":         "",
                "emotion":         emotion,
                "history":         history_str,
                "player_input":    player_input,
                "response":        response,
            })
            pair_idx += 1

    return rows


# ---------------------------------------------------------------------------
# 4. LIGHT converter
# ---------------------------------------------------------------------------
# Raw schema:  task, setting{name, description}, characters{self_name,
#              partner_name, self_persona}, dialogue[] (alternating strings)
#
# Strategy: dialogue is a flat list of alternating utterances.
# self_name speaks at even indices (0, 2, 4, …),
# partner_name speaks at odd indices (1, 3, 5, …).
# For every turn at index i (i >= 1), history = dialogue[0..i-2],
# player_input = dialogue[i-1], response = dialogue[i].
# We generate samples where the "self" character responds.
# ---------------------------------------------------------------------------

def convert_light(split: str) -> list[dict]:
    path = DATA_DIR / "light" / f"light_{split}.jsonl"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return []

    raw = read_jsonl(path)
    rows = []

    for idx, entry in enumerate(raw):
        conv_id = f"light-{split}-{idx}"

        setting = entry.get("setting", {})
        scene_name = clean_text(setting.get("name", ""))
        scene_desc = clean_text(setting.get("description", ""))
        context = f"{scene_name}: {scene_desc}" if scene_name else scene_desc

        chars = entry.get("characters", {})
        self_name    = clean_text(chars.get("self_name", ""))
        self_persona = clean_text(chars.get("self_persona", ""))

        # Persona: include character name and persona description
        persona = f"{self_name}: {self_persona}" if self_name else self_persona

        dialogue = entry.get("dialogue", [])
        if len(dialogue) < 2:
            continue

        # self speaks at even indices (0, 2, 4, …),
        # partner speaks at odd indices (1, 3, 5, …).
        # We only create training samples where "self" responds,
        # i.e., response index i is even (i = 2, 4, 6, …).
        # This ensures persona consistency: the model learns to
        # generate replies as the character whose persona we provide.
        pair_idx = 0
        for i in range(2, len(dialogue), 2):
            # Build history from all turns before the partner's prompt
            history_parts = []
            for j in range(i - 1):
                utterance = clean_text(dialogue[j])
                role = "NPC" if j % 2 == 0 else "Player"
                history_parts.append(f"{role}: {utterance}")
            history_str = "\n".join(history_parts) if history_parts else ""

            player_input = clean_text(dialogue[i - 1])  # partner's utterance
            response     = clean_text(dialogue[i])       # self's reply

            rows.append({
                "conversation_id": conv_id,
                "turn_id":         pair_idx,
                "source_dataset":  "light",
                "context":         context,
                "persona":         persona,
                "emotion":         "",
                "history":         history_str,
                "player_input":    player_input,
                "response":        response,
            })
            pair_idx += 1

    return rows


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

CONVERTERS = {
    "dailydialog":  convert_dailydialog,
    "personachat":  convert_personachat,
    "empathetic":   convert_empathetic,
    "light":        convert_light,
}

SPLITS = ["train", "validation", "test"]


def main():
    print("=" * 60)
    print("Dataset Conversion Pipeline")
    print("=" * 60)

    # Aggregate rows per split
    split_rows: dict[str, list[dict]] = {s: [] for s in SPLITS}

    for name, converter in CONVERTERS.items():
        print(f"\n--- Converting: {name} ---")
        for split in SPLITS:
            rows = converter(split)
            if rows:
                split_rows[split].extend(rows)
                print(f"  {split:>12}: {len(rows):>7,} rows")

    # Write unified files
    print(f"\n--- Writing unified files to {OUT_DIR} ---")
    for split in SPLITS:
        rows = split_rows[split]
        if rows:
            out_path = OUT_DIR / f"{split}.jsonl"
            write_jsonl(rows, out_path)
            print(f"  {split:>12}: {len(rows):>7,} rows -> {out_path.name}")

    # Summary
    print("\n" + "=" * 60)
    total = sum(len(v) for v in split_rows.values())
    print(f"Total unified rows: {total:,}")
    print("=" * 60)

    # Quick sanity check: print one sample from each dataset
    print("\n--- Sample rows (one per dataset) ---\n")
    seen = set()
    for split in SPLITS:
        for row in split_rows[split]:
            ds = row["source_dataset"]
            if ds not in seen:
                seen.add(ds)
                print(json.dumps(row, indent=2, ensure_ascii=False))
                print()
            if len(seen) == 4:
                break
        if len(seen) == 4:
            break


if __name__ == "__main__":
    main()

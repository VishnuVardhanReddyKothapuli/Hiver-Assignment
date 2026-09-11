"""Resumable terminal review; never pre-fills a human judgment."""
import datetime
import os
from .common import DATA, INTENTS, read_csv, read_jsonl, write_csv
from .evaluation import examples_with_split
from .judge import DIMENSIONS

def atomic_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    write_csv(temporary, rows, list(rows[0]))
    os.replace(temporary, path)

def ask(prompt, choices=None):
    while True:
        value = input(prompt + " (q to save and quit): ").strip()
        if value == "q":
            raise KeyboardInterrupt
        if value and (choices is None or value in choices):
            return value
        print("Enter one of " + ", ".join(choices) if choices else "A nonempty answer is required.")

def annotate(kind="intent", partition=None):
    path = DATA / ("human_labels.csv" if kind == "intent" else "reply_ratings.csv")
    rows = read_csv(path)
    examples = {r["id"]: r for r in examples_with_split()} if kind == "intent" else {r["sample_id"]: r for r in read_jsonl(DATA/"reply_review.jsonl")}
    annotator = ask("Your name or reviewer ID")
    try:
        for n, row in enumerate(rows):
            if row.get("label_source") == "human":
                continue
            key = row["id"] if kind == "intent" else row["sample_id"]
            example = examples[key]
            if partition and example.get("split", example.get("partition")) != partition:
                continue
            print(f"\nExample {n+1}/{len(rows)} | {key}")
            print("Prior context:", example.get("prior_context") or "[unavailable]")
            print("Customer:", example["text"])
            updated = dict(row)
            if kind == "intent":
                print("Intents:", ", ".join(INTENTS))
                updated["intent"] = ask("Primary intent", INTENTS)
                updated["should_escalate"] = ask("Requires a human given the available context?", ("true", "false"))
                updated["labelled_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            else:
                print("Draft:", example["reply"])
                print("System escalates:", example["should_escalate"])
                for evidence in example["evidence"]:
                    print(f"Evidence {evidence['reply_id']}: {evidence['text']} -> {evidence['reply']}")
                for d in DIMENSIONS:
                    updated[d] = ask(d + " score", ("0", "1", "2"))
            updated["reason"] = ask("Brief rationale (include ambiguity or evidence as needed)")
            updated.update(annotator=annotator, label_source="human")
            rows[n] = updated
            atomic_csv(path, rows)
    except (KeyboardInterrupt, EOFError):
        print("\nSaved completed examples; resume with the same command.")


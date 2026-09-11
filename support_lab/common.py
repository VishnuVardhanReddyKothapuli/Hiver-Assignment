import csv
import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
INTENTS = ("account_access", "billing", "subscription", "playback", "library", "how_to", "feedback", "other")
STOP = set("a an the i me my you your we our it its is are was were be been to of for on in at and or but this that with have has do does did can could would please hi hey spotify spotifycares thanks thank just im so not".split())

def clean(text):
    text = html.unescape(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", " [EMAIL] ", text)
    text = re.sub(r"@[\w_]+", " ", text)
    text = re.sub(r"\b(?:\+?\d[\d ()-]{7,}\d)\b", " [NUMBER] ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokens(text):
    return [w for w in re.findall(r"[a-z]+", clean(text).lower()) if w not in STOP and len(w) > 1]

def normalized(text):
    return " ".join(re.findall(r"[a-z0-9]+", clean(text).lower()))

def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

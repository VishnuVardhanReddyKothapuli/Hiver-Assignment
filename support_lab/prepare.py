"""Streaming source extraction; group-disjoint sampling before weak labels."""
import argparse
import csv
import io
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from email.utils import parsedate_to_datetime
from pathlib import Path

from .common import DATA, clean, digest, file_hash, normalized, write_csv, write_json, write_jsonl

SOURCE = "https://www.kaggle.com/api/v1/datasets/download/thoughtvector/customer-support-on-twitter"

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while x != root:
            previous = self.parent[x]
            self.parent[x] = root
            x = previous
        return root

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

def source_rows(path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith("twcs.csv")]
            if len(names) != 1:
                raise ValueError("Archive must contain exactly one twcs.csv")
            with io.TextIOWrapper(z.open(names[0]), encoding="utf-8") as f:
                yield from csv.DictReader(f)
    else:
        with path.open(encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)

def prepare(path, brand="SpotifyCares", seed=41, train_limit=6000):
    if (DATA / "human_labels.csv").exists():
        raise FileExistsError("Refusing to overwrite an existing annotation set")
    # First pass finds both directions of every direct brand/customer edge.
    brand_rows = {}
    wanted = set()
    count = 0
    for r in source_rows(path):
        count += 1
        if r["author_id"] == brand and r["inbound"].lower() == "false":
            brand_rows[r["tweet_id"]] = r
            wanted.update(x for x in (r["in_response_to_tweet_id"] + "," + r["response_tweet_id"]).split(",") if x)
    print(f"Read {count:,} tweets; {len(brand_rows):,} brand tweets", flush=True)
    customers = {}
    for r in source_rows(path):
        if r["inbound"].lower() == "true" and (r["tweet_id"] in wanted or r["in_response_to_tweet_id"] in brand_rows):
            customers[r["tweet_id"]] = r
    graph = UnionFind()
    for r in list(brand_rows.values()) + list(customers.values()):
        node = "t:" + r["tweet_id"]
        if r["in_response_to_tweet_id"]:
            graph.union(node, "t:" + r["in_response_to_tweet_id"])
        for reply in r["response_tweet_id"].split(","):
            if reply:
                graph.union(node, "t:" + reply)
        if r["tweet_id"] in customers:
            graph.union(node, "u:" + r["author_id"])
            # Exact normalized repeats remain in one split, including short replies.
            graph.union(node, "d:" + digest(normalized(r["text"])))
    replies = defaultdict(list)
    for r in brand_rows.values():
        if r["in_response_to_tweet_id"] in customers:
            replies[r["in_response_to_tweet_id"]].append(r)
    pairs = []
    discarded = Counter()
    for tid, choices in sorted(replies.items()):
        c = customers[tid]
        valid = [r for r in choices if parsedate_to_datetime(r["created_at"]) >= parsedate_to_datetime(c["created_at"])]
        if not valid:
            discarded["reply_precedes_message"] += 1
            continue
        reply = min(valid, key=lambda r: (parsedate_to_datetime(r["created_at"]), r["tweet_id"]))
        if not normalized(c["text"]):
            discarded["empty_message"] += 1
            continue
        parent = brand_rows.get(c["in_response_to_tweet_id"], customers.get(c["in_response_to_tweet_id"]))
        if parent and parsedate_to_datetime(parent["created_at"]) > parsedate_to_datetime(c["created_at"]):
            parent = None
        pairs.append({"id": tid, "reply_id": reply["tweet_id"], "author_id": c["author_id"],
                      "group_id": digest(graph.find("t:" + tid))[:20], "text": clean(c["text"]),
                      "reply": clean(reply["text"]), "created_at": parsedate_to_datetime(c["created_at"]).isoformat(),
                      "reply_created_at": parsedate_to_datetime(reply["created_at"]).isoformat(),
                      "prior_context": clean(parent["text"]) if parent else "",
                      "is_followup": bool(c["in_response_to_tweet_id"]), "brand": brand})
    # Group hash split is stable, with no label-dependent allocation or test tuning.
    partitions = defaultdict(list)
    for row in pairs:
        bucket = int(digest(str(seed) + row["group_id"])[:8], 16) % 100
        partitions["train" if bucket < 75 else "dev" if bucket < 85 else "test"].append(row)
    rng = random.Random(seed)
    chosen = {}
    for split, size in (("train", train_limit), ("dev", 40), ("test", 160)):
        pool = partitions[split]
        # One example per group in evaluation avoids dependent bootstrap observations.
        if split != "train":
            groups = defaultdict(list)
            for row in pool:
                groups[row["group_id"]].append(row)
            pool = [rng.choice(sorted(v, key=lambda r: r["id"])) for _, v in sorted(groups.items())]
        if len(pool) < size:
            raise ValueError(f"Need {size} {split} examples, got {len(pool)}")
        chosen[split] = sorted(rng.sample(pool, size), key=lambda r: r["id"])
        write_jsonl(DATA / f"{split}.jsonl", chosen[split])
    labels = [{"id": r["id"], "split": split, "intent": "", "should_escalate": "", "reason": "",
               "annotator": "", "label_source": "", "labelled_at": ""}
              for split in ("dev", "test") for r in chosen[split]]
    label_path = DATA / "human_labels.csv"
    if label_path.exists():
        raise FileExistsError("Refusing to overwrite human_labels.csv; preserve it before resampling")
    write_csv(label_path, labels, list(labels[0]))
    manifest = {"source_url": SOURCE, "source_sha256": file_hash(path), "source_rows": count,
                "brand": brand, "brand_tweets": len(brand_rows), "customer_tweets": len(customers),
                "eligible_pairs": len(pairs), "groups": len({r["group_id"] for r in pairs}),
                "seed": seed, "discarded": dict(discarded),
                "pool_counts": {k: len(v) for k, v in partitions.items()},
                "sampling": "75/10/15 group hash split; uniform groups then one random message per group for evaluation",
                "files": {f"{s}.jsonl": {"rows": len(v), "sha256": file_hash(DATA / f"{s}.jsonl")} for s, v in chosen.items()}}
    write_json(DATA / "manifest.json", manifest)
    print(manifest, flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DATA / "raw/twcs.zip")
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if (DATA / "human_labels.csv").exists():
        p.error("data/human_labels.csv already exists; do not overwrite annotation work")
    if args.download and not args.source.exists():
        args.source.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.source.with_suffix(".part")
        urllib.request.urlretrieve(SOURCE, temporary)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Download was not a ZIP; download twcs.csv manually from Kaggle")
        temporary.replace(args.source)
    prepare(args.source)

if __name__ == "__main__":
    main()

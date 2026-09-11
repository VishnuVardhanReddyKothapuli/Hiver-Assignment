import json
import platform
import random
import time
from collections import Counter

from .agent import Agent, weak_escalate, weak_intent
from .common import ARTIFACTS, DATA, INTENTS, digest, file_hash, normalized, read_csv, read_jsonl, write_csv, write_json, write_jsonl
from .metrics import evaluate

SYSTEMS = ("trivial", "simple", "retrieval")
LABEL_FIELDS = ("id", "split", "intent", "should_escalate", "reason", "annotator", "label_source", "labelled_at")

def validate_labels(rows, examples, require_complete=True):
    lookup = {r["id"]: r for r in examples}
    seen, labels = set(), {}
    for row in rows:
        tid = row["id"]
        if tid in seen:
            raise ValueError(f"Duplicate human label: {tid}")
        seen.add(tid)
        if tid not in lookup:
            raise ValueError(f"Unknown label id: {tid}")
        if not row["intent"] and not require_complete:
            continue
        if row["intent"] not in INTENTS or row["should_escalate"] not in ("true", "false"):
            raise ValueError(f"Missing/invalid labels for {tid}")
        if row["label_source"] != "human" or not all(row.get(k, "").strip() for k in ("reason", "annotator", "labelled_at")):
            raise ValueError(f"Human provenance incomplete for {tid}")
        if row["split"] != lookup[tid]["split"]:
            raise ValueError(f"Split changed for {tid}")
        labels[tid] = row
    if require_complete and set(labels) != set(lookup):
        raise ValueError("Human labels must cover exactly the frozen dev and test IDs")
    return labels

def audit_splits(parts):
    checks = {}
    for i, a in enumerate(parts):
        for b in list(parts)[i+1:]:
            for field in ("id", "reply_id", "author_id", "group_id"):
                overlap = {r[field] for r in parts[a]} & {r[field] for r in parts[b]}
                checks[f"{a}/{b}/{field}"] = len(overlap)
            checks[f"{a}/{b}/normalized_text"] = len({normalized(r["text"]) for r in parts[a]} & {normalized(r["text"]) for r in parts[b]})
    if any(checks.values()):
        raise ValueError(f"Split leakage: {checks}")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name in parts:
        if file_hash(DATA / f"{name}.jsonl") != manifest["files"][f"{name}.jsonl"]["sha256"]:
            raise ValueError(f"{name} file no longer matches frozen manifest")
    return checks

def examples_with_split():
    return [dict(r, split=s) for s in ("dev", "test") for r in read_jsonl(DATA / f"{s}.jsonl")]

def run(mode="diagnostic", threshold=.55):
    started = time.perf_counter()
    parts = {s: read_jsonl(DATA / f"{s}.jsonl") for s in ("train", "dev", "test")}
    checks = audit_splits(parts)
    examples = examples_with_split()
    if mode == "gold":
        labels = validate_labels(read_csv(DATA / "human_labels.csv"), examples)
    else:
        labels = {r["id"]: {"intent": weak_intent(r["text"]), "should_escalate": str(weak_escalate(r["text"])).lower()} for r in examples}
    agent = Agent(parts["train"], threshold=threshold)
    test = parts["test"]
    y = [labels[r["id"]]["intent"] for r in test]
    escalations = [labels[r["id"]]["should_escalate"] == "true" for r in test]
    all_predictions, results = [], {}
    for system in SYSTEMS:
        predictions, latency = [], []
        for row in test:
            tick = time.perf_counter()
            pred = agent.predict(row["text"], system, row["is_followup"]).as_dict()
            latency.append((time.perf_counter()-tick)*1000)
            predictions.append(pred)
            all_predictions.append(dict(pred, id=row["id"], system=system, text=row["text"],
                                        prior_context=row["prior_context"], is_followup=row["is_followup"]))
        result = evaluate(y, escalations, predictions)
        result["latency_ms_median"] = sorted(latency)[len(latency)//2]
        result["latency_ms_p95"] = sorted(latency)[int(.95*(len(latency)-1))]
        result["routing_reasons"] = dict(Counter(p["reason"] for p in predictions))
        result["slices"] = {}
        for name, idx in (("followup", [i for i,r in enumerate(test) if r["is_followup"]]),
                          ("first_contact", [i for i,r in enumerate(test) if not r["is_followup"]])):
            result["slices"][name] = evaluate([y[i] for i in idx], [escalations[i] for i in idx], [predictions[i] for i in idx]) if idx else {"n": 0}
        results[system] = result
    report = {"status": "HUMAN_LABELLED" if mode == "gold" else "DIAGNOSTIC_ONLY_NOT_ACCURACY_EVIDENCE",
              "label_source": "human" if mode == "gold" else "training_heuristic_circular",
              "test_n": len(test), "threshold": threshold, "seed": 41, "python": platform.python_version(),
              "data_hashes": {s: file_hash(DATA/f"{s}.jsonl") for s in parts},
              "split_overlap_counts": checks, "systems": results, "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["Historical reply is not a verified resolution", "Routing safety does not measure reply correctness", "Score is not calibrated probability", "Only one brand and old Twitter data"]}
    if mode == "gold":
        report["human_labels_sha256"] = file_hash(DATA / "human_labels.csv")
    write_json(ARTIFACTS / f"{mode}_results.json", report)
    write_jsonl(ARTIFACTS / f"{mode}_predictions.jsonl", all_predictions)
    lines = [f"# {'Human-labelled evaluation' if mode == 'gold' else 'Diagnostic: agreement with training heuristics'}", "",
             "These are human-label results." if mode == "gold" else "**Not independent accuracy. The target labels come from the same heuristics used to train the models.**",
             "", "| System | N | Intent macro-F1 | Intent accuracy | Auto coverage | Unsafe auto / auto | Escalation recall |", "|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "N/A" if v is None else f"{v:.3f}"
    for name, m in results.items():
        lines.append(f"| {name} | {m['n']} | {fmt(m['macro_f1_fixed_8'])} | {fmt(m['accuracy'])} | {fmt(m['coverage'])} | {m['missed_escalations']}/{m['auto_count']} | {fmt(m['escalation_recall'])} |")
    lines.extend(["", "Intervals, class support, confusion matrices, follow-up slices, and routing reasons are in the JSON artifact.",
                  "Zero auto-handled examples means safety precision is undefined, not 100%. Human reply quality and judge agreement require separate ratings."])
    (ARTIFACTS/f"{mode}_results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))
    return report

def make_review(predictions_path):
    """Freeze 20 messages x three systems, paired and blinded for rating."""
    if (DATA/"reply_ratings.csv").exists():
        raise FileExistsError("Reply ratings already exist; preserve them before making a new set")
    predictions = read_jsonl(predictions_path)
    ids = sorted({p["id"] for p in predictions})
    ids = random.Random(53).sample(ids, min(20, len(ids)))
    pool, keys = [], []
    for p in predictions:
        if p["id"] not in ids:
            continue
        sample_id = digest(p["system"] + p["id"] + p["reply"])[:16]
        partition = "calibration" if ids.index(p["id"]) < 10 else "audit"
        pool.append({"sample_id": sample_id, "partition": partition, "message_id": p["id"], "text": p["text"],
                     "reply": p["reply"], "evidence": p["evidence"], "prior_context": p["prior_context"],
                     "should_escalate": p["should_escalate"]})
        keys.append({"sample_id": sample_id, "system": p["system"], "message_id": p["id"]})
    random.Random(71).shuffle(pool)
    write_jsonl(DATA/"reply_review.jsonl", pool)
    write_jsonl(ARTIFACTS/"reply_review_key.jsonl", keys)
    fields = ["sample_id", "grounding", "relevance", "safety", "helpfulness", "reason", "annotator", "label_source"]
    write_csv(DATA/"reply_ratings.csv", [{k: row["sample_id"] if k == "sample_id" else "" for k in fields} for row in pool], fields)
    write_json(ARTIFACTS/"reply_review_manifest.json", {"predictions_sha256": file_hash(predictions_path),
               "review_sha256": file_hash(DATA/"reply_review.jsonl"), "sampling": "20 uniform message IDs, all three systems; 10 calibration and 10 audit messages"})
    print(f"Prepared {len(pool)} blinded reply ratings; system key is separate.")

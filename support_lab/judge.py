import json
from .common import ARTIFACTS, DATA, ROOT, file_hash, read_csv, read_jsonl, write_json, write_jsonl
from .llm import structured_call
from .metrics import kappa, ratio

DIMENSIONS = ("grounding", "relevance", "safety", "helpfulness")

def validate_rating(value):
    if not all(type(value.get(d)) is int and 0 <= value[d] <= 2 for d in DIMENSIONS):
        raise ValueError("Each rubric score must be an integer 0, 1, or 2")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError("A rating must include a reason")

def passes(v):
    return v["grounding"] == 2 and v["safety"] == 2 and v["relevance"] == 2 and v["helpfulness"] >= 1

def run_judge(model):
    prompt = (ROOT/"prompts/judge.txt").read_text(encoding="utf-8")
    schema = {"type": "object", "additionalProperties": False,
              "properties": {**{d: {"type": "integer", "enum": [0, 1, 2]} for d in DIMENSIONS}, "reason": {"type": "string"}},
              "required": [*DIMENSIONS, "reason"]}
    output = []
    review = read_jsonl(DATA/"reply_review.jsonl")
    for i, row in enumerate(review):
        # No human scores, model/system name, partition, or future historical reply.
        payload = {k: row[k] for k in ("text", "reply", "evidence", "prior_context", "should_escalate")}
        result = structured_call(model, prompt, payload, "reply_quality", schema)
        validate_rating(result["value"])
        output.append(dict(result, sample_id=row["sample_id"], partition=row["partition"],
                           review_sha256=file_hash(DATA/"reply_review.jsonl"), prompt_sha256=file_hash(ROOT/"prompts/judge.txt")))
        print(f"Judged {i+1}/{len(review)}", flush=True)
    write_jsonl(ARTIFACTS/"judge_scores.jsonl", output)

def agreement():
    humans = read_csv(DATA/"reply_ratings.csv")
    judges = read_jsonl(ARTIFACTS/"judge_scores.jsonl")
    review = read_jsonl(DATA/"reply_review.jsonl")
    expected = {r["sample_id"] for r in review}
    h, j = {}, {}
    for row in humans:
        if row["sample_id"] in h:
            raise ValueError("Duplicate human reply rating")
        if row["label_source"] != "human" or not row["annotator"].strip():
            raise ValueError("Complete human reply ratings before reporting agreement")
        scores = {d: int(row[d]) for d in DIMENSIONS}
        scores["reason"] = row["reason"]
        validate_rating(scores)
        h[row["sample_id"]] = scores
    for row in judges:
        if row["sample_id"] in j or row["review_sha256"] != file_hash(DATA/"reply_review.jsonl"):
            raise ValueError("Duplicate or stale judge scores")
        validate_rating(row["value"])
        j[row["sample_id"]] = row["value"]
    if set(h) != expected or set(j) != expected:
        raise ValueError("Human and judge ratings must exactly cover the frozen reply review set")
    if len({r["model"] for r in judges}) != 1 or len({r["prompt_sha256"] for r in judges}) != 1:
        raise ValueError("Agreement requires one frozen judge model and rubric")
    output = {"status": "MEASURED_HUMAN_JUDGE_AGREEMENT", "human_ratings_sha256": file_hash(DATA/"reply_ratings.csv"),
              "judge_scores_sha256": file_hash(ARTIFACTS/"judge_scores.jsonl"), "partitions": {}}
    key = {r["sample_id"]: r["system"] for r in read_jsonl(ARTIFACTS/"reply_review_key.jsonl")}
    for partition in ("calibration", "audit"):
        ids = [r["sample_id"] for r in review if r["partition"] == partition]
        result = {"n": len(ids), "distinct_messages": len({r["message_id"] for r in review if r["partition"] == partition})}
        for d in DIMENSIONS:
            a, b = [h[i][d] for i in ids], [j[i][d] for i in ids]
            result[d] = {"exact_agreement": ratio(sum(x == y for x,y in zip(a,b)), len(ids)),
                         "quadratic_weighted_kappa": kappa(a,b,True),
                         "mean_absolute_error": ratio(sum(abs(x-y) for x,y in zip(a,b)),len(ids))}
        a, b = [int(passes(h[i])) for i in ids], [int(passes(j[i])) for i in ids]
        result["pass_fail"] = {"kappa": kappa(a,b), "agreement": ratio(sum(x == y for x,y in zip(a,b)), len(ids)),
                               "judge_false_passes": sum(x == 0 and y == 1 for x,y in zip(a,b)),
                               "human_fails": a.count(0)}
        result["by_system"] = {s: {"n": sum(key[i] == s for i in ids),
                                  "human_pass_rate": ratio(sum(passes(h[i]) for i in ids if key[i] == s), sum(key[i] == s for i in ids)),
                                  "judge_pass_rate": ratio(sum(passes(j[i]) for i in ids if key[i] == s), sum(key[i] == s for i in ids))}
                                for s in sorted(set(key.values()))}
        result["disagreements"] = [{"sample_id": i, "human": h[i], "judge": j[i]} for i in ids if any(h[i][d] != j[i][d] for d in DIMENSIONS)]
        output["partitions"][partition] = result
    write_json(ARTIFACTS/"judge_agreement.json", output)
    print(json.dumps(output, indent=2))

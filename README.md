# SpotifyCares support lab — Hiver take-home

An evidence-first support-agent experiment on real Twitter conversations. It classifies messages into eight intents, retrieves historical replies, drafts a response, and returns an explicit auto/human routing reason. Everything needed for the offline experiment is included; Python **3.11+**, no packages, GPU, API key, or dataset download required.

**Submission status: runnable implementation, measured diagnostics, and review queues are complete. The required human labeling and human–LLM judge agreement are not yet complete.** No machine labels are presented as hand labels. The current retrieval policy auto-handles 0/160 test messages, so it has not demonstrated useful automation. See [the report](REPORT.md) for the negative result and why it matters.

## Reproduce in under 15 minutes

From this repository's root, in PowerShell, Command Prompt, or a Unix shell:

```sh
python --version
python scripts/reproduce.py
```

This runs the regression tests, verifies the frozen data hashes and split isolation, and regenerates [diagnostic results](artifacts/diagnostic_results.md), [full metrics](artifacts/diagnostic_results.json), and [all predictions](artifacts/diagnostic_predictions.jsonl). It prints its actual elapsed time. The bundled 6,000 training pairs and 200 evaluation examples are a small derived subset of the **real Kaggle dataset**, not fabricated fixtures. Network access is unnecessary. Human annotation and first-time API calls are outside this reproduction time budget and must not be described as completed in 15 minutes.

Verified locally on Python 3.14.6 / Windows: **20 tests passed; full offline reproduction took 9.06 seconds**. Runtime depends on the machine. No live API calls or human judgments are included in that measurement.

Try the agent:

```sh
python -m support_lab predict "My music keeps pausing during playback"
python -m support_lab predict "I was charged twice for Premium"
python -m support_lab predict "It still does not work" --followup
```

Output includes `intent`, an **uncalibrated ranking score**, `reply`, `should_escalate`, a machine-readable `reason`, historical evidence with dataset tweet IDs and cosine similarities, and the reused `source_reply_id` when applicable. It drafts only; it does not post messages or operate Spotify accounts. Known follow-ups must be marked by the caller. Historical data is not a current Spotify policy source.

## Complete the human evidence

Read the [annotation protocol and data card](data/README.md). The frozen queue is 40 development examples plus **160 held-out test examples**, within the requested 150–250 labeling budget.

```sh
python -m support_lab review --partition dev
python -m support_lab review --partition test
python -m support_lab evaluate --mode gold
```

The terminal tool shows the incoming text and available prior context, without a suggested label or future brand answer. Enter your judgments and short rationale; each complete row saves immediately. Type `q` to stop and run the same command to resume. You may instead edit [human_labels.csv](data/human_labels.csv). Freeze model/routing choices after development, before inspecting test labels. Gold mode rejects incomplete, duplicate, unknown, or non-human-provenance annotations. It cannot verify the truth of a reviewer's attestation.

The separate [reply review queue](data/reply_review.jsonl) contains 60 drafts: the same 20 messages across three systems, shuffled and with system names hidden. Thirty are rubric calibration, thirty are held-out audit. Read the [judge rubric](prompts/judge.txt), then:

```sh
python -m support_lab review --kind reply --partition calibration
python -m support_lab review --kind reply --partition audit
```

The existing reply review queue is frozen against the shipped diagnostic predictions. If model outputs change, preserve old ratings and create a new review set with `python -m support_lab prepare-replies --predictions artifacts/gold_predictions.jsonl`; the command refuses to overwrite an existing ratings CSV. Keep each reviewed experiment's predictions, key, rubric, and ratings together. Do not reuse scores for changed drafts.

## Optional live LLM drafting and judge

Set `OPENAI_API_KEY` in your environment and choose an explicit model ID available to your account. Never commit the key. The adapter uses the [OpenAI Responses API with structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs). No model is silently selected, and no API results are included as if they had been run.

```sh
python -m support_lab predict "My music keeps pausing" --model YOUR_MODEL_ID
python -m support_lab judge --model YOUR_JUDGE_MODEL_ID
python -m support_lab agreement
```

These commands make billable API calls. LLM drafting produces a human-review candidate; valid JSON and citation IDs do not prove grounding. Judge mode scores the existing 60 frozen **offline-system drafts**, not the output of an unrelated `predict --model` call. It sends cleaned message/context, draft, and retrieved evidence; never human scores or the hidden system name. `store=false` is set on requests; this is not a claim of zero provider retention. Exact requests are content-hash cached in the gitignored `artifacts/cache` directory for resumability. Failed calls do not generate scores. Run agreement only after the 60 human ratings and real judge calls exist. The result distinguishes calibration from audit and includes per-system human/judge pass rates and false passes.

## Systems and evaluation

| System | Intent classification | Draft | Routing |
|---|---|---|---|
| Trivial | Training majority | Generic handoff | Always human |
| Simple | Multinomial Naive Bayes trained with weak rule labels | Per-intent template | Sensitive/unclear requests to human |
| Retrieval | 35% NB score + 65% similarity-weighted neighbor intent vote | Matching historical clarification, otherwise handoff | Risk, follow-up, similarity, confidence, and reply-content gates |

Metrics include fixed-eight-class macro-F1, accuracy, class support/confusion, bootstrap intervals, escalation recall, coverage, unsafe-auto rate with Wilson intervals, auto-only intent accuracy, over-escalation, latency, and first-contact/follow-up slices. Undefined rates are `null`, not zero or perfect. Automated routing metrics do **not** measure whether a reply actually resolved an issue. The separate LLM/human rubric supplies reply-quality assessment.

## Source rebuild and repository map

The official source is [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter). The downloaded archive contains 2,811,774 tweets. [manifest.json](data/manifest.json) pins its SHA-256 and exact derived data. Dataset-derived material is CC BY-NC-SA 4.0 with attribution to Thought Vector; code is MIT.

To rebuild from raw, use a **fresh copy** of the code without generated `data/train.jsonl`, `dev.jsonl`, `test.jsonl`, `manifest.json`, or annotation CSVs, then:

```sh
python -m support_lab.prepare --download
```

Alternatively pass `--source /path/to/twcs.csv` or `--source /path/to/twcs.zip`. The public download may require Kaggle access if upstream permissions change; manually downloading the CSV is supported. Extraction is two streaming passes and is separate from the fast headline reproduction. Existing annotation files are protected from overwrites.

| Path | Purpose |
|---|---|
| [REPORT.md](REPORT.md) | Results, five real failure examples, limitations, next week |
| [DECISIONS.md](DECISIONS.md) | 15 non-obvious decisions and reasons |
| [data/README.md](data/README.md) | Source, split construction, annotation instructions, review design |
| `support_lab/prepare.py` | Source extraction, graph grouping, frozen sampling |
| `support_lab/agent.py` | Models, evidence retrieval, routing, extractive drafting |
| `support_lab/evaluation.py`, `metrics.py` | Leakage audit, metrics, uncertainty, reply sampling |
| `support_lab/llm.py`, `judge.py` | Live API adapter, rubric scoring, human agreement |
| `support_lab/review.py` | Resumable human annotation |
| `tests/` | Behavioral regressions and evaluation-integrity checks |

Check remaining work at any time with `python -m support_lab status`.
# Hiver-Assignment
# Hiver-Assignment
# Hiver-Assignment

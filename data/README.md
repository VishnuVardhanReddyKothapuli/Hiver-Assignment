# Data card and annotation protocol

Source: [Customer Support on Twitter, thoughtvector](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter). Attribution: Thought Vector and dataset contributors. The source lists **CC BY-NC-SA 4.0**; the included derived tweet subset remains under that license. See [license terms](https://creativecommons.org/licenses/by-nc-sa/4.0/). The code license does not relicense the data. Use is for this noncommercial take-home experiment.

`manifest.json` records the downloaded ZIP SHA-256, seed, extraction counts, and frozen split hashes. The large raw ZIP is deliberately excluded from Git. The small derived JSONL files are included so reproduction requires no download or account.

## Extraction and sampling

Two streaming passes through the original CSV collect SpotifyCares tweets and inbound messages connected in either direction. A training unit is one inbound tweet plus its earliest directly linked, non-earlier brand reply. Missing reply pairs and empty normalized messages cannot train response retrieval. Preserve anonymized source IDs; these IDs are dataset identifiers, not original Twitter status URLs.

Construct connected components from reply edges (including dangling endpoints), shared customer authors, and exact normalized customer text. Splits are assigned by a seeded hash of the component (75% train, 10% development, 15% test). Sample 6,000 training pairs. For evaluation, sample one random message per group, then uniformly select 40 development groups and 160 test groups. This is a **group-weighted sample, not a uniform sample of incoming tweets**. Repeated short utterances can connect large groups; this intentionally trades representativeness for isolation. The JSON manifest gives actual row counts, which need not match split percentages.

The frozen annotation queue contains **200 real examples: 40 development + 160 held-out test**. No intent-based stratification is used: that would require assuming labels before annotation. Rare intents may have low support; always report all eight classes and their supports. Intent suggestions for model training come from rules; they are **weak labels**, not human labels. No AI-generated annotations have been entered in `human_labels.csv`.

Only an available immediate preceding message is preserved as `prior_context`. No future brand reply is shown during intent annotation. The model uses the incoming message alone and routes known follow-ups to human review. A live caller must supply `--followup` when applicable; automatic conversation-state detection is outside this CLI's scope.

Handles and links are removed, and residual emails and long digit sequences are masked. This is limited redaction, not a proof that all personal information is removed: names and device/version information remain. Stored historical replies can have broken references after URL removal. The agent may only reuse non-specific clarifying questions, and the optional LLM receives cleaned examples.

## Human labeling instructions

Read this protocol before `python -m support_lab review --partition dev`. Complete development first. Use development only to revise rules or thresholds. Freeze those choices before labeling test. A single human reviewer is acceptable for this small experiment but must be disclosed; ask a second reviewer to independently label 30 examples if possible, and resolve disagreements in notes without discarding originals.

Read customer text and available preceding context. Identify the **primary requested action**, not just a keyword. For multiple issues, prioritize account compromise/access, a disputed financial action, then the main unresolved complaint. Use `other` when the issue is too vague or outside the taxonomy. If a different language cannot be understood by the reviewer, mark `other`, escalate, and note `language_unreviewed`; this is a limitation, not a correct semantic label.

| Intent | Include | Boundary |
|---|---|---|
| account_access | Login failure, reset, account compromise, username/email access | A mention of an account alone does not imply access trouble |
| billing | Unexpected/duplicate charge, refund, payment failure | A general plan question belongs to subscription |
| subscription | Premium activation, cancellation, Family/Student eligibility, plan changes | Charged wrong amount is billing |
| playback | Cannot play, pauses, crashes, connectivity affecting playback | Missing or unavailable titles without playback failure is library |
| library | Playlist sync, saved/downloaded content, missing catalog, library organization | A proposed feature belongs to how_to |
| how_to | Product usage question, availability of a feature, feature request | Active malfunction belongs to its operational intent |
| feedback | Praise, thanks, complaint without another actionable request | Do not label a concrete bug as feedback because the tone is angry |
| other | Ambiguous fragments, unsupported language, unrelated request | Add the specific ambiguity to the rationale |

`should_escalate=true` means a human should review **before a reply is sent** because of account-specific access/actions, disputed money, security/privacy concern, uncertain policy, unresolved repeated troubleshooting, unsupported language, or inadequate context. `false` means a safe first-step clarification, acknowledgement, or supported general instruction can reasonably be sent without account access. Do not label every subscription question risky automatically: a general usage question may be safe. Label the ideal behavior, not the implemented system's current behavior. These distinctions intentionally allow the human labels to disagree with training heuristics.

For each row record intent, `true`/`false`, a short rationale, reviewer ID, `label_source=human`, and timestamp. The terminal tool writes these only after your entry and saves each completed example. `q` exits without saving an incomplete example. CSV editing is also supported. The validator checks provenance fields but cannot verify that a human actually did the work; reviewers are responsible for truthful attestation.

## Reply quality and judge agreement

`reply_review.jsonl` freezes 20 randomly sampled test messages, each with all three systems' drafts: **60 ratings**. Ten message groups (30 drafts) are calibration; ten disjoint message groups (30 drafts) are audit. Rating order is shuffled and model names are withheld; style may still reveal the system. Read `prompts/judge.txt` and score each draft independently on grounding, relevance, safety, and helpfulness (0–2), adding a reason. No scores are pre-filled. The reference brand reply is not a gold answer and is withheld.

Run `review --kind reply --partition calibration` first. If the rubric changes, regenerate all judge scores; freeze rubric and model before the audit human review. Never tune the rubric after inspecting audit discrepancies and then call that same audit held-out. `agreement` requires all 60 real human scores and matching real judge records. It reports exact agreement, quadratic weighted kappa per dimension, pass/fail kappa, false passes, disagreements, and per-system pass rates, separately for calibration and audit. The independent audit has only **10 distinct messages**, so agreement estimates are preliminary; the 30 drafts are correlated and should not be treated as 30 independent messages.

The pass rule is grounding=2, relevance=2, safety=2, helpfulness>=1. A polite generic handoff can fail relevance. Zero automation coverage cannot establish reply safety. Missing scores stay missing.

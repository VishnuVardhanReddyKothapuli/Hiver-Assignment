# SpotifyCares: an honest support-agent evaluation

**Status:** working offline pipeline and reproducible diagnostic comparison; independent human evaluation and live LLM-judge agreement remain pending. This is an experiment with negative results, not a claim of a deployment-ready support agent.

## 1. Problem framing

For SpotifyCares, “good” means understanding the customer's primary problem, offering a relevant and supported next step, and referring financial/account/security cases to a person before replying. A plausible answer is not enough: an incorrect account action or obsolete policy claim is worse than a justified handoff. A useful system must also automate a nontrivial fraction of suitable conversations.

The source is Thought Vector's [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter), which includes 2,811,774 tweets and 43,265 SpotifyCares brand messages in the downloaded version. Direct reply reconstruction produced 41,050 eligible customer/brand pairs. We define eight intents: account access, billing, subscription, playback, library, how-to, feedback, and other.

The system retrieves historical **responses**, not verified successful resolutions. Public threads often move to DM, so successful outcomes are unobserved. The offline draft reuses only a matching non-specific clarifying question with an explicit source reply ID, or requests specialist review. We did not build account operations, Twitter posting, current product-policy lookup, multilingual modeling, or full dialogue-state inference. The optional API generator is available but its output always requires review until separately validated.

## 2. Experiment and results

Shared customer identities, observed reply edges, and exact normalized duplicates stay in one connected-component split. Use 6,000 training pairs, 40 development examples, and 160 test examples; evaluation samples distinct groups, not tweets uniformly. Train the intent models with explicit keyword heuristics. The 200-example human-label queue is blank by design, awaiting actual review. Labeling instructions and provenance fields are in [the data card](data/README.md).

The following **diagnostic** compares predictions to the same heuristic family used for training. It establishes reproducibility and detects behavioral problems; it is not independent accuracy.

| System | Intent macro-F1, fixed 8 classes | Intent agreement | Auto coverage | Heuristic-risk misses / auto replies | Escalation recall |
|---|---:|---:|---:|---:|---:|
| Majority intent + always-human | 0.067 | 0.369 | 0.0% | 0/0 — undefined | 1.000 |
| Naive Bayes + templates | 0.700 | 0.794 | 25.0% | 3/40 | 0.974 |
| Retrieval + conservative gates | 0.611 | 0.688 | 0.0% | 0/0 — undefined | 1.000 |

The retrieval model underperforms the simple classifier on this proxy and produces no automatic replies in the held-out sample. We retain these results instead of adjusting thresholds after examining test outputs. **There is no demonstrated useful automation operating point yet.** The trivial system's perfect escalation recall illustrates why coverage must accompany safety claims.

Reproduce with `python scripts/reproduce.py`. Exact class support, confusion matrices, bootstrap macro-F1 intervals, Wilson intervals, slices, latency, split audits, and all 480 test predictions are in [the artifacts](artifacts/diagnostic_results.json). Macro-F1 averages all eight declared classes, including zero-support classes. Unsafe-auto counts measure routing against the target escalation label, not full semantic reply safety.

Independent results are explicitly **not available**: human intent/routing labels 0/200, human reply ratings 0/60, and actual LLM judge calls not run. Gold evaluation refuses incomplete or non-human labels. The report should be updated with the generated gold and agreement artifacts after these steps, preserving the diagnostic table as a separate experiment.

## 3. Reply quality and judge agreement design

The frozen review pool contains three system replies for each of 20 sampled test messages. Ten messages calibrate the rubric and ten disjoint messages audit it. Draft order is shuffled and system labels hidden. Both human and LLM receive the customer message, available preceding context, candidate draft, proposed route, and supplied retrieval evidence. They do not receive the future reference reply or each other's scores.

Rate grounding, relevance, safety, and helpfulness from 0–2. A pass requires grounding=2, relevance=2, safety=2, and helpfulness>=1. A fact-free handoff can be grounded while failing relevance. A pleasant tone cannot rescue unsupported advice. The [full rubric](prompts/judge.txt) gives anchors and requires a rationale.

The harness computes exact agreement, quadratic weighted kappa per dimension, binary pass/fail kappa, mean absolute error, false passes, per-system pass rates, and all disagreements, separately for calibration and audit. Undefined kappa for constant ratings stays undefined. The held-out audit has only 10 independent messages: 30 paired drafts do not provide 30 independent observations. **No agreement statistic is claimed before a real person and an actual LLM have rated the same drafts.**

## 4. Five observed failure modes

These are qualitative inspections of real test messages and stored predictions, not hand-labelled gold examples. Dataset IDs identify the corresponding rows in `data/test.jsonl` and `artifacts/diagnostic_predictions.jsonl`.

1. **Clear operational requests collapse to `other`.** Tweet **100138**: “My playlists don't sync between the Desktop app (on Mac), my phone (Android), and the web player.” The retrieval system predicts `other` and `insufficient_context`, despite a reasonably explicit library-sync problem. Its nearest training case, 885118, mixes web-player sync and Premium. Hypothesis: weak-label noise and mixed neighbor topics dominate the actual requested action. Next: independently label training edge cases and evaluate per-intent confusion.
2. **High similarity hides missing conversation context.** Tweet **1295917**: “iPhone 7, iOS 11, version 8.4.22.515.” The nearest case has cosine **1.0** despite describing different device/version numbers. The tokenizer drops numbers and sees the same remaining words. The agent escalates, but a similarity-only system could confidently reuse the wrong troubleshooting. Next: retain device/version entities and model dialogue context; audit near-duplicate representations separately from exact-text split checks.
3. **Conservative thresholds discard useful clarification opportunities.** Tweet **1080685** reports intermittent buzzing during music. The correct broad intent is predicted as playback, but the best match scores only **0.310**, so the result is a generic handoff. A device/version clarification exists in retrieved reply 950427. Hypothesis: lexical retrieval struggles with varied descriptions, while the fixed .55 gate suppresses coverage. Next: compare semantic retrieval and tune only on human-labeled development examples, reporting a risk–coverage curve.
4. **English tokenization finds superficial multilingual neighbors.** Tweet **1514282** asks in Spanish for BTR albums. The retrieved case 2776164 begins “Hola” but asks about Spotify in India; its reply discusses country launches. The agent routes to a human, yet retrieves misleading evidence and labels the request `other`. Next: add a language-aware routing feature and evaluate Spanish content requests separately rather than counting them as correctly understood abstentions.
5. **Entity-specific catalog questions invite obsolete answers.** Tweet **1757050** asks “where’s reputation at ??” Its nearest case has cosine **1.0** and a historical reply promising availability when the album reaches Spotify. That is not a reliable current answer, and the system predicts `how_to` before escalating for insufficient context. Hypothesis: shorthand entity questions need catalog context; historical wording can look relevant while being temporally invalid. Next: separate catalog intent and ground current availability in an authorized current source.

## 5. What is misleading about my headline number?

**“100% escalation recall” can mean doing nothing automatically.** Here the retrieval system has zero coverage, so auto-handling precision is undefined. **“79.4% intent agreement” is circular:** the simple model learns from the same rules that generate its diagnostic targets. Neither number proves trustworthiness.

The dataset includes conversations with brand replies, underrepresenting unanswered customers. One example per merged group changes the sampling distribution; short repeated utterances can connect large groups. The split is random by group, not chronological: retrieved historical evidence can postdate an evaluated message. Near-duplicate semantics and tokenization collisions survive exact-duplicate controls. Current product behavior is not represented by this old snapshot.

The test has only 160 examples and some intents have little support. A bootstrap interval accounts for sampling variability under the chosen sample, not labeling bias, brand shift, temporal shift, or uncertain resolution. The retrieval score is not a calibrated probability. A future judge could share model biases or reward generic politeness; agreement with one human would measure consistency with that reviewer, not universal correctness. Human review, adjudication, and real outcome data remain necessary.

## 6. One more week

Days 1–2: finish 200 intent/routing labels and 60 reply ratings, obtain an independent second reviewer for a subset, and document disagreements. Freeze the rubric before held-out judge comparison. Days 3–4: improve weak-label training data, preserve entities and dialogue context, compare semantic retrieval to the existing lexical baseline, and choose operating thresholds using development only. Days 5–6: evaluate once on frozen test, examine risk–coverage and language/intent slices, and run blinded human–judge agreement with an explicit model snapshot. Day 7: build a new temporal holdout and a small shadow-mode pilot with humans approving every reply; measure whether the suggestions reduce review time and actually help resolve issues. Do not enable automatic posting on this evidence alone.

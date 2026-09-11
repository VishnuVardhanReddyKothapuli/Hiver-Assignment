# Decision log

1. **SpotifyCares:** enough conversations (43,265 brand tweets) for a focused experiment, with account, financial, and technical requests to test routing. Inspect source examples before defining the taxonomy.
2. **Eight operational intents:** use the customer's requested action; reserve `other` for genuine uncertainty. Avoid importing Banking77 labels that do not describe music support.
3. **Standard-library Python:** no model downloads or dependency resolver; a reviewer can reproduce the shipped experiment quickly on an ordinary CPU.
4. **Extractive retrieval as the default AI:** TF-IDF neighbors and a weak-supervised classifier make evidence inspectable and keep offline runs deterministic. An optional LLM drafts replies but is not needed to reproduce diagnostics.
5. **Weak supervision for training only:** no budget for manually labeling 6,000 training rows. Human evaluation must be independent; training-rule agreement is labeled circular and not advertised as accuracy.
6. **Group before split:** merge shared authors, reply edges, and normalized duplicates. Prevent the same customer's follow-ups and copied requests appearing in train and test, at the cost of altered sampling weights.
7. **Group-random rather than temporal split:** supports this small historical snapshot experiment. It does not establish forward-time generalization; random splits can retrieve a response written later than the evaluated message.
8. **40 development and 160 test examples:** fit within the requested 150–250 hand-label budget while reserving a development set. The actual hand-label work remains a human task.
9. **Do not infer resolution from a reply:** a DM handoff is not proof the problem was solved. Exclude account operations and obsolete product claims from automatic replies.
10. **Conservative fixed routing threshold:** .55 cosine and .60 mixed intent score are pre-human-evaluation engineering defaults, not calibrated probabilities. Preserve the observed zero-coverage result rather than optimizing on test results.
11. **No unconditional reuse of nearest replies:** allow only sufficiently similar, intent-matched, non-specific clarifying questions. Otherwise return a specialist-review draft with an explicit machine-readable reason.
12. **Report coverage beside risk:** always-escalate attains perfect escalation recall. Undefined auto precision stays null; Wilson intervals expose small denominators.
13. **Blind paired reply review:** sample the same 20 messages across all systems, split by message into calibration and audit, and hide system labels. A separate rubric-based judge never sees human scores.
14. **No invented completeness:** provenance validators reject machine labels as gold; missing API/human ratings do not become mock quality scores. API outputs are cached by exact prompt, payload, model, and schema.
15. **Optional LLM output requires review:** valid JSON and valid citation IDs do not establish semantic grounding. Do not let an unvalidated rewrite inherit the extractive system's auto-handling decision.

"""Weak-supervised intent models, historical retrieval, and explicit routing."""
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

from .common import INTENTS, clean, tokens

RULES = (
    ("account_access", r"\b(log ?in|sign ?in|password|hacked|hack|stolen|username|email address|locked out)\b"),
    ("billing", r"\b(charged?|charging|refund|payment|paying|paid|money|credit card|debit|bill|billing)\b"),
    ("subscription", r"\b(premium|subscription|subscribe|cancel|family|student|trial|discount|upgrade)\b"),
    ("playback", r"\b(play|playing|playback|pause|pausing|skip|skipping|crash|crashes|crashing|buffer|buffering|offline|connect|connection|stops|stopping|sound)\b"),
    ("library", r"\b(playlist|playlists|library|saved|deleted|disappeared|missing|download|downloads|downloaded|album|albums|song|songs)\b"),
    ("how_to", r"\b(how|where|can i|possible|change|settings|feature)\b"),
    ("feedback", r"\b(thank|thanks|love|awesome|great|sucks|terrible|hate|amazing)\b"),
)
HANDOFF = "I'd like a support specialist to review this before suggesting a next step. Please don't post passwords or payment details here."
TEMPLATES = {
    "playback": "Sorry playback isn't working as expected. Which device, operating system, and Spotify version are you using?",
    "library": "Could you describe which songs or playlists are affected and which device you're using?",
    "how_to": "Could you tell us which feature you're trying to use and on which device?",
    "feedback": "Thanks for sharing your feedback with us.",
}
RISK = re.compile(r"\b(hack\w*|stolen|fraud\w*|suicid\w*|kill\w*|lawyer|lawsuit|police|charge\w*|refund\w*|password|payment|billing|email|account|cancel\w*|money)\b", re.I)
INJECTION = re.compile(r"ignore.{0,30}(instruction|previous|system)|system prompt|reveal.{0,30}(secret|key)|pretend.{0,30}(agent|system)", re.I)
UNSAFE_REPLY = re.compile(r"\b(refund\w*|charged?|payment|password|email|account|dm|direct message|private message|fixed|resolved|cancelled|canceled|credited|maintenance|currently|tomorrow|today|yesterday|backstage|pass.{0,8}(on|team)|forward\w*|escalat\w*|reinstall\w*|reset\w*|delete\w*)\b|[£$€]|\d", re.I)

def weak_intent(text):
    """A training heuristic, never a human or independent evaluation label."""
    t = clean(text).lower()
    for intent, pattern in RULES:
        if re.search(pattern, t):
            return intent
    return "other"

def risk_reason(text, intent):
    if INJECTION.search(text):
        return "instruction_injection"
    if RISK.search(text) or intent in {"account_access", "billing", "subscription"}:
        return "account_or_sensitive_action"
    if intent == "other" or len(tokens(text)) < 3:
        return "insufficient_context"
    return None

def weak_escalate(text):
    return bool(risk_reason(text, weak_intent(text)))

class NaiveBayes:
    def __init__(self, rows):
        self.counts = {c: Counter() for c in INTENTS}
        self.docs = Counter()
        vocab = set()
        for row in rows:
            label = weak_intent(row["text"])
            words = tokens(row["text"])
            self.counts[label].update(words)
            self.docs[label] += 1
            vocab.update(words)
        self.total = len(rows)
        self.vocab = vocab
        self.denoms = {c: sum(self.counts[c].values()) + len(vocab) for c in INTENTS}

    def probabilities(self, text):
        scores = {}
        for c in INTENTS:
            scores[c] = math.log((self.docs[c] + 1) / (self.total + len(INTENTS)))
            for word in tokens(text):
                if word in self.vocab:
                    scores[c] += math.log((self.counts[c][word] + 1) / self.denoms[c])
        m = max(scores.values())
        values = {c: math.exp(s - m) for c, s in scores.items()}
        total = sum(values.values())
        return {c: v / total for c, v in values.items()}

class Retriever:
    """L2-normalized TF-IDF, inverted index, deterministic cosine ranking."""
    def __init__(self, rows):
        self.rows = rows
        counts = [Counter(tokens(r["text"])) for r in rows]
        df = Counter(w for c in counts for w in c)
        self.idf = {w: math.log((1 + len(rows)) / (1 + n)) + 1 for w, n in df.items()}
        self.index = defaultdict(list)
        for i, counts_i in enumerate(counts):
            for w, score in self.vector(counts_i).items():
                self.index[w].append((i, score))

    def vector(self, counts):
        weighted = {w: (1 + math.log(n)) * self.idf[w] for w, n in counts.items() if w in self.idf}
        norm = math.sqrt(sum(v * v for v in weighted.values())) or 1
        return {w: v / norm for w, v in weighted.items()}

    def search(self, text, k=5):
        scores = defaultdict(float)
        for word, weight in self.vector(Counter(tokens(text))).items():
            for i, score in self.index[word]:
                scores[i] += score * weight
        ranked = sorted(scores, key=lambda i: (-scores[i], self.rows[i]["id"]))[:k]
        return [(self.rows[i], scores[i]) for i in ranked]

@dataclass
class Prediction:
    intent: str
    score: float
    reply: str
    should_escalate: bool
    reason: str
    evidence: list
    source_reply_id: str | None
    backend: str

    def as_dict(self):
        return asdict(self)

class Agent:
    def __init__(self, train, threshold=0.55):
        if not train:
            raise ValueError("Training corpus is empty")
        self.nb = NaiveBayes(train)
        self.retriever = Retriever(train)
        self.majority = self.nb.docs.most_common(1)[0][0]
        self.threshold = threshold

    def predict(self, text, system="retrieval", is_followup=False):
        if system == "trivial":
            return Prediction(self.majority, 0.0, HANDOFF, True, "always_escalate", [], None, system)
        probs = self.nb.probabilities(text)
        if system == "simple":
            intent = max(probs, key=probs.get)
            reason = risk_reason(text, intent)
            return Prediction(intent, probs[intent], HANDOFF if reason else TEMPLATES.get(intent, HANDOFF),
                              bool(reason), reason or "simple_template", [], None, system)
        if system != "retrieval":
            raise ValueError(f"Unknown system: {system}")
        hits = self.retriever.search(text)
        votes = Counter()
        for r, s in hits:
            votes[weak_intent(r["text"])] += s ** 2
        total = sum(votes.values()) or 1
        mixed = {c: .35 * probs[c] + .65 * votes[c] / total for c in INTENTS}
        intent = max(mixed, key=mixed.get)
        evidence = [{"id": r["id"], "reply_id": r["reply_id"], "text": r["text"], "reply": r["reply"],
                     "similarity": round(s, 6)} for r, s in hits]
        reason = risk_reason(text, intent)
        if not reason and is_followup:
            reason = "followup_requires_context_review"
        if not reason and (not hits or hits[0][1] < self.threshold or mixed[intent] < .6):
            reason = "weak_or_conflicting_evidence"
        candidate, source = None, None
        if not reason:
            for r, sim in hits:
                draft = re.sub(r"\s*/[A-Z]{1,4}\b", "", r["reply"]).strip()
                question = re.search(r"\b(?:what (?:device|happens)|which (?:device|operating system)|can you (?:let us know|tell us|describe)|could you (?:let us know|tell us|describe))\b[^?]{5,200}\?", draft, re.I)
                # Only extract matching non-specific clarifying questions. Historical
                # replies are not verified outcomes and may contain obsolete advice.
                if (sim >= self.threshold and weak_intent(r["text"]) == intent and question
                        and not UNSAFE_REPLY.search(draft) and len(draft) >= 25):
                    candidate, source = question.group(0), r["reply_id"]
                    break
            if candidate is None:
                reason = "no_reusable_historical_reply"
        return Prediction(intent, round(mixed[intent], 6), candidate or HANDOFF, bool(reason),
                          reason or "supported_low_risk_clarification", evidence, source, system)

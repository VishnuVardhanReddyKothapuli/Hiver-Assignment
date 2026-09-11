"""Explicit denominators, fixed taxonomy, and uncertainty for small samples."""
import math
import random
from collections import Counter
from .common import INTENTS

def ratio(a, b):
    return a / b if b else None

def wilson(successes, n, z=1.96):
    if not n:
        return None
    p = successes / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0, center-half), min(1, center+half)]

def classification(y, pred):
    details = {}
    for c in INTENTS:
        tp = sum(a == b == c for a, b in zip(y, pred))
        fp = sum(a != c and b == c for a, b in zip(y, pred))
        fn = sum(a == c and b != c for a, b in zip(y, pred))
        details[c] = {"support": y.count(c), "precision": ratio(tp, tp+fp), "recall": ratio(tp, tp+fn),
                      "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}
    correct = sum(a == b for a, b in zip(y, pred))
    return {"n": len(y), "accuracy": ratio(correct, len(y)), "accuracy_wilson95": wilson(correct, len(y)),
            "macro_f1_fixed_8": sum(v["f1"] for v in details.values()) / len(INTENTS), "per_intent": details,
            "confusion": {c: dict(Counter(b for a, b in zip(y, pred) if a == c)) for c in INTENTS}}

def evaluate(y, escalate, predictions):
    intents = [p["intent"] for p in predictions]
    result = classification(y, intents)
    auto = [i for i, p in enumerate(predictions) if not p["should_escalate"]]
    required = sum(escalate)
    missed = sum(escalate[i] for i in auto)
    result.update({"auto_count": len(auto), "coverage": ratio(len(auto), len(y)),
                   "required_escalations": required, "missed_escalations": missed,
                   "escalation_recall": ratio(required-missed, required),
                   "unsafe_auto_rate": ratio(missed, len(auto)), "unsafe_auto_wilson95": wilson(missed, len(auto)),
                   "auto_intent_accuracy": ratio(sum(y[i] == intents[i] for i in auto), len(auto)),
                   "over_escalation_rate": ratio(sum(not e and p["should_escalate"] for e, p in zip(escalate, predictions)), len(y)-required)})
    rng = random.Random(41)
    boots = []
    for _ in range(500):
        idx = [rng.randrange(len(y)) for _ in y]
        boots.append(classification([y[i] for i in idx], [intents[i] for i in idx])["macro_f1_fixed_8"])
    result["macro_f1_bootstrap95"] = [sorted(boots)[12], sorted(boots)[487]] if boots else None
    return result

def kappa(a, b, weighted=False):
    if not a:
        return None
    levels = sorted(set(a+b))
    if len(levels) < 2:
        return None  # Undefined, not perfect agreement.
    ca, cb = Counter(a), Counter(b)
    cost = (lambda x,y: (x-y)**2) if weighted else (lambda x,y: float(x != y))
    observed = sum(cost(x,y) for x,y in zip(a,b)) / len(a)
    expected = sum(ca[x]*cb[y]*cost(x,y) for x in levels for y in levels) / len(a)**2
    return 1 - observed/expected if expected else None

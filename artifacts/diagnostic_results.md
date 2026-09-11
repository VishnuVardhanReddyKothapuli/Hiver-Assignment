# Diagnostic: agreement with training heuristics

**Not independent accuracy. The target labels come from the same heuristics used to train the models.**

| System | N | Intent macro-F1 | Intent accuracy | Auto coverage | Unsafe auto / auto | Escalation recall |
|---|---:|---:|---:|---:|---:|---:|
| trivial | 160 | 0.067 | 0.369 | 0.000 | 0/0 | 1.000 |
| simple | 160 | 0.700 | 0.794 | 0.250 | 3/40 | 0.974 |
| retrieval | 160 | 0.611 | 0.688 | 0.000 | 0/0 | 1.000 |

Intervals, class support, confusion matrices, follow-up slices, and routing reasons are in the JSON artifact.
Zero auto-handled examples means safety precision is undefined, not 100%. Human reply quality and judge agreement require separate ratings.

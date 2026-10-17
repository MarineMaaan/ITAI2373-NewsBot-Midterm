### Findings from this execution
- Retained **1438** unique substantial articles; split sizes **[1006, 216, 216]**.
- Training CV selected **Linear SVM**. Reserved-test accuracy **0.981**, macro-F1 **0.981**.
- **business** has the largest mean NUM proportion; structured amounts can help an analyst find quantity-heavy coverage.
- Mean heuristic tone ranges from **politics (0.054)** to **entertainment (0.590)**; this is lexical tone, not verified public opinion.
- Grammar ablation macro-F1: **0.971 → 0.929**; difference CI **[-0.074, -0.014]**. It suggests a decline on this split.
- Custom NER exact-span test F1: **0.625**, on 12 synthetic held-out sentences only.
- Human-review policy accepted **75.0%** of held-out articles. Editorial time saved must be measured in a real workflow, not inferred from classifier accuracy.

### Recommendations
Use this as an assisted routing prototype. Review low-margin cases and audit errors by category.
Check entities against original articles; co-occurrence is only a lead for investigation.
For game reviews, collect licensed review text, define relevant labels (bugs, performance,
gameplay, value), annotate sentiment toward the game rather than whole-document tone,
and split by game/source/time to measure generalization. Begin with a small human-reviewed pilot.

### Limits and next experiments
BBC is a historical single-source benchmark. Its labels omit health, its CSV lacks
publication timestamps, and its casing limits NER. Exact deduplication does not remove
all near duplicates. Repeat grouped and temporal tests when metadata is available.
Sentiment/emotions and semantic roles need gold annotations. Bilingual probes and
custom NER data are synthetic and small. No production, fairness or real-time claim is made.

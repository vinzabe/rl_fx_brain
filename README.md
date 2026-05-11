# model-extraction-detector (mxd)

Server-side detector for **model-extraction / model-stealing / distillation**
attacks against hosted LLM and ML APIs.

Adversaries try to clone your model by sending a flood of carefully chosen
queries and harvesting the responses. The "obvious" attacker (1M queries/min)
is easy to spot; the **interesting** ones are slow, templated, or focused on
the decision boundary. `mxd` looks for the lexical, volumetric, and
embedding-space signatures these attacks leave behind.

## Signals

Per-actor rolling window of `QueryEvent`s, scored on **20 features**:

| Family    | Features                                                                          |
|-----------|-----------------------------------------------------------------------------------|
| Volume    | `count`, `duration_seconds`, `rate_per_second`, `burst_score`                     |
| Lexical   | `mean/stdev/cov query_len`, `mean_word_count`, `unique_words_ratio`, char entropy |
| Timing    | `mean_iat`, `stdev_iat`, `p10_iat`                                                |
| Repetition| `near_dup_ratio`, `templated_query_ratio`                                          |
| Probing   | `boundary_probe_ratio` (logits / probabilities / embeddings keywords)             |
| Diversity | `embedding_diversity` (mean pairwise cosine distance, sampled)                    |
| Responses | `response_token_mean`, `response_token_std`                                       |

A small set of conservative rule thresholds (each weighted) flags suspicious
actors with an explainable trigger list. An optional **IsolationForest**
anomaly model can be fit on a benign baseline and combined with the rule
score for additional sensitivity.

An **LLM incident analyst** turns the trigger list and feature dict into a
narrative report (`headline`, `attack_category`, `summary`, `techniques`,
`recommended_actions`). Hallucination guards drop any cited trigger name not
present in the input feature space.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Scan a JSONL of QueryEvents
python -m mxd.cli scan events.jsonl -o report.json

# Same + LLM verdict per suspicious actor
python -m mxd.cli scan events.jsonl --analyse -o report.json

# CI mode (exit 1 if any actor flagged)
python -m mxd.cli scan events.jsonl --fail-on-suspicious
```

Programmatic:

```python
from mxd import (
    DetectionPipeline, ExtractionDetector,
    LLMExtractionAnalyst, generate_attacker_corpus,
)

events = generate_attacker_corpus(archetype="templated", n=300)
pipe = DetectionPipeline()
verdicts = pipe.evaluate_batch(events)

for actor_id, v in verdicts.items():
    if v.is_suspicious:
        report = LLMExtractionAnalyst().analyse(v.to_dict())
        print(actor_id, v.severity, report.attack_category, report.headline)
```

### JSONL event schema

One JSON object per line:

```json
{
  "actor_id": "user-42",
  "timestamp": 1700000123.4,
  "query_text": "Classify entity number 17 as one of: spam, ham, unknown.",
  "response_text": "spam",
  "query_token_count": 12,
  "response_token_count": 1,
  "embedding": [0.123, -0.456, ...],
  "endpoint": "chat",
  "ip": "203.0.113.4"
}
```

`embedding` is optional but enables the `embedding_diversity` signal.

### Baseline anomaly model

```python
from mxd import ActorWindow, ExtractionDetector, generate_benign_corpus

benign = [
    ActorWindow(actor_id=f"u_{i}", window_seconds=3600)
    for i in range(20)
]
# (in real use: populate windows from your historical query log)

d = ExtractionDetector()
d.fit_baseline(benign)        # IsolationForest on benign windows
d.save("baseline.joblib")

# later:
d2 = ExtractionDetector()
d2.load("baseline.joblib")    # schema-versioned
```

## Why this exists

Model extraction is the silent supply-chain attack against your hosted model.
Today's API gateways watch QPS and bytes; they don't watch lexical
templating, embedding-space coverage, or "give me the logits" probing.
`mxd` is the rule pack and feature extractor I wanted to drop in front of my
own inference endpoint.

## Layout

```
mxd/
  events.py     QueryEvent + ActorWindow (rolling, evicting)
  features.py   20-feature vector + helpers (entropy, jaccard, cosine)
  detector.py   Rule thresholds + IsolationForest baseline
  synth.py      Synthetic attacker / benign corpora for tests + demos
  pipeline.py   Streaming + batch evaluation per actor
  analyst.py    LLM incident report (hallucination-guarded)
  cli.py        `python -m mxd.cli scan ...`
tests/
  test_mxd.py   57 tests (52 mocked + 5 live LLM)
```

## Tests

```bash
pytest tests/                  # 52 mocked
LLM_LIVE=1 pytest tests/       # +5 live LLM smoke tests
```

## Security

See `SECURITY.md`. **g@abejar.net** for disclosures.

## License

MIT — see `LICENSE`.

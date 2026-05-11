"""Server-side model-extraction / model-stealing detector.

Detects query patterns indicating an adversary is attempting to reconstruct,
distil or extract a hosted ML/LLM model:

- Volume anomalies (rate, burst, total per actor)
- Embedding-space coverage (extraction attacks need diverse queries)
- Synthetic-looking queries (uniform length, low entropy, lexical templating)
- Adversarial / boundary probing (margin-mining, jailbreak templates)
- Repeated near-duplicate queries
- Diversity ratio (unique-tokens / total-tokens)
"""
from .events import QueryEvent, ActorWindow
from .features import FeatureExtractor
from .detector import ExtractionDetector, Verdict
from .pipeline import DetectionPipeline
from .synth import generate_attacker_corpus, generate_benign_corpus
from .analyst import LLMExtractionAnalyst, IncidentReport

__all__ = [
    "QueryEvent",
    "ActorWindow",
    "FeatureExtractor",
    "ExtractionDetector",
    "Verdict",
    "DetectionPipeline",
    "generate_attacker_corpus",
    "generate_benign_corpus",
    "LLMExtractionAnalyst",
    "IncidentReport",
]

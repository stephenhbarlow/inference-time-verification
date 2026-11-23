"""Finding-level inference-time verification for set-valued extraction."""

from .candidates import Candidate, build_candidate_pool
from .metrics import (
    calculate_document_metrics,
    calculate_extraction_metrics,
    match_findings,
)
from .generation import FindingGenerator, FindingVerifier, GenerationConfig
from .pipeline import ITVResult, run_itv
from .text import normalise_sentence

__all__ = [
    "Candidate",
    "FindingGenerator",
    "FindingVerifier",
    "GenerationConfig",
    "ITVResult",
    "build_candidate_pool",
    "calculate_document_metrics",
    "calculate_extraction_metrics",
    "match_findings",
    "normalise_sentence",
    "run_itv",
]


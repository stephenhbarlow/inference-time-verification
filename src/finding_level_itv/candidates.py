"""Construction of a normalised finding-level candidate set."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable, Sequence

from .text import normalise_sentence


@dataclass
class Candidate:
    """One unique normalised finding and its generation occurrences."""

    text: str
    normalized: str
    first_seen_generation: int
    occurrences: list[int] = field(default_factory=list)

    @property
    def votes(self) -> int:
        """Number of generations in which the candidate appeared."""
        return len(self.occurrences)


def build_candidate_pool(generations: Sequence[Iterable[str]]) -> list[Candidate]:
    """Union stochastic generations into an ordered, de-duplicated candidate set.

    A candidate receives at most one occurrence from each generation. The first
    observed surface form is retained for contextual verification, while equality
    and voting use the function `normalise_sentence`.
    """
    candidates: dict[str, Candidate] = {}
    for generation_index, findings in enumerate(generations, start=1):
        seen_this_generation: set[str] = set()
        for finding in findings:
            normalized = normalise_sentence(finding)
            if not normalized or normalized in seen_this_generation:
                continue
            seen_this_generation.add(normalized)
            candidate = candidates.setdefault(
                normalized,
                Candidate(
                    text=finding,
                    normalized=normalized,
                    first_seen_generation=generation_index,
                ),
            )
            candidate.occurrences.append(generation_index)
    return list(candidates.values())


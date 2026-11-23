"""Model-agnostic implementation of finding-level ITV."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .candidates import Candidate, build_candidate_pool


Generator = Callable[[str], Sequence[str]]
Verifier = Callable[[str, str], bool]


@dataclass
class ITVResult:
    """Generated samples, candidate decisions and final extracted findings."""

    generations: list[list[str]]
    candidates: list[Candidate]
    decisions: list[bool]

    @property
    def findings(self) -> list[str]:
        return [
            candidate.text
            for candidate, accepted in zip(self.candidates, self.decisions)
            if accepted
        ]

    @property
    def report_label(self) -> int:
        """A report is positive when any finding is accepted."""
        return int(bool(self.findings))


def run_itv(
    report: str,
    generator: Generator,
    verifier: Verifier,
    *,
    num_generations: int = 10,
) -> ITVResult:
    """Generate a candidate union and verify each unique finding independently."""
    if num_generations < 1:
        raise ValueError("num_generations must be positive")
    generations = [list(generator(report)) for _ in range(num_generations)]
    candidates = build_candidate_pool(generations)
    decisions = [verifier(candidate.text, report) for candidate in candidates]
    return ITVResult(generations, candidates, decisions)


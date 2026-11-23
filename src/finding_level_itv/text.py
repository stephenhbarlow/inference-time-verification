"""Finding normalization and report sentence processing."""

from __future__ import annotations

from collections.abc import Callable


def normalise_sentence(sentence: str) -> str:
    """Normalize a finding for candidate de-duplication and exact matching.

    The spelling is retained from the research implementation. Leading numbering
    and punctuation are removed, but internal punctuation is left unchanged.
    """
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string")
    return (
        sentence.lower()
        .strip()
        .lstrip("0123456789,;:.)(- ")
        .replace("\n", " ")
    )


# American spelling for users who expect it; both names have identical behavior.
normalize_sentence = normalise_sentence


def repair_numbered_fragments(sentences: list[str]) -> list[str]:
    """Attach short numbering fragments such as ``"1."`` to the next sentence.

    Consecutive fragments are accumulated without mutating the input list.
    """
    repaired: list[str] = []
    pending = ""
    for sentence in sentences:
        if len(sentence) <= 2:
            pending = f"{pending} {sentence}".strip()
            continue
        if pending:
            sentence = f"{pending} {sentence}"
            pending = ""
        repaired.append(sentence)
    # A trailing numbering fragment contains no finding text, so it is dropped.
    return repaired


def split_report(
    report: str,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> list[str]:
    """Split a report using NLTK unless an explicit tokenizer is supplied."""
    if tokenizer is None:
        try:
            from nltk.tokenize import sent_tokenize
        except ImportError as error:  # pragma: no cover - dependency guidance
            raise ImportError(
                "NLTK is required for default report splitting; alternatively "
                "pass a sentence tokenizer explicitly"
            ) from error
        tokenizer = sent_tokenize
    return repair_numbered_fragments(list(tokenizer(report)))


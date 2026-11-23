"""Prompt construction and response parsing for finding verification.

The verifier accepts a candidate when the case-sensitive substring ``"yes"``
appears in the first five stripped decoded characters. The stricter helpers can
be used to measure instruction-following without changing that decision rule.
"""

from __future__ import annotations

import re


def sharegpt_verifier_messages(
    candidate: str, report: str
) -> list[dict[str, str]]:
    """Build the ShareGPT message sequence used for verifier adaptation.

    The contextual preamble is represented as a ``gpt`` message, followed by the
    report/candidate instruction as a ``human`` message. Use the same sequence for
    training and inference to avoid a serialization shift.
    """
    return [
        {
            "from": "gpt",
            "value": "The following text is a PET-CT report for lung cancer:",
        },
        {
            "from": "human",
            "value": (
                f"REPORT: {report}\n\n"
                "INSTRUCTION: In the context of the report would the following "
                "sentence be related to an actionable incidental finding?\n\n"
                f'"{candidate}"\n\n'
                "Answer 'yes' or 'no' only."
            ),
        },
    ]


def llama3_verifier_prompt(candidate: str, report: str) -> str:
    """Serialize a verifier prompt with Llama 3 header and turn tokens."""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        "The following text is a PET-CT report for lung cancer:"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        f"REPORT: {report}\n\n"
        "INSTRUCTION: In the context of the report would the following sentence "
        "be related to an actionable incidental finding?\n\n"
        f'"{candidate}"\n\n'
        "Answer 'yes' or 'no' only."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )


def verifier_accepts_response(decoded: str) -> bool:
    """Accept when lowercase ``yes`` occurs in the first five stripped characters."""
    return "yes" in decoded.strip()[:5]


def parse_leading_yes_no(decoded: str) -> str:
    """Classify a leading yes/no token for format diagnostics, case-insensitively.

    Explanatory prefixes such as ``"Answer: yes"`` are deliberately invalid:
    the model was explicitly instructed to begin with yes or no only.
    """
    match = re.match(r"^(yes|no)\b", decoded.strip(), flags=re.IGNORECASE)
    return match.group(1).lower() if match else "invalid"


def is_exact_yes_no_response(decoded: str) -> bool:
    """Return whether the entire stripped response is exactly yes or no."""
    return decoded.strip().lower() in {"yes", "no"}

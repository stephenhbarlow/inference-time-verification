from finding_level_itv.prompts import (
    llama3_verifier_prompt,
    sharegpt_verifier_messages,
    verifier_accepts_response,
)


def test_training_prompt_contains_report_and_candidate_in_separate_fields():
    messages = sharegpt_verifier_messages("candidate", "synthetic report")
    assert len(messages) == 2
    assert messages[0]["from"] == "gpt"
    assert messages[1]["from"] == "human"
    assert "synthetic report" in messages[1]["value"]
    assert '"candidate"' in messages[1]["value"]


def test_llama3_prompt_contains_expected_turn_boundaries():
    prompt = llama3_verifier_prompt("candidate", "synthetic report")
    assert prompt.startswith("<|begin_of_text|>")
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>")


def test_verifier_decision_is_case_sensitive_and_limited_to_response_start():
    assert verifier_accepts_response(" yes")
    assert not verifier_accepts_response("Yes")
    assert not verifier_accepts_response("Answer: yes")


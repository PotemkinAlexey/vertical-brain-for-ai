from vertical_brain.llm.mock_llm import MockLLM


def test_mock_llm_answers_only_from_context():
    answer = MockLLM().answer_from_context(
        "How does Delta schema evolution work?",
        [
            "[WORK/DataArt/Databricks][silver] Delta uses mergeSchema for schema evolution.",
            "[TRADING][silver] Trading context should not be passed here.",
        ],
    )

    assert "Based only on locked context:" in answer
    assert "Delta uses mergeSchema" in answer
    assert "Trading context" not in answer


def test_mock_llm_reports_missing_context():
    answer = MockLLM().answer_from_context("How does Delta schema evolution work?", [])

    assert answer == "I do not have allowed context to answer this question yet."

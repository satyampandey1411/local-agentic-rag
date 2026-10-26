"""
Number-word handling in the answer grader (``alias_matches`` / ``normalize_number_words``).

The rule under test: English number words and "<n> million" scales are rewritten to digits
only when unambiguous (phrase value >= 11, well-formed, joined by a space or hyphen), as a
second chance after the plain comparison failed.  One-to-ten stay words, so "one of the"
never satisfies the key "1".
"""

from __future__ import annotations

import pytest

from local_agentic_rag.eval.metrics import (
    alias_matches,
    fact_recall,
    is_numeric_alias,
    normalize_number_words,
    normalize_text,
    question_metrics,
    token_f1,
)


class TestRewrite:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("seventeen", "17"),
            ("twenty-one", "21"),
            ("twenty one", "21"),
            ("eleven", "11"),
            ("ninety-nine", "99"),
            ("one hundred", "100"),
            ("one hundred and five", "105"),
            ("one hundred five", "105"),
            ("two hundred and fifty", "250"),
            ("two thousand", "2000"),
            ("two thousand and five", "2005"),
            ("one thousand two hundred thirty-four", "1234"),
            ("three hundred thousand", "300000"),
            ("twenty million", "20000000"),
            ("the budget was twenty-one million dollars", "the budget was 21000000 dollars"),
            ("1.2 million", "1200000"),
            ("3 billion", "3000000000"),
            ("1,200 thousand", "1200000"),
            ("2.5 thousand", "2500"),
            ("twenty percent", "20%"),
        ],
    )
    def test_converted(self, text: str, expected: str) -> None:
        assert normalize_number_words(normalize_text(text)) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "one",
            "one of the three",
            "two",
            "seven days",
            "ten",  # 10 is below the threshold
            "four two",  # two units are not 42
            "hundred",
            "thousand",
            "a million",  # no digits and no number word before the scale
            "seventeenth",
            "twenty-first",
            "one hundredth",
            "twenty second delay",
            "thirty-third",
            "plain words only",
            "2.5 m",
            "5 crore",
        ],
    )
    def test_left_alone(self, text: str) -> None:
        assert normalize_number_words(normalize_text(text)) == normalize_text(text)

    def test_comma_separated_numbers_are_not_merged(self) -> None:
        # "one thousand, two hundred" is two numbers; guessing 1,200 would be unsafe.
        assert normalize_number_words(normalize_text("one thousand, two hundred")) == "1000, 200"

    def test_dangling_and_is_kept_as_a_word(self) -> None:
        assert normalize_number_words("one hundred and the rest") == "100 and the rest"

    def test_year_style_is_not_merged(self) -> None:
        assert normalize_number_words("nineteen eighty-four") == "19 84"

    def test_idempotent(self) -> None:
        once = normalize_number_words(normalize_text("one hundred and five and 1.2 million"))
        assert normalize_number_words(once) == once


class TestAliasMatching:
    @pytest.mark.parametrize(
        ("answer", "alias"),
        [
            ("There were seventeen incidents.", "17"),
            ("There were 17 incidents.", "seventeen"),
            ("It has twenty-one members", "21"),
            ("It has 21 members", "twenty-one"),
            ("It has Twenty One members", "21"),
            ("The fee is one hundred and five dollars", "105"),
            ("The fee is 105 dollars", "one hundred and five dollars"),
            ("About two thousand people", "2000"),
            ("About 2,000 people", "two thousand"),
            ("Revenue was $1.2 million", "1,200,000"),
            ("Revenue was 1,200,000 dollars", "1.2 million"),
            ("Revenue was 1.2 million", "1200000"),
            ("Population: three million", "3,000,000"),
            ("Population: 3 million", "three million"),
            ("A twenty percent rise", "20%"),
            ("A 20% rise", "twenty percent"),
            # Devanagari digits still compare equal, and combine with number words.
            ("There were seventeen incidents.", "१७"),
            ("कुल १७ घटनाएँ", "seventeen"),
            ("कुल १७ घटनाएँ", "17"),
        ],
    )
    def test_matches(self, answer: str, alias: str) -> None:
        assert alias_matches(answer, alias)

    @pytest.mark.parametrize(
        ("answer", "alias"),
        [
            # Small number words are prose, not numbers.
            ("one of the three options", "1"),
            ("one of the three options", "3"),
            ("Only seven were found", "7"),
            ("ten days", "10"),
            ("one", "1"),
            ("1", "one"),
            # Boundaries are preserved.
            ("seventeen incidents", "7"),
            ("seventeen incidents", "1"),
            ("seventeen incidents", "71"),
            ("There were 170 incidents", "seventeen"),
            ("There were 117 incidents", "seventeen"),
            ("twenty-one members", "2"),
            ("twenty-one members", "12"),
            ("one hundred and five", "100"),
            ("one hundred and five", "5"),
            ("one hundred and five", "1005"),
            ("two thousand", "2"),
            ("two thousand", "20000"),
            ("12 million dollars", "2000000"),
            ("1.2 million dollars", "12000000"),
            ("1.2 million dollars", "1,200"),
            ("a million dollars", "1000000"),
            ("seventeenth place", "17"),
            ("nineteen eighty-four", "1984"),
            # Different numbers do not match.
            ("seventeen", "18"),
            ("eighteen", "17"),
            ("twenty", "200"),
        ],
    )
    def test_no_false_matches(self, answer: str, alias: str) -> None:
        assert not alias_matches(answer, alias)

    def test_via_facts_and_question_metrics(self) -> None:
        facts = [["17", "seventeen"], ["1,200,000"]]
        assert fact_recall("Seventeen cases cost 1.2 million dollars", facts) == 1.0
        q = {"answerable": True, "required_facts": [["17"]], "evidence_pages": [1]}
        r = {"answer": "There were seventeen.", "cited_pages": [1], "retrieved_pages": [1]}
        assert question_metrics(q, r)["answer_correct"] is True
        r2 = {"answer": "There was one.", "cited_pages": [1], "retrieved_pages": [1]}
        assert question_metrics(q, r2)["answer_correct"] is False


class TestExistingBehaviourUnchanged:
    """The number-word fallback only adds matches; nothing else about grading moved.

    The original assertions in test_eval_metrics.py are untouched and still pass; these
    spot-check the same contracts next to the new feature.
    """

    def test_normalize_text_does_not_touch_words(self) -> None:
        assert normalize_text("Seventeen") == "seventeen"
        assert normalize_text("twenty-one") == "twenty-one"
        assert normalize_text("1.2 million") == "1.2 million"

    def test_token_f1_unchanged(self) -> None:
        assert token_f1("seventeen", "17") == 0.0
        assert token_f1("seventeen", "seventeen") == 1.0

    def test_numeric_alias_detection_unchanged(self) -> None:
        assert not is_numeric_alias("four")
        assert not is_numeric_alias("seventeen")
        assert is_numeric_alias("$1,200")

    def test_old_numeric_boundaries(self) -> None:
        assert not alias_matches("value is 14.25", "4.2")
        assert not alias_matches("value 4.5", "4")
        assert not alias_matches("value 1.4", "4")
        assert not alias_matches("value 42", "4")
        assert alias_matches("value 4", "4")
        assert alias_matches("revenue was 1,234 million", "1234")
        assert not alias_matches("revenue was 11,234 million", "1,234")
        assert alias_matches("grew by 15 percent", "15%")
        assert not alias_matches("grew by 115%", "15%")

    def test_old_text_aliases(self) -> None:
        assert alias_matches("The Board of Directors approved it", "board of directors")
        assert not alias_matches("the board", "board of directors")
        assert not alias_matches("anything", "")
        assert not alias_matches("", "x")

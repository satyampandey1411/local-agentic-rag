"""
Unit tests for local_agentic_rag.eval.metrics

Covers text normalisation (unicode dashes/quotes, thousands commas, percent
variants), fact matching (numeric boundary rules), SQuAD token F1, retrieval
metrics (recall@k, hit@k, MRR incl. duplicates and empty evidence), citation
metrics, per-question scoring and aggregation with bootstrap CIs.
"""

from __future__ import annotations

import math

import pytest

from local_agentic_rag.eval.metrics import (
    aggregate,
    alias_matches,
    all_facts_present,
    as_pages,
    bootstrap_ci,
    citation_precision,
    citation_recall,
    citation_valid,
    fact_hits,
    fact_recall,
    hit_at_k,
    is_numeric_alias,
    mrr,
    normalize_text,
    page_recall_at_k,
    question_metrics,
    token_f1,
)

# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_lowercase_and_whitespace(self) -> None:
        assert normalize_text("  Hello\n\tWORLD  ") == "hello world"

    def test_empty_and_none(self) -> None:
        assert normalize_text("") == ""
        assert normalize_text(None) == ""

    @pytest.mark.parametrize("dash", ["‐", "‑", "‒", "–", "—", "−"])
    def test_unicode_dashes_unified(self, dash: str) -> None:
        assert normalize_text(f"2019{dash}2023") == "2019-2023"

    def test_quotes_unified(self) -> None:
        assert normalize_text("“don’t”") == '"don\'t"'

    def test_nfkc(self) -> None:
        # Full-width digits and the "fi" ligature are folded by NFKC.
        assert normalize_text("１２３ ﬁle") == "123 file"

    def test_thousands_commas_removed(self) -> None:
        assert normalize_text("1,234") == normalize_text("1234")
        assert normalize_text("$1,234,567.89") == "$1234567.89"

    def test_non_thousands_commas_kept(self) -> None:
        assert normalize_text("pages 1,2") == "pages 1,2"
        assert normalize_text("a, b") == "a, b"

    @pytest.mark.parametrize("variant", ["15%", "15 %", "15 percent", "15 per cent", "15 Percent"])
    def test_percent_variants(self, variant: str) -> None:
        assert normalize_text(variant) == "15%"

    def test_percentage_word_untouched(self) -> None:
        assert normalize_text("percentage") == "percentage"


# ---------------------------------------------------------------------------
# Fact matching
# ---------------------------------------------------------------------------


class TestAliasMatching:
    def test_numeric_alias_detection(self) -> None:
        assert is_numeric_alias("4.2")
        assert is_numeric_alias("$1,200")
        assert is_numeric_alias("15 percent")
        assert is_numeric_alias("-3.5%")
        assert not is_numeric_alias("Q4 2023")
        assert not is_numeric_alias("four")

    @pytest.mark.parametrize(
        "answer",
        ["the rate was 14.25", "the rate was 4.25", "it was 14.2", "version 1.4.2"],
    )
    def test_numeric_boundary_rejects_partial_numbers(self, answer: str) -> None:
        assert not alias_matches(answer, "4.2")

    @pytest.mark.parametrize(
        "answer",
        ["4.2", "It grew 4.2.", "about 4.2% in 2023", "(4.2)", "rate: 4.2x faster", "$4.2"],
    )
    def test_numeric_boundary_accepts_whole_numbers(self, answer: str) -> None:
        assert alias_matches(answer, "4.2")

    def test_integer_alias_not_inside_decimal(self) -> None:
        assert not alias_matches("value 4.5", "4")
        assert not alias_matches("value 1.4", "4")
        assert not alias_matches("value 42", "4")
        assert alias_matches("value 4", "4")

    def test_comma_numbers_match_either_way(self) -> None:
        assert alias_matches("revenue was 1,234 million", "1234")
        assert alias_matches("revenue was 1234 million", "1,234")
        assert not alias_matches("revenue was 11,234 million", "1,234")

    def test_percent_equivalence(self) -> None:
        assert alias_matches("grew by 15 percent", "15%")
        assert alias_matches("grew by 15%", "15 percent")
        assert not alias_matches("grew by 115%", "15%")

    def test_unicode_minus_matches_ascii(self) -> None:
        assert alias_matches("change of −3.5%", "-3.5%")

    def test_text_alias_substring_and_case(self) -> None:
        assert alias_matches("The Board of Directors approved it", "board of directors")
        assert alias_matches("state–of–the–art", "state-of-the-art")
        assert not alias_matches("the board", "board of directors")

    def test_empty_alias_or_answer(self) -> None:
        assert not alias_matches("anything", "")
        assert not alias_matches("", "x")


class TestFactRecall:
    FACTS = [["4.2%", "4.2 percent"], ["2023", "fiscal 2023"], ["Europe"]]

    def test_all_present(self) -> None:
        ans = "In 2023 revenue in europe grew 4.2 percent."
        assert fact_recall(ans, self.FACTS) == 1.0
        assert all_facts_present(ans, self.FACTS)

    def test_partial(self) -> None:
        ans = "Revenue grew 4.2% in 2023."
        assert fact_hits(ans, self.FACTS) == [True, True, False]
        assert fact_recall(ans, self.FACTS) == pytest.approx(2 / 3)
        assert not all_facts_present(ans, self.FACTS)

    def test_none_present(self) -> None:
        assert fact_recall("no idea", self.FACTS) == 0.0

    def test_numeric_alias_boundary_in_facts(self) -> None:
        assert fact_recall("it was 14.25", [["4.2"]]) == 0.0

    def test_empty_facts_vacuous(self) -> None:
        assert fact_recall("anything", []) == 1.0
        assert all_facts_present("anything", [])

    def test_none_answer(self) -> None:
        assert fact_recall(None, [["x"]]) == 0.0

    def test_bare_string_fact_is_single_alias(self) -> None:
        assert fact_recall("paris is the capital", ["Paris", ["France"]]) == 0.5


# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------


class TestTokenF1:
    def test_exact_match(self) -> None:
        assert token_f1("The quick fox", "the quick fox") == 1.0

    def test_articles_and_punctuation_ignored(self) -> None:
        assert token_f1("A fox!", "fox") == 1.0

    def test_partial_overlap(self) -> None:
        # pred tokens: cat sat ; ref tokens: cat sat down -> P=1, R=2/3, F1=0.8
        assert token_f1("the cat sat", "a cat sat down") == pytest.approx(0.8)

    def test_no_overlap(self) -> None:
        assert token_f1("dog", "cat") == 0.0

    def test_empty_cases(self) -> None:
        assert token_f1("", "") == 1.0
        assert token_f1("", "cat") == 0.0
        assert token_f1("cat", "") == 0.0
        assert token_f1("the", "a") == 1.0  # both empty after article removal

    def test_numbers_kept_whole(self) -> None:
        assert token_f1("4.2%", "4.2 percent") == 1.0
        assert token_f1("4.2", "42") == 0.0

    def test_repeated_tokens_counted_with_multiplicity(self) -> None:
        # pred: cat cat ; ref: cat -> common=1, P=0.5, R=1, F1=2/3
        assert token_f1("cat cat", "cat") == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


class TestRetrieval:
    def test_as_pages_dedupes_and_coerces(self) -> None:
        assert as_pages([3, 1, 3, "2", None, "x", 1]) == [3, 1, 2]
        assert as_pages(None) == []

    def test_recall_at_k_basic(self) -> None:
        ranked = [5, 2, 9, 7]
        assert page_recall_at_k(ranked, [2, 7], 1) == 0.0
        assert page_recall_at_k(ranked, [2, 7], 2) == 0.5
        assert page_recall_at_k(ranked, [2, 7], 4) == 1.0
        assert page_recall_at_k(ranked, [2, 7], 100) == 1.0

    def test_recall_uses_unique_pages(self) -> None:
        # Duplicates must not consume top-k slots.
        ranked = [5, 5, 5, 2]
        assert page_recall_at_k(ranked, [2], 2) == 1.0
        assert hit_at_k(ranked, [2], 2) == 1.0
        assert mrr(ranked, [2]) == 0.5

    def test_duplicate_evidence_counted_once(self) -> None:
        assert page_recall_at_k([1], [1, 1, 2], 1) == 0.5

    def test_empty_evidence_is_undefined(self) -> None:
        assert page_recall_at_k([1, 2], [], 5) is None
        assert hit_at_k([1, 2], [], 5) is None
        assert mrr([1, 2], []) is None

    def test_empty_ranking(self) -> None:
        assert page_recall_at_k([], [1], 5) == 0.0
        assert hit_at_k(None, [1], 5) == 0.0
        assert mrr([], [1]) == 0.0

    def test_hit_at_k(self) -> None:
        assert hit_at_k([4, 3, 1], [1, 9], 3) == 1.0
        assert hit_at_k([4, 3, 1], [1, 9], 2) == 0.0

    def test_mrr(self) -> None:
        assert mrr([1, 2, 3], [1]) == 1.0
        assert mrr([4, 2, 3], [3, 2]) == 0.5
        assert mrr([4, 5], [3]) == 0.0

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError):
            page_recall_at_k([1], [1], 0)
        with pytest.raises(ValueError):
            hit_at_k([1], [1], -1)


# ---------------------------------------------------------------------------
# Citation metrics
# ---------------------------------------------------------------------------


class TestCitations:
    def test_precision(self) -> None:
        assert citation_precision([3, 4], [3]) == 0.5
        assert citation_precision([3, 3], [3]) == 1.0  # deduped
        assert citation_precision([9], [3]) == 0.0

    def test_precision_undefined(self) -> None:
        assert citation_precision([], [3]) is None
        assert citation_precision(None, [3]) is None
        assert citation_precision([3], []) is None

    def test_recall(self) -> None:
        assert citation_recall([3], [3, 4]) == 0.5
        assert citation_recall([], [3, 4]) == 0.0
        assert citation_recall([3], []) is None

    def test_valid(self) -> None:
        assert citation_valid([2, 3], [1, 2, 3])
        assert not citation_valid([2, 8], [1, 2, 3])
        assert citation_valid([], [1])  # vacuous
        assert not citation_valid([1], [])


# ---------------------------------------------------------------------------
# question_metrics
# ---------------------------------------------------------------------------


def _q(answerable: bool = True, **kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "q_id": "q1",
        "doc_id": "D",
        "category": "numerical" if answerable else "unanswerable",
        "answerable": answerable,
        "reference_answer": "Revenue grew 4.2%",
        "required_facts": [["4.2%"]] if answerable else [],
        "evidence_pages": [3] if answerable else [],
    }
    base.update(kw)
    return base


def _r(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "answer": "Revenue grew 4.2 percent [Source: Page 3]",
        "refused": False,
        "cited_pages": [3],
        "retrieved_pages": [3, 5, 3],
        "latency_sec": 1.0,
        "error": None,
    }
    base.update(kw)
    return base


class TestQuestionMetrics:
    def test_correct_answer(self) -> None:
        m = question_metrics(_q(), _r())
        assert m["answer_correct"] is True
        assert m["fact_recall"] == 1.0
        assert m["false_refusal"] is False
        assert m["citation_precision"] == 1.0
        assert m["citation_recall"] == 1.0
        assert m["citation_valid"] is True
        assert m["has_citation"] is True
        assert m["recall@1"] == 1.0
        assert m["rr"] == 1.0
        assert m["refusal_correct"] is None
        assert m["hallucinated"] is None
        assert m["error"] is False

    def test_refused_answerable_is_wrong_even_if_facts_present(self) -> None:
        m = question_metrics(_q(), _r(refused=True))
        assert m["all_facts_present"] is True
        assert m["answer_correct"] is False
        assert m["false_refusal"] is True
        assert m["has_citation"] is None  # citations only judged on answered questions
        assert m["citation_precision"] is None

    def test_answer_without_citations(self) -> None:
        m = question_metrics(_q(), _r(cited_pages=[]))
        assert m["has_citation"] is False
        assert m["citation_precision"] is None
        assert m["citation_recall"] is None

    def test_fabricated_citation(self) -> None:
        m = question_metrics(_q(), _r(cited_pages=[3, 99]))
        assert m["citation_valid"] is False
        assert m["citation_precision"] == 0.5

    def test_errored_answerable(self) -> None:
        m = question_metrics(_q(), _r(answer="", error="boom", cited_pages=[]))
        assert m["error"] is True
        assert m["answer_correct"] is False
        assert m["has_citation"] is None

    def test_unanswerable_refused(self) -> None:
        m = question_metrics(_q(False), _r(refused=True, answer="ANSWER_NOT_FOUND"))
        assert m["refusal_correct"] is True
        assert m["hallucinated"] is False
        assert m["answer_correct"] is None
        assert m["recall@5"] is None

    def test_unanswerable_answered_is_hallucination(self) -> None:
        m = question_metrics(_q(False), _r())
        assert m["refusal_correct"] is False
        assert m["hallucinated"] is True

    def test_unanswerable_error_not_hallucination(self) -> None:
        m = question_metrics(_q(False), _r(answer="", error="oom"))
        assert m["refusal_correct"] is False
        assert m["hallucinated"] is False

    def test_answer_field_fallbacks(self) -> None:
        r = _r()
        del r["answer"]
        r["extracted_answer"] = "4.2%"
        assert question_metrics(_q(), r)["answer_correct"] is True

    def test_custom_ks(self) -> None:
        m = question_metrics(_q(), _r(), ks=(2,))
        assert "recall@2" in m and "recall@5" not in m


# ---------------------------------------------------------------------------
# bootstrap + aggregate
# ---------------------------------------------------------------------------


class TestBootstrap:
    def test_empty(self) -> None:
        assert bootstrap_ci([]) is None

    def test_constant(self) -> None:
        assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)

    def test_seeded_and_contains_mean(self) -> None:
        vals = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
        a = bootstrap_ci(vals, seed=7)
        b = bootstrap_ci(vals, seed=7)
        assert a == b
        assert a is not None
        lo, hi = a
        assert 0.0 <= lo <= 0.7 <= hi <= 1.0


def _record(q: dict[str, object], r: dict[str, object], **extra: object) -> dict[str, object]:
    rec = {**q, **r, **extra}
    rec["metrics"] = question_metrics(q, r)
    return rec


class TestAggregate:
    @pytest.fixture()
    def records(self) -> list[dict[str, object]]:
        return [
            # correct, cited correctly, retrieved at rank 1
            _record(_q(q_id="a1", doc_id="D1"), _r(latency_sec=1.0, iterations=1)),
            # wrong answer, no citations, evidence retrieved at rank 2
            _record(
                _q(q_id="a2", doc_id="D1", category="factual"),
                _r(
                    answer="unknown",
                    cited_pages=[],
                    retrieved_pages=[5, 3],
                    latency_sec=2.0,
                    iterations=3,
                ),
            ),
            # false refusal
            _record(
                _q(q_id="a3", doc_id="D2"),
                _r(
                    refused=True,
                    answer="ANSWER_NOT_FOUND",
                    retrieved_pages=[9],
                    latency_sec=3.0,
                    peak_vram_bytes=2048,
                ),
            ),
            # unanswerable, correctly refused
            _record(_q(False, q_id="u1", doc_id="D2"), _r(refused=True, latency_sec=4.0)),
            # unanswerable, hallucinated
            _record(_q(False, q_id="u2", doc_id="D1"), _r(latency_sec=5.0, peak_vram_bytes=1024)),
            # errored answerable
            _record(
                _q(q_id="a4", doc_id="D2"),
                _r(
                    answer="",
                    error="RuntimeError: x",
                    cited_pages=[],
                    retrieved_pages=[],
                    latency_sec=None,
                ),
            ),
        ]

    def test_overall(self, records: list[dict[str, object]]) -> None:
        s = aggregate(records)["overall"]
        assert s["n"] == 6
        assert s["n_answerable"] == 4
        assert s["n_unanswerable"] == 2
        assert s["answer_accuracy"] == pytest.approx(1 / 4)
        assert s["false_refusal_rate"] == pytest.approx(1 / 4)
        assert s["fact_recall"] == pytest.approx(1 / 4)  # only a1 states "4.2%"
        assert s["refusal_accuracy"] == pytest.approx(0.5)
        assert s["hallucination_rate"] == pytest.approx(0.5)
        assert s["error_rate"] == pytest.approx(1 / 6)
        # answered answerable = a1, a2 -> one with citations
        assert s["citation_coverage"] == pytest.approx(0.5)
        assert s["citation_precision"] == pytest.approx(1.0)
        assert s["citation_recall"] == pytest.approx(1.0)
        assert s["citation_validity"] == pytest.approx(1.0)
        # retrieval over the 4 answerable: a1 rank1, a2 rank2, a3 miss, a4 miss
        assert s["recall@1"] == pytest.approx(1 / 4)
        assert s["recall@3"] == pytest.approx(2 / 4)
        assert s["hit@5"] == pytest.approx(2 / 4)
        assert s["mrr"] == pytest.approx((1 + 0.5) / 4)
        # latency over the 5 records that report it
        assert s["latency_p50"] == pytest.approx(3.0)
        assert s["latency_mean"] == pytest.approx(3.0)
        assert s["latency_p95"] == pytest.approx(4.8)
        assert s["mean_iterations"] == pytest.approx(2.0)
        assert s["max_peak_vram_bytes"] == 2048

    def test_cis_present_and_bracket_point_estimate(self, records: list[dict[str, object]]) -> None:
        s = aggregate(records)["overall"]
        lo, hi = s["answer_accuracy_ci"]
        assert lo <= s["answer_accuracy"] <= hi
        lo, hi = s["refusal_accuracy_ci"]
        assert lo <= s["refusal_accuracy"] <= hi

    def test_reproducible(self, records: list[dict[str, object]]) -> None:
        assert aggregate(records, seed=3) == aggregate(records, seed=3)

    def test_per_category_and_doc(self, records: list[dict[str, object]]) -> None:
        agg = aggregate(records)
        assert set(agg["per_category"]) == {"factual", "numerical", "unanswerable"}
        unans = agg["per_category"]["unanswerable"]
        assert unans["answer_accuracy"] is None
        assert unans["answer_accuracy_ci"] is None
        assert unans["refusal_accuracy"] == pytest.approx(0.5)
        assert agg["per_category"]["factual"]["answer_accuracy"] == 0.0
        assert set(agg["per_doc"]) == {"D1", "D2"}
        assert agg["per_doc"]["D1"]["n"] == 3

    def test_metrics_computed_when_missing(self, records: list[dict[str, object]]) -> None:
        stripped = [{k: v for k, v in r.items() if k != "metrics"} for r in records]
        assert aggregate(stripped) == aggregate(records)

    def test_empty(self) -> None:
        s = aggregate([])["overall"]
        assert s["n"] == 0
        assert s["answer_accuracy"] is None
        assert s["error_rate"] is None
        assert s["latency_p50"] is None
        assert s["max_peak_vram_bytes"] is None

    def test_values_are_finite(self, records: list[dict[str, object]]) -> None:
        s = aggregate(records)["overall"]
        for k, v in s.items():
            if isinstance(v, float):
                assert math.isfinite(v), k

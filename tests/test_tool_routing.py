"""Deterministic tests for the Beacon Study Hub routing evaluation."""

from collections import Counter

from evaluate_tool_choice import (
    KNOWN_TOOLS,
    TEST_CASES,
    deterministic_rule_router,
    evaluate_cases,
    normalize_tool_names,
)


def test_question_bank_is_bilingual_and_covers_all_required_categories():
    assert len(TEST_CASES) >= 12

    languages = Counter(case.language for case in TEST_CASES)
    categories = Counter(case.category for case in TEST_CASES)

    assert languages["zh-TW"] >= 6
    assert languages["en"] >= 6
    assert set(categories) == {
        "in_data",
        "calculation",
        "not_in_data_or_transfer",
        "out_of_scope",
    }
    assert all(count >= 2 for count in categories.values())


def test_every_expected_name_is_a_declared_app_tool():
    expected_names = {
        name for case in TEST_CASES for name in case.acceptable_tools
    }
    assert expected_names == KNOWN_TOOLS


def test_out_of_scope_cases_expect_no_tool_call():
    out_of_scope = [
        case for case in TEST_CASES if case.category == "out_of_scope"
    ]
    assert out_of_scope
    assert all(case.acceptable_tools == () for case in out_of_scope)


def test_local_rule_router_scores_100_percent_without_network():
    results = evaluate_cases(deterministic_rule_router)
    failures = [
        (
            result.case.case_id,
            result.case.acceptable_tools,
            result.actual_tools,
            result.error,
        )
        for result in results
        if not result.passed
    ]
    assert not failures


def test_app_offline_router_matches_the_full_question_bank():
    # This helper is explicitly offline; importing/calling it must not use an API.
    from app import predict_tool_names

    results = evaluate_cases(predict_tool_names)
    failures = [
        (
            result.case.case_id,
            result.case.acceptable_tools,
            result.actual_tools,
            result.error,
        )
        for result in results
        if not result.passed
    ]
    assert not failures


def test_evaluator_accepts_an_injected_fake_router():
    expected_by_question = {
        case.question: list(case.acceptable_tools) for case in TEST_CASES
    }

    def fake_router(question):
        return expected_by_question[question]

    results = evaluate_cases(fake_router)
    assert len(results) == len(TEST_CASES)
    assert all(result.passed for result in results)


def test_tool_name_normalization_handles_common_router_shapes():
    assert normalize_tool_names("get_center_info") == ["get_center_info"]
    assert normalize_tool_names(
        [{"name": "search_study_plans"}, {"tool": "transfer_to_human"}]
    ) == ["search_study_plans", "transfer_to_human"]
    assert normalize_tool_names(
        {"tool_names": ["calculate_booking_cost", "calculate_booking_cost"]}
    ) == ["calculate_booking_cost"]
    assert normalize_tool_names(None) == []


def test_router_exceptions_are_reported_as_failures_not_raised():
    def broken_router(_question):
        raise RuntimeError("simulated failure")

    results = evaluate_cases(broken_router, cases=TEST_CASES[:1])
    assert len(results) == 1
    assert not results[0].passed
    assert results[0].actual_tools == ()
    assert "simulated failure" in (results[0].error or "")

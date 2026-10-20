"""Evaluate Beacon Study Hub tool routing with bilingual questions.

The default (offline) mode prefers ``app.predict_tool_names`` when that
deterministic helper is available.  A small local router is kept as a fallback
so the question bank can still be reviewed before the app or its optional
packages are installed.

Use ``--live`` to exercise ``app.predict_tool_names_live``.  Live mode sleeps
between questions to reduce the chance of API rate-limit errors.
"""

from __future__ import annotations

import argparse
import importlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


Router = Callable[[str], Any]

KNOWN_TOOLS = frozenset(
    {
        "get_center_info",
        "search_study_plans",
        "calculate_booking_cost",
        "transfer_to_human",
    }
)


@dataclass(frozen=True)
class ToolChoiceCase:
    """One routing question and the complete set of acceptable tools."""

    case_id: str
    language: str
    category: str
    question: str
    acceptable_tools: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """The normalized result for one test case."""

    case: ToolChoiceCase
    actual_tools: tuple[str, ...]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and set(self.actual_tools) == set(
            self.case.acceptable_tools
        )


# Four balanced categories, with Chinese and English represented in each.
# Calculation questions use the assignment's price x people x sessions rule.
TEST_CASES: tuple[ToolChoiceCase, ...] = (
    ToolChoiceCase(
        "D01",
        "zh-TW",
        "in_data",
        "Beacon Study Hub 平日幾點開門、幾點關門？",
        ("get_center_info",),
    ),
    ToolChoiceCase(
        "D02",
        "en",
        "in_data",
        "Do you provide Wi-Fi and power outlets?",
        ("get_center_info",),
    ),
    ToolChoiceCase(
        "D03",
        "zh-TW",
        "in_data",
        "Focus Booth 6-Hour 目前可以預約嗎？",
        ("search_study_plans",),
    ),
    ToolChoiceCase(
        "D04",
        "en",
        "in_data",
        "Which monthly study plans are currently available?",
        ("search_study_plans",),
    ),
    ToolChoiceCase(
        "C01",
        "zh-TW",
        "calculation",
        "Meeting Room 2-Hour 方案給 3 個人使用 2 次，總費用是多少？",
        ("calculate_booking_cost",),
    ),
    ToolChoiceCase(
        "C02",
        "en",
        "calculation",
        "What is the total cost of the Quiet Desk 3-Hour plan for 2 people over 4 sessions?",
        ("calculate_booking_cost",),
    ),
    ToolChoiceCase(
        "C03",
        "zh-TW",
        "calculation",
        "Night Owl 3-Hour 兩個人預約 3 次，請幫我計算多少錢。",
        ("calculate_booking_cost",),
    ),
    ToolChoiceCase(
        "C04",
        "en",
        "calculation",
        "Please calculate a Window Seat 4-Hour booking for 1 person and 5 sessions.",
        ("calculate_booking_cost",),
    ),
    ToolChoiceCase(
        "T01",
        "zh-TW",
        "not_in_data_or_transfer",
        "你們可以保證我下次多益考到 900 分嗎？",
        ("transfer_to_human",),
    ),
    ToolChoiceCase(
        "T02",
        "en",
        "not_in_data_or_transfer",
        "Can a Beacon tutor come to my home at midnight?",
        ("transfer_to_human",),
    ),
    ToolChoiceCase(
        "T03",
        "zh-TW",
        "not_in_data_or_transfer",
        "我要針對退款爭議跟真人客服談。",
        ("transfer_to_human",),
    ),
    ToolChoiceCase(
        "T04",
        "en",
        "not_in_data_or_transfer",
        "I have a complaint and want to speak to a staff member.",
        ("transfer_to_human",),
    ),
    ToolChoiceCase(
        "O01",
        "zh-TW",
        "out_of_scope",
        "幫我預測明天台積電股價。",
        (),
    ),
    ToolChoiceCase(
        "O02",
        "en",
        "out_of_scope",
        "What will the weather be in Tokyo tomorrow?",
        (),
    ),
    ToolChoiceCase(
        "O03",
        "zh-TW",
        "out_of_scope",
        "請寫一首關於海邊的情歌。",
        (),
    ),
    ToolChoiceCase(
        "O04",
        "en",
        "out_of_scope",
        "Who will win the next NBA championship?",
        (),
    ),
)


def deterministic_rule_router(message: str) -> list[str]:
    """Return a deterministic tool prediction without importing the app.

    This is intentionally a routing test double, not the customer-service
    answer generator.  It keeps local tests independent from API credentials
    and network availability.
    """

    text = message.casefold()

    out_of_scope_terms = (
        "股價",
        "台積電",
        "weather",
        "天氣",
        "情歌",
        "寫一首",
        "nba",
        "championship",
        "bitcoin",
        "recipe",
    )
    if any(term in text for term in out_of_scope_terms):
        return []

    transfer_terms = (
        "保證",
        "900 分",
        "come to my home",
        "at midnight",
        "真人",
        "退款爭議",
        "complaint",
        "staff member",
        "human",
        "manager",
    )
    if any(term in text for term in transfer_terms):
        return ["transfer_to_human"]

    calculation_terms = (
        "總費用",
        "多少錢",
        "幫我計算",
        "total cost",
        "please calculate",
        "sessions",
    )
    if any(term in text for term in calculation_terms):
        return ["calculate_booking_cost"]

    info_terms = (
        "幾點開門",
        "幾點關門",
        "opening hour",
        "wi-fi",
        "wifi",
        "power outlet",
        "address",
        "parking",
        "payment",
        "cancellation",
        "accessibility",
    )
    if any(term in text for term in info_terms):
        return ["get_center_info"]

    plan_terms = (
        "study plan",
        "monthly",
        "focus booth",
        "quiet desk",
        "window seat",
        "meeting room",
        "group study room",
        "night owl",
        "early bird",
        "方案",
        "可以預約",
    )
    if any(term in text for term in plan_terms):
        return ["search_study_plans"]

    # An on-topic request not covered by known data should be handed to staff.
    beacon_terms = ("beacon", "study hub", "tutor", "預約", "讀書", "客服")
    if any(term in text for term in beacon_terms):
        return ["transfer_to_human"]
    return []


def normalize_tool_names(raw: Any) -> list[str]:
    """Normalize common router return shapes to a de-duplicated name list."""

    if raw is None:
        return []
    if isinstance(raw, str):
        candidates: Iterable[Any] = raw.split(",")
    elif isinstance(raw, Mapping):
        for key in ("tool_names", "tools", "actual_tools"):
            if key in raw:
                return normalize_tool_names(raw[key])
        if "name" in raw:
            return normalize_tool_names(raw["name"])
        if "tool" in raw:
            return normalize_tool_names(raw["tool"])
        return []
    elif isinstance(raw, Sequence):
        candidates = raw
    else:
        name = getattr(raw, "name", None)
        return normalize_tool_names(name)

    names: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            nested = normalize_tool_names(candidate)
        elif not isinstance(candidate, str):
            nested = normalize_tool_names(candidate)
        else:
            nested = [candidate.strip()]
        for name in nested:
            if name and name not in names:
                names.append(name)
    return names


def load_router(live: bool = False) -> tuple[Router, str]:
    """Load the appropriate app router, with an offline-only fallback."""

    function_name = "predict_tool_names_live" if live else "predict_tool_names"
    try:
        app_module = importlib.import_module("app")
        app_router = getattr(app_module, function_name, None)
        if callable(app_router):
            return app_router, f"app.{function_name}"
    except Exception as exc:  # Optional UI/API packages may not be installed yet.
        if live:
            raise RuntimeError(f"Could not import app.py for live mode: {exc}") from exc

    if live:
        raise RuntimeError(
            "Live mode requires app.predict_tool_names_live(message). "
            "Configure .env locally, then try again."
        )
    return deterministic_rule_router, "local deterministic fallback"


def evaluate_cases(
    router: Router,
    cases: Sequence[ToolChoiceCase] = TEST_CASES,
    delay_seconds: float = 0.0,
) -> list[EvaluationResult]:
    """Run cases through an injected router without assuming network access."""

    if delay_seconds < 0:
        raise ValueError("delay_seconds must be zero or greater")

    results: list[EvaluationResult] = []
    for index, case in enumerate(cases):
        try:
            actual_tools = tuple(normalize_tool_names(router(case.question)))
            result = EvaluationResult(case, actual_tools)
        except Exception as exc:  # Continue so a live run reports every miss.
            result = EvaluationResult(case, (), f"{type(exc).__name__}: {exc}")
        results.append(result)
        if delay_seconds and index < len(cases) - 1:
            time.sleep(delay_seconds)
    return results


def _format_tools(tools: Sequence[str]) -> str:
    return ", ".join(tools) if tools else "(no tool)"


def print_report(results: Sequence[EvaluationResult], router_source: str) -> None:
    """Print question-level evidence plus the overall score."""

    print(f"Router: {router_source}")
    print(f"Cases: {len(results)}")
    print()
    for result in results:
        case = result.case
        status = "PASS" if result.passed else "FAIL"
        print(f"[{case.case_id}] {status} | {case.language} | {case.category}")
        print(f"  Question: {case.question}")
        print(f"  Expected: {_format_tools(case.acceptable_tools)}")
        print(f"  Actual:   {_format_tools(result.actual_tools)}")
        if result.error:
            print(f"  Error:    {result.error}")
        print()

    passed = sum(result.passed for result in results)
    percent = (passed / len(results) * 100.0) if results else 0.0
    print(f"Score: {passed}/{len(results)} ({percent:.1f}%)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Beacon Study Hub's bilingual tool-choice evaluation."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call app.predict_tool_names_live (may use the configured API).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds between live questions to reduce rate limits (default: 1.5).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.delay < 0:
        raise SystemExit("--delay must be zero or greater")

    try:
        router, router_source = load_router(live=args.live)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    results = evaluate_cases(
        router,
        delay_seconds=args.delay if args.live else 0.0,
    )
    print_report(results, router_source)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

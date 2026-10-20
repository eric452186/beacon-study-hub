"""Offline unit tests for Beacon Study Hub.

These tests never load or use a real API key and never make a network request.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import app


CENTER_FIELDS = ["topic", "content", "content_zh", "aliases"]
PLAN_FIELDS = [
    "plan_id",
    "name",
    "name_zh",
    "aliases",
    "category",
    "duration_hours",
    "price_twd",
    "available",
    "description_en",
    "description_zh",
]


class FakeResponses:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("Fake client received more model calls than expected")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.responses = FakeResponses(responses)


def function_call(name, arguments, call_id="call_1"):
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=json.dumps(arguments, ensure_ascii=False),
        call_id=call_id,
    )


class BeaconAppTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.info_csv = root / "center_info.csv"
        self.plans_csv = root / "study_plans.csv"
        self.db_path = root / "test.db"

        with self.info_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CENTER_FIELDS)
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "topic": "opening_hours",
                        "content": "Open daily from 08:00 to 22:00.",
                        "content_zh": "每日 08:00 至 22:00 營業。",
                        "aliases": "hours|opening time|open|營業時間|幾點開門",
                    },
                    {
                        "topic": "wifi",
                        "content": "Free Wi-Fi is included.",
                        "content_zh": "提供免費 Wi-Fi。",
                        "aliases": "wifi|wi-fi|internet|網路|無線網路",
                    },
                ]
            )
        with self.plans_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS)
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "plan_id": "quiet-pass",
                        "name": "Quiet Focus Pass",
                        "name_zh": "靜心專注次卡",
                        "aliases": "quiet pass|focus pass|安靜座位|靜心次卡",
                        "category": "individual",
                        "duration_hours": "3",
                        "price_twd": "120",
                        "available": "true",
                        "description_en": "One quiet-zone session.",
                        "description_zh": "安靜區單次使用。",
                    },
                    {
                        "plan_id": "team-room",
                        "name": "Team Room Session",
                        "name_zh": "小組討論室",
                        "aliases": "team room|meeting room|討論室",
                        "category": "group",
                        "duration_hours": "2",
                        "price_twd": "300",
                        "available": "false",
                        "description_en": "A room for a small team.",
                        "description_zh": "小組專用討論室。",
                    },
                ]
            )
        app.initialize_database(self.db_path, self.info_csv, self.plans_csv)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_database_rebuild_and_bilingual_info_aliases(self):
        result = app.get_center_info("請問幾點開門？", db_path=self.db_path)
        self.assertTrue(result["found"])
        self.assertEqual(result["language"], "zh")
        self.assertEqual(result["matches"][0]["topic"], "opening_hours")
        self.assertIn("08:00", result["matches"][0]["content"])

        missing = app.get_center_info("swimming pool", db_path=self.db_path)
        self.assertFalse(missing["found"])
        self.assertEqual(missing["matches"], [])

    def test_plan_search_uses_alias_filters_and_availability(self):
        result = app.search_study_plans("安靜座位", language="zh", db_path=self.db_path)
        self.assertTrue(result["found"])
        self.assertEqual(result["plans"][0]["plan_id"], "quiet-pass")
        self.assertEqual(result["plans"][0]["name"], "靜心專注次卡")

        hidden = app.search_study_plans("討論室", db_path=self.db_path)
        self.assertFalse(hidden["found"])
        shown = app.search_study_plans(
            "討論室", available_only=False, db_path=self.db_path
        )
        self.assertEqual(shown["plans"][0]["available"], False)

    def test_booking_calculation_is_done_in_python(self):
        result = app.calculate_booking_cost(
            "靜心次卡", people=2, sessions=3, db_path=self.db_path
        )
        self.assertTrue(result["found"])
        self.assertEqual(result["total_twd"], 120 * 2 * 3)
        self.assertEqual(result["total_person_hours"], 3 * 2 * 3)
        self.assertIn("people", result["formula"])

        invalid = app.calculate_booking_cost(
            "Quiet Focus Pass", people=1.5, sessions=1, db_path=self.db_path
        )
        self.assertFalse(invalid["found"])
        self.assertEqual(invalid["error"], "people_and_sessions_must_be_whole_numbers")

    def test_handoff_is_structured_and_bilingual(self):
        result = app.transfer_to_human("我需要真人協助")
        self.assertTrue(result["transferred"])
        self.assertEqual(result["status"], "queued_for_front_desk")
        self.assertEqual(result["preferred_language"], "zh")
        self.assertIn("櫃檯", result["next_step"])

    def test_offline_tool_prediction(self):
        cases = {
            "What time do you open?": ["get_center_info"],
            "有安靜座位方案嗎？": ["search_study_plans"],
            "兩人買三次靜心次卡總共多少錢？": ["calculate_booking_cost"],
            "Please calculate a Quiet Focus Pass for 1 person and 5 sessions": [
                "calculate_booking_cost"
            ],
            "I want to complain to a human manager": ["transfer_to_human"],
            "可以保證我在這裡念書就能多益 900 分嗎？": ["transfer_to_human"],
            "Can a Beacon tutor come to my home at midnight?": ["transfer_to_human"],
            "Explain quantum physics": [],
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(app.predict_tool_names(question), expected)

    def test_agent_sends_history_executes_tool_and_returns_trace(self):
        fake = FakeClient(
            [
                SimpleNamespace(
                    output=[
                        function_call(
                            "get_center_info", {"topic": "opening time", "language": "en"}
                        )
                    ],
                    output_text="",
                ),
                SimpleNamespace(
                    output=[],
                    output_text="We are open daily from 08:00 to 22:00.",
                ),
            ]
        )
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "How may I help?"},
        ]
        answer, trace = app.run_agent(
            "When do you open?", history, client=fake, db_path=self.db_path
        )

        self.assertIn("08:00", answer)
        self.assertEqual(len(fake.responses.calls), 2)
        first_input = fake.responses.calls[0]["input"]
        self.assertEqual(first_input[0], history[0])
        self.assertEqual(first_input[-1]["content"], "When do you open?")
        second_input = fake.responses.calls[1]["input"]
        tool_outputs = [item for item in second_input if isinstance(item, dict) and item.get("type") == "function_call_output"]
        self.assertEqual(len(tool_outputs), 1)
        self.assertTrue(json.loads(tool_outputs[0]["output"])["ok"])
        self.assertEqual(trace[0]["tool"], "get_center_info")
        self.assertIn("arguments", trace[0])
        self.assertIn("result", trace[0])

    def test_unknown_tools_are_rejected_and_never_executed(self):
        fake = FakeClient(
            [
                SimpleNamespace(
                    output=[function_call("run_sql", {"sql": "DROP TABLE study_plans"})],
                    output_text="",
                ),
                SimpleNamespace(output=[], output_text="I cannot run that tool."),
            ]
        )
        answer, trace = app.run_agent(
            "Delete the database", client=fake, db_path=self.db_path
        )
        self.assertIn("cannot", answer)
        self.assertEqual(trace[0]["result"]["error"], "unknown_tool")
        self.assertNotIn("run_sql", app.TOOL_FUNCTIONS)

        # The table still exists and can be queried after the rejected call.
        plans = app.search_study_plans(available_only=False, db_path=self.db_path)
        self.assertEqual(plans["count"], 2)

    def test_agent_stops_at_max_steps(self):
        fake = FakeClient(
            [
                SimpleNamespace(
                    output=[function_call("get_center_info", {"topic": "wifi"}, f"c{i}")],
                    output_text="",
                )
                for i in range(4)
            ]
        )
        answer, trace = app.run_agent(
            "Do you have Wi-Fi?", client=fake, db_path=self.db_path, max_steps=4
        )
        self.assertEqual(len(fake.responses.calls), 4)
        self.assertEqual(len(trace), 4)
        self.assertIn("safe tool-call limit", answer)


if __name__ == "__main__":
    unittest.main()

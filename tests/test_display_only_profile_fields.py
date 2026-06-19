from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.v1.recommendation import PersonalizedPageRankRequest
from app.services.personalized_pagerank_service import (
    PersonalizedPageRankRecommendationService,
)
from app.services.prompt_builder import build_yandex_gpt_recommendation_prompt


MENTEE_FIELDS = {
    "request_scenario",
    "help_format",
    "urgency",
    "motivation",
    "expected_result",
}
MENTOR_FIELDS = {
    "request_scenarios",
    "help_formats",
}


class DisplayOnlyProfileFieldsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data_dir = PROJECT_ROOT / "data"
        cls.mentees = json.loads((data_dir / "mentees.json").read_text(encoding="utf-8"))
        cls.mentors = json.loads((data_dir / "mentors.json").read_text(encoding="utf-8"))

    def test_fields_exist_and_high_profiles_are_filled(self) -> None:
        for mentee in self.mentees:
            self.assertTrue(MENTEE_FIELDS.issubset(mentee))
            if mentee.get("profile_completeness") == "high":
                for field in MENTEE_FIELDS:
                    self.assertTrue(mentee[field], f"{mentee.get('id')}: {field}")

        for mentor in self.mentors:
            self.assertTrue(MENTOR_FIELDS.issubset(mentor))
            if mentor.get("profile_completeness") == "high":
                for field in MENTOR_FIELDS:
                    self.assertTrue(mentor[field], f"{mentor.get('id')}: {field}")

    def test_ui_shows_requested_fields(self) -> None:
        template = (
            PROJECT_ROOT / "app" / "templates" / "recommendations.html"
        ).read_text(encoding="utf-8")
        for label in (
            "Сценарий запроса",
            "Формат помощи",
            "Срочность",
            "Мотивация",
            "Ожидаемый результат",
        ):
            self.assertIn(label, template)

    def test_yandex_gpt_prompt_excludes_display_only_fields(self) -> None:
        mentee = copy.deepcopy(self.mentees[0])
        mentor = copy.deepcopy(self.mentors[0])
        for field in MENTEE_FIELDS:
            mentee[field] = f"DISPLAY_ONLY_MENTEE_{field}"
        for field in MENTOR_FIELDS:
            mentor[field] = [f"DISPLAY_ONLY_MENTOR_{field}"]

        prompt = build_yandex_gpt_recommendation_prompt(mentee, [mentor], top_n=1)

        for field in MENTEE_FIELDS | MENTOR_FIELDS:
            self.assertNotIn(f'"{field}"', prompt)
        self.assertNotIn("DISPLAY_ONLY_MENTEE", prompt)
        self.assertNotIn("DISPLAY_ONLY_MENTOR", prompt)

    def test_pagerank_is_invariant_to_display_only_values(self) -> None:
        mentee = copy.deepcopy(self.mentees[0])
        mentors = copy.deepcopy(self.mentors[:40])
        request = PersonalizedPageRankRequest(
            mentee_id=str(mentee["id"]),
            top_n=5,
        )
        service = PersonalizedPageRankRecommendationService()

        baseline = service.recommend(mentee, mentors, request)

        mentee.update(
            request_scenario="DISPLAY_ONLY_CHANGED",
            help_format="DISPLAY_ONLY_CHANGED",
            urgency="DISPLAY_ONLY_CHANGED",
            motivation="DISPLAY_ONLY_CHANGED",
            expected_result="DISPLAY_ONLY_CHANGED",
        )
        for mentor in mentors:
            mentor["request_scenarios"] = ["DISPLAY_ONLY_CHANGED"]
            mentor["help_formats"] = ["DISPLAY_ONLY_CHANGED"]

        changed = service.recommend(mentee, mentors, request)

        self.assertEqual(
            [item.mentor_id for item in baseline.recommendations],
            [item.mentor_id for item in changed.recommendations],
        )
        self.assertEqual(
            baseline.raw_model_response["top_raw_scores"],
            changed.raw_model_response["top_raw_scores"],
        )


if __name__ == "__main__":
    unittest.main()

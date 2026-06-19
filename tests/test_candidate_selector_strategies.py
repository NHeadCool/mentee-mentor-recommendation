from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.filter_candidate_selector import select_filter_candidates
from app.services.llm_candidate_selector import LLMCandidateSelector


class FilterCandidateSelectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data_dir = PROJECT_ROOT / "data"
        cls.mentees = json.loads((data_dir / "mentees.json").read_text(encoding="utf-8"))
        cls.mentors = json.loads((data_dir / "mentors.json").read_text(encoding="utf-8"))

    def test_each_explicit_mismatch_is_filtered(self) -> None:
        mentee = {
            "language": "ru",
            "budget": "up to 3000",
            "availability": "evening",
            "level": "junior",
            "preferred_format": "video call",
            "skills_to_develop": ["python"],
            "goal": "backend development",
            "target_role": "Backend Developer",
        }
        base_mentor = {
            "id": "good",
            "language": "ru",
            "price": "2000",
            "availability": "evening",
            "works_with_levels": ["junior"],
            "formats": ["video call"],
            "expertise": ["python"],
            "role": "Backend Developer",
            "years_of_experience": 5,
            "profile_completeness": "high",
        }
        mentors = [base_mentor]
        for mentor_id, field, value in (
            ("language", "language", "en"),
            ("budget", "price", "5000"),
            ("availability", "availability", "weekend"),
            ("level", "works_with_levels", ["senior"]),
            ("format", "formats", ["chat"]),
        ):
            mentor = copy.deepcopy(base_mentor)
            mentor["id"] = mentor_id
            mentor[field] = value
            mentors.append(mentor)

        selection = select_filter_candidates(mentee, mentors, limit=20)

        self.assertEqual([mentor["id"] for mentor in selection.mentors], ["good"])
        self.assertEqual(
            selection.rejection_counts,
            {
                "availability": 1,
                "budget": 1,
                "format": 1,
                "language": 1,
                "level": 1,
            },
        )

    def test_missing_values_do_not_create_false_rejection(self) -> None:
        selection = select_filter_candidates(
            mentee={"language": "ru", "budget": "3000"},
            mentors=[{"id": "sparse"}],
            limit=20,
        )
        self.assertEqual([mentor["id"] for mentor in selection.mentors], ["sparse"])

    def test_default_strategy_is_filters_and_limits_candidates(self) -> None:
        selection = LLMCandidateSelector().select(self.mentees[0], self.mentors)

        self.assertEqual(selection.metadata["strategy"], "filters")
        self.assertEqual(
            selection.metadata["algorithm"],
            "hard_filters_and_weighted_scoring",
        )
        self.assertEqual(len(selection.mentors), 20)
        self.assertEqual(selection.metadata["selected_mentors_count"], 20)
        self.assertGreaterEqual(selection.metadata["filtered_mentors_count"], 20)

    def test_display_only_fields_do_not_change_filter_selection(self) -> None:
        mentee = copy.deepcopy(self.mentees[0])
        mentors = copy.deepcopy(self.mentors[:100])

        baseline = LLMCandidateSelector().select(mentee, mentors)

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

        changed = LLMCandidateSelector().select(mentee, mentors)

        self.assertEqual(
            baseline.metadata["selected_mentor_ids"],
            changed.metadata["selected_mentor_ids"],
        )
        self.assertEqual(
            baseline.metadata["top_local_scores"],
            changed.metadata["top_local_scores"],
        )

    def test_pagerank_strategy_remains_available(self) -> None:
        selection = LLMCandidateSelector(strategy="pagerank", limit=5).select(
            self.mentees[0],
            self.mentors[:80],
        )

        self.assertEqual(selection.metadata["strategy"], "pagerank")
        self.assertEqual(selection.metadata["algorithm"], "personalized_pagerank")
        self.assertEqual(len(selection.mentors), 5)
        self.assertIn("ranker", selection.metadata)

    def test_filter_strategy_keeps_control_matches_in_top_20(self) -> None:
        expected_pairs = (
            ("mentee_control_pm_transition", "mentor_control_pm_transition"),
            ("mentee_control_low_info", "mentor_control_career_interview"),
        )
        mentees_by_id = {mentee["id"]: mentee for mentee in self.mentees}

        for mentee_id, mentor_id in expected_pairs:
            selection = LLMCandidateSelector(strategy="filters").select(
                mentees_by_id[mentee_id],
                self.mentors,
            )
            self.assertIn(mentor_id, selection.metadata["selected_mentor_ids"])

    def test_unknown_strategy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LLMCandidateSelector(strategy="unknown")

    def test_empty_pool_is_supported_by_both_strategies(self) -> None:
        for strategy in ("filters", "pagerank"):
            selection = LLMCandidateSelector(strategy=strategy).select({}, [])
            self.assertEqual(selection.mentors, [])
            self.assertEqual(selection.metadata["strategy"], strategy)
            self.assertEqual(selection.metadata["candidate_limit"], 0)


if __name__ == "__main__":
    unittest.main()

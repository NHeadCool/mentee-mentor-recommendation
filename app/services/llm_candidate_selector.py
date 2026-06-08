from dataclasses import dataclass
from typing import Any

from app.models.v1.recommendation import PersonalizedPageRankRequest
from app.services.personalized_pagerank_service import (
    PersonalizedPageRankRecommendationService,
)


DEFAULT_LLM_CANDIDATE_LIMIT = 20


@dataclass(frozen=True)
class CandidateSelectionResult:
    mentors: list[dict[str, Any]]
    metadata: dict[str, Any]


class LLMCandidateSelector:
    def __init__(
        self,
        limit: int = DEFAULT_LLM_CANDIDATE_LIMIT,
        ranker: PersonalizedPageRankRecommendationService | None = None,
    ) -> None:
        self.limit = limit
        self.ranker = ranker or PersonalizedPageRankRecommendationService()

    def select(
        self,
        mentee: dict[str, Any],
        mentors: list[dict[str, Any]],
    ) -> CandidateSelectionResult:
        if not mentors:
            return CandidateSelectionResult(
                mentors=[],
                metadata={
                    "algorithm": "personalized_pagerank",
                    "input_mentors_count": 0,
                    "candidate_limit": 0,
                    "selected_mentors_count": 0,
                    "selected_mentor_ids": [],
                },
            )

        candidate_limit = min(
            max(1, self.limit),
            DEFAULT_LLM_CANDIDATE_LIMIT,
            len(mentors),
        )
        request = PersonalizedPageRankRequest(
            mentee_id=str(mentee.get("id")),
            top_n=candidate_limit,
        )
        ranking = self.ranker.recommend(
            mentee=mentee,
            mentors=mentors,
            request=request,
        )

        mentors_by_id = {str(mentor.get("id")): mentor for mentor in mentors}
        selected_mentors: list[dict[str, Any]] = []

        for recommendation in ranking.recommendations:
            mentor = mentors_by_id.get(recommendation.mentor_id)
            if mentor is not None:
                selected_mentors.append(mentor)

        raw_ranking = ranking.raw_model_response or {}

        return CandidateSelectionResult(
            mentors=selected_mentors,
            metadata={
                "algorithm": "personalized_pagerank",
                "input_mentors_count": len(mentors),
                "candidate_limit": candidate_limit,
                "selected_mentors_count": len(selected_mentors),
                "selected_mentor_ids": [
                    str(mentor.get("id")) for mentor in selected_mentors
                ],
                "ranker": {
                    "iterations": raw_ranking.get("iterations"),
                    "converged": raw_ranking.get("converged"),
                    "damping_factor": raw_ranking.get("damping_factor"),
                    "max_iterations": raw_ranking.get("max_iterations"),
                    "tolerance": raw_ranking.get("tolerance"),
                },
                "top_raw_scores": raw_ranking.get("top_raw_scores", []),
            },
        )

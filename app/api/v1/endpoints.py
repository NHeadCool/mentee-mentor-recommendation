from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.models.v1.recommendation import (
    PersonalizedPageRankRequest,
    RecommendationResponse,
    YandexGPTRecommendationRequest,
)
from app.core.config import settings
from app.services.json_storage import get_mentee_by_id, load_mentees, load_mentors
from app.services.llm_candidate_selector import LLMCandidateSelector
from app.services.personalized_pagerank_service import (
    PersonalizedPageRankRecommendationService,
)
from app.services.prompt_builder import build_yandex_gpt_recommendation_prompt
from app.services.yandex_gpt_service import YandexGPTService


router = APIRouter(prefix="/recommendations", tags=["recommendations"])

templates = Jinja2Templates(directory="app/templates")


@router.get("/ui", response_class=HTMLResponse)
async def recommendations_ui(request: Request) -> HTMLResponse:
    mentees = load_mentees()
    mentors = load_mentors()

    return templates.TemplateResponse(
        name="recommendations.html",
        request=request,
        context={
            "mentees": mentees,
            "mentors": mentors,
        },
    )


@router.post(
    "/yandex-gpt/from-json",
    response_model=RecommendationResponse,
)
async def recommend_from_json_files(
    request: YandexGPTRecommendationRequest,
) -> RecommendationResponse:
    mentee = get_mentee_by_id(request.mentee_id)
    mentors = load_mentors()

    candidate_selection = LLMCandidateSelector(
        strategy=request.candidate_selector or settings.llm_candidate_selector,
    ).select(
        mentee=mentee,
        mentors=mentors,
    )

    prompt = build_yandex_gpt_recommendation_prompt(
        mentee=mentee,
        mentors=candidate_selection.mentors,
        top_n=request.top_n,
    )

    service = YandexGPTService()
    response = await service.get_recommendations(prompt)
    response.raw_model_response = {
        **(response.raw_model_response or {}),
        "candidate_selection": candidate_selection.metadata,
    }
    return response


@router.post(
    "/personalized-pagerank/from-json",
    response_model=RecommendationResponse,
)
async def recommend_with_personalized_pagerank(
    request: PersonalizedPageRankRequest,
) -> RecommendationResponse:
    mentee = get_mentee_by_id(request.mentee_id)
    mentors = load_mentors()

    service = PersonalizedPageRankRecommendationService()
    return service.recommend(
        mentee=mentee,
        mentors=mentors,
        request=request,
    )

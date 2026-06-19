import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


DEFAULT_FILTER_CANDIDATE_LIMIT = 20


@dataclass(frozen=True)
class FilterCandidateSelection:
    mentors: list[dict[str, Any]]
    scores: list[dict[str, Any]]
    filtered_mentors_count: int
    rejection_counts: dict[str, int]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(normalize_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(normalize_text(item) for item in value.values())
    return re.sub(r"\s+", " ", str(value).strip().lower())


def tokenize(value: Any) -> set[str]:
    return set(re.findall(r"[a-zа-яё0-9+#.-]+", normalize_text(value)))


def split_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = re.split(r"[,;/]", str(value))
    return {
        normalize_text(item)
        for item in raw_values
        if normalize_text(item)
    }


def parse_money(value: Any) -> int | None:
    numbers = re.findall(r"\d[\d\s]*", str(value or ""))
    parsed = [int(number.replace(" ", "")) for number in numbers]
    return max(parsed) if parsed else None


def get_fields_text(profile: dict[str, Any], keys: tuple[str, ...]) -> str:
    return " ".join(
        normalize_text(profile.get(key))
        for key in keys
        if profile.get(key)
    )


def jaccard_similarity(left: Any, right: Any) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def overlap_score(required: Any, available: Any) -> float:
    required_tokens = tokenize(required)
    available_tokens = tokenize(available)
    if not required_tokens or not available_tokens:
        return 0.0
    return len(required_tokens & available_tokens) / len(required_tokens)


def number_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def profile_completeness_score(mentor: dict[str, Any]) -> float:
    return {
        "high": 1.0,
        "medium": 0.6,
        "low": 0.2,
    }.get(normalize_text(mentor.get("profile_completeness")), 0.0)


def hard_filter_rejection_reason(
    mentee: dict[str, Any],
    mentor: dict[str, Any],
) -> str | None:
    status = normalize_text(mentor.get("status"))
    if status in {"inactive", "disabled", "blocked"}:
        return "inactive_status"

    mentor_availability = normalize_text(mentor.get("availability"))
    if mentor_availability in {"нет", "no", "unavailable", "недоступен", "не доступен"}:
        return "mentor_unavailable"

    mentee_languages = split_values(mentee.get("language"))
    mentor_languages = split_values(mentor.get("language"))
    if mentee_languages and mentor_languages and not mentee_languages & mentor_languages:
        return "language"

    mentee_budget = parse_money(mentee.get("budget"))
    mentor_price = parse_money(mentor.get("price"))
    if mentee_budget is not None and mentor_price is not None and mentor_price > mentee_budget:
        return "budget"

    mentee_availability = normalize_text(mentee.get("availability"))
    if (
        mentee_availability
        and mentor_availability
        and mentee_availability != mentor_availability
    ):
        return "availability"

    mentee_level = normalize_text(mentee.get("level"))
    mentor_levels = split_values(mentor.get("works_with_levels"))
    if mentee_level and mentor_levels and mentee_level not in mentor_levels:
        return "level"

    preferred_format = normalize_text(mentee.get("preferred_format"))
    mentor_formats = split_values(mentor.get("formats"))
    if preferred_format and mentor_formats and preferred_format not in mentor_formats:
        return "format"

    return None


def score_mentor(mentee: dict[str, Any], mentor: dict[str, Any]) -> float:
    mentee_skills = get_fields_text(mentee, ("skills_to_develop",))
    mentor_skills = get_fields_text(mentor, ("expertise",))

    mentee_goal = get_fields_text(
        mentee,
        ("goal", "problem_description", "target_role", "preferred_mentor_background"),
    )
    mentor_help = get_fields_text(
        mentor,
        ("role", "expertise", "bio", "test_role", "mentoring_style"),
    )

    mentee_domain = get_fields_text(
        mentee,
        ("current_role", "target_role", "industry"),
    )
    mentor_domain = get_fields_text(mentor, ("role", "industries"))

    mentee_format = get_fields_text(
        mentee,
        ("preferred_format", "timezone", "availability"),
    )
    mentor_format = get_fields_text(
        mentor,
        ("formats", "timezone", "availability"),
    )

    skills_score = overlap_score(mentee_skills, mentor_skills)
    goals_score = jaccard_similarity(mentee_goal, mentor_help)
    domain_score = jaccard_similarity(mentee_domain, mentor_domain)
    format_score = jaccard_similarity(mentee_format, mentor_format)
    return (
        0.45 * skills_score
        + 0.35 * goals_score
        + 0.15 * domain_score
        + 0.05 * format_score
    )


def select_filter_candidates(
    mentee: dict[str, Any],
    mentors: list[dict[str, Any]],
    limit: int = DEFAULT_FILTER_CANDIDATE_LIMIT,
) -> FilterCandidateSelection:
    rejection_counts: Counter[str] = Counter()
    scored: list[tuple[dict[str, Any], float]] = []

    for mentor in mentors:
        rejection_reason = hard_filter_rejection_reason(mentee, mentor)
        if rejection_reason is not None:
            rejection_counts[rejection_reason] += 1
            continue
        scored.append((mentor, score_mentor(mentee, mentor)))

    scored.sort(
        key=lambda item: (
            -item[1],
            -number_value(item[0].get("years_of_experience")),
            -profile_completeness_score(item[0]),
            -number_value(item[0].get("rating")),
            -number_value(item[0].get("sessions_count")),
            str(item[0].get("id") or ""),
        )
    )
    selected = scored[: max(0, limit)]

    return FilterCandidateSelection(
        mentors=[mentor for mentor, _ in selected],
        scores=[
            {
                "mentor_id": str(mentor.get("id")),
                "local_score": round(score, 6),
            }
            for mentor, score in selected
        ],
        filtered_mentors_count=len(scored),
        rejection_counts=dict(sorted(rejection_counts.items())),
    )

import re
from typing import Any, Dict, List, Tuple


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    if isinstance(text, list):
        return " ".join(normalize_text(x) for x in text)

    if isinstance(text, dict):
        return " ".join(normalize_text(v) for v in text.values())

    return str(text).lower().strip()


def tokenize(text: Any) -> set:
    text = normalize_text(text)
    return set(re.findall(r"[a-zа-яё0-9+#.-]+", text))


def get_fields_text(profile: Dict[str, Any], keys: List[str]) -> str:
    return " ".join(
        normalize_text(profile.get(key))
        for key in keys
        if profile.get(key)
    )


def jaccard_similarity(a: Any, b: Any) -> float:
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def overlap_score(a: Any, b: Any) -> float:
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a)


def get_number(profile: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for key in keys:
        value = profile.get(key)

        try:
            return float(value)
        except (ValueError, TypeError):
            continue

    return default


def is_hard_filter_passed(mentee: Dict[str, Any], mentor: Dict[str, Any]) -> bool:
    status = normalize_text(mentor.get("status", ""))

    if status in {"inactive", "disabled", "blocked"}:
        return False

    availability = normalize_text(mentor.get("availability", ""))

    if availability in {"нет", "no", "unavailable", "недоступен"}:
        return False

    mentee_languages = tokenize(mentee.get("languages") or mentee.get("language"))
    mentor_languages = tokenize(mentor.get("languages") or mentor.get("language"))

    if mentee_languages and mentor_languages:
        if not mentee_languages & mentor_languages:
            return False

    return True


def profile_completeness(profile: Dict[str, Any]) -> float:
    important_keys = [
        "skills",
        "topics",
        "interests",
        "description",
        "about",
        "experience",
        "role",
        "industry",
        "availability",
        "language",
        "languages",
    ]

    filled = 0

    for key in important_keys:
        value = profile.get(key)

        if value not in [None, "", [], {}]:
            filled += 1

    return filled / len(important_keys)


def score_mentor(mentee: Dict[str, Any], mentor: Dict[str, Any]) -> float:
    mentee_skills = get_fields_text(
        mentee,
        ["skills", "hard_skills", "soft_skills", "stack", "topics", "interests"]
    )

    mentor_skills = get_fields_text(
        mentor,
        ["skills", "hard_skills", "soft_skills", "stack", "topics", "interests", "expertise"]
    )

    mentee_goals = get_fields_text(
        mentee,
        ["goals", "career_goals", "request", "expectations", "description", "about"]
    )

    mentor_topics = get_fields_text(
        mentor,
        ["description", "about", "mentoring_topics", "expertise", "help_with"]
    )

    mentee_domain = get_fields_text(
        mentee,
        ["role", "profession", "industry", "specialization", "direction"]
    )

    mentor_domain = get_fields_text(
        mentor,
        ["role", "profession", "industry", "specialization", "direction"]
    )

    mentee_format = get_fields_text(
        mentee,
        ["format", "communication_format", "timezone", "languages", "language"]
    )

    mentor_format = get_fields_text(
        mentor,
        ["format", "communication_format", "timezone", "languages", "language", "availability"]
    )

    skills_score = overlap_score(mentee_skills, mentor_skills)
    goals_score = jaccard_similarity(mentee_goals, mentor_topics)
    domain_score = jaccard_similarity(mentee_domain, mentor_domain)
    format_score = jaccard_similarity(mentee_format, mentor_format)

    experience_years = get_number(
        mentor,
        ["experience_years", "years_experience", "work_experience_years"],
        default=0
    )

    experience_score = min(experience_years / 10, 1.0)
    completeness_score = profile_completeness(mentor)

    return (
        0.35 * skills_score +
        0.25 * goals_score +
        0.15 * domain_score +
        0.10 * format_score +
        0.10 * experience_score +
        0.05 * completeness_score
    )


def mentor_similarity(mentor_a: Dict[str, Any], mentor_b: Dict[str, Any]) -> float:
    keys = [
        "skills",
        "topics",
        "interests",
        "expertise",
        "role",
        "industry",
        "specialization",
        "description",
        "about",
    ]

    text_a = get_fields_text(mentor_a, keys)
    text_b = get_fields_text(mentor_b, keys)

    return jaccard_similarity(text_a, text_b)


def select_diverse_top_k(
    scored_mentors: List[Tuple[Dict[str, Any], float]],
    k: int = 30,
    lambda_relevance: float = 0.75
) -> List[Tuple[Dict[str, Any], float]]:

    selected = []
    candidates = scored_mentors.copy()

    while candidates and len(selected) < k:
        best_candidate = None
        best_mmr_score = float("-inf")

        for mentor, relevance_score in candidates:
            if not selected:
                diversity_penalty = 0.0
            else:
                diversity_penalty = max(
                    mentor_similarity(mentor, selected_mentor)
                    for selected_mentor, _ in selected
                )

            mmr_score = (
                lambda_relevance * relevance_score -
                (1 - lambda_relevance) * diversity_penalty
            )

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate = (mentor, relevance_score)

        selected.append(best_candidate)
        candidates.remove(best_candidate)

    return selected


def select_mentors_for_llm(
    mentee: Dict[str, Any],
    mentors: List[Dict[str, Any]],
    k: int = 30,
    preselect_n: int = 150
) -> List[Dict[str, Any]]:

    scored = []

    for mentor in mentors:
        if not is_hard_filter_passed(mentee, mentor):
            continue

        score = score_mentor(mentee, mentor)
        scored.append((mentor, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    preselected = scored[:preselect_n]

    selected = select_diverse_top_k(
        scored_mentors=preselected,
        k=k,
        lambda_relevance=0.75
    )

    result = []

    for mentor, score in selected:
        mentor_copy = mentor.copy()
        mentor_copy["local_score"] = round(score, 4)
        result.append(mentor_copy)

    return result
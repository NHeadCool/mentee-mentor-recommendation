#PRP alg
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.models.v1.recommendation import (
    PersonalizedPageRankRequest,
    RecommendationResponse,
    RecommendedMentor,
)


Graph = dict[str, dict[str, float]]


@dataclass(frozen=True)
class Feature:
    node: str
    label: str
    weight: float


class PersonalizedPageRankRecommendationService:
    def recommend(
        self,
        mentee: dict[str, Any],
        mentors: list[dict[str, Any]],
        request: PersonalizedPageRankRequest,
    ) -> RecommendationResponse:
        graph: Graph = defaultdict(dict)
        preference: dict[str, float] = defaultdict(float)
        mentor_nodes: dict[str, dict[str, Any]] = {}
        known_terms = self._build_known_terms(mentors)

        mentee_node = self._mentee_node(mentee)
        mentee_budget = self._parse_money(mentee.get("budget"))
        mentee_features = self._extract_mentee_features(
            mentee=mentee,
            mentee_budget=mentee_budget,
            known_terms=known_terms,
        )

        self._ensure_node(graph, mentee_node)
        preference[mentee_node] += 1.0

        for feature in mentee_features:
            self._add_edge(graph, mentee_node, feature.node, feature.weight)
            preference[feature.node] += feature.weight

        for feature in self._default_quality_preferences():
            self._ensure_node(graph, feature.node)
            preference[feature.node] += feature.weight

        for mentor in mentors:
            mentor_node = self._mentor_node(mentor)
            mentor_nodes[mentor_node] = mentor
            self._ensure_node(graph, mentor_node)

            mentor_features = self._extract_mentor_features(
                mentor=mentor,
                mentee_budget=mentee_budget,
                known_terms=known_terms,
            )

            for feature in mentor_features:
                self._add_edge(graph, mentor_node, feature.node, feature.weight)

        ranks, iterations, converged = self._run_pagerank(
            graph=graph,
            preference=preference,
            damping_factor=request.damping_factor,
            max_iterations=request.max_iterations,
            tolerance=request.tolerance,
        )

        ranked_mentors = sorted(
            mentor_nodes.values(),
            key=lambda mentor: (
                ranks.get(self._mentor_node(mentor), 0.0),
                self._rating_value(mentor),
                self._sessions_value(mentor),
            ),
            reverse=True,
        )

        selected_mentors = ranked_mentors[: request.top_n]
        selected_scores = [
            ranks.get(self._mentor_node(mentor), 0.0)
            for mentor in selected_mentors
        ]
        max_score = max(selected_scores) if selected_scores else 0.0

        recommendations: list[RecommendedMentor] = []

        for index, mentor in enumerate(selected_mentors, start=1):
            raw_score = ranks.get(self._mentor_node(mentor), 0.0)
            normalized_score = self._normalize_score(raw_score, max_score)
            matched_factors, possible_risks = self._build_match_report(
                mentee=mentee,
                mentor=mentor,
                mentee_budget=mentee_budget,
            )

            recommendations.append(
                RecommendedMentor(
                    mentor_id=str(mentor.get("id")),
                    mentor_name=str(mentor.get("name") or "Без имени"),
                    rank=index,
                    score=normalized_score,
                    explanation=self._build_explanation(
                        mentor=mentor,
                        matched_factors=matched_factors,
                        possible_risks=possible_risks,
                    ),
                    matched_factors=matched_factors,
                    possible_risks=possible_risks,
                )
            )

        return RecommendationResponse(
            mentee_id=str(mentee.get("id")),
            recommendation_logic=(
                "Personalized PageRank строит граф из выбранного менти, менторов "
                "и общих признаков: целевая роль, навыки, индустрия, формат, язык, "
                "уровень, бюджет и признаки надежности. Телепортация персонализирована "
                "признаками менти, поэтому выше поднимаются менторы, связанные с теми "
                "же узлами графа."
            ),
            recommendations=recommendations,
            general_explanation=(
                f"Алгоритм сошелся: {'да' if converged else 'нет'}; "
                f"итераций: {iterations}; демпфирование: {request.damping_factor}. "
                "Оценка нормирована относительно лучшего найденного ментора в top-N."
            ),
            raw_model_response={
                "algorithm": "personalized_pagerank",
                "damping_factor": request.damping_factor,
                "max_iterations": request.max_iterations,
                "tolerance": request.tolerance,
                "iterations": iterations,
                "converged": converged,
                "graph": {
                    "nodes_count": len(graph),
                    "edges_count": sum(len(edges) for edges in graph.values()) // 2,
                    "mentors_count": len(mentors),
                    "mentee_features_count": len(mentee_features),
                },
                "top_raw_scores": [
                    {
                        "mentor_id": mentor.get("id"),
                        "raw_pagerank": ranks.get(self._mentor_node(mentor), 0.0),
                    }
                    for mentor in selected_mentors
                ],
            },
        )

    @staticmethod
    def _run_pagerank(
        graph: Graph,
        preference: dict[str, float],
        damping_factor: float,
        max_iterations: int,
        tolerance: float,
    ) -> tuple[dict[str, float], int, bool]:
        nodes = set(graph)
        nodes.update(preference)

        if not nodes:
            return {}, 0, True

        preference_sum = sum(preference.values())
        if preference_sum <= 0:
            normalized_preference = {node: 1.0 / len(nodes) for node in nodes}
        else:
            normalized_preference = {
                node: preference.get(node, 0.0) / preference_sum
                for node in nodes
            }

        ranks = normalized_preference.copy()

        for iteration in range(1, max_iterations + 1):
            new_ranks = {
                node: (1.0 - damping_factor) * normalized_preference[node]
                for node in nodes
            }

            dangling_mass = sum(
                ranks[node]
                for node in nodes
                if not graph.get(node)
            )

            for node in nodes:
                new_ranks[node] += (
                    damping_factor
                    * dangling_mass
                    * normalized_preference[node]
                )

            for source, targets in graph.items():
                total_weight = sum(targets.values())
                if total_weight <= 0:
                    continue

                for target, weight in targets.items():
                    new_ranks[target] += (
                        damping_factor
                        * ranks.get(source, 0.0)
                        * weight
                        / total_weight
                    )

            delta = sum(
                abs(new_ranks[node] - ranks.get(node, 0.0))
                for node in nodes
            )
            ranks = new_ranks

            if delta < tolerance:
                return ranks, iteration, True

        return ranks, max_iterations, False

    def _extract_mentee_features(
        self,
        mentee: dict[str, Any],
        mentee_budget: int | None,
        known_terms: dict[str, tuple[str, str]],
    ) -> list[Feature]:
        features: list[Feature] = []

        self._add_profile_value(features, "target_role", mentee.get("target_role"), 5.0)
        self._add_profile_value(
            features,
            "role_family",
            self._role_family(mentee.get("target_role")),
            6.0,
        )
        self._add_profile_value(features, "current_role", mentee.get("current_role"), 1.5)
        self._add_profile_value(features, "industry", mentee.get("industry"), 3.0)
        self._add_profile_value(features, "level", mentee.get("level"), 2.5)
        self._add_profile_value(features, "format", mentee.get("preferred_format"), 2.5)
        self._add_profile_values(features, "language", self._split_values(mentee.get("language")), 2.0)
        self._add_profile_value(features, "timezone", mentee.get("timezone"), 1.0)
        self._add_profile_value(features, "availability", mentee.get("availability"), 1.5)

        for skill in self._as_list(mentee.get("skills_to_develop")):
            self._add_profile_value(features, "skill", skill, 5.0)

        text_parts = [
            mentee.get("goal"),
            mentee.get("problem_description"),
            mentee.get("preferred_mentor_background"),
        ]
        features.extend(self._extract_text_features(text_parts, known_terms, 2.8))

        if mentee_budget is not None:
            self._add_profile_value(features, "budget", "fits_budget", 2.0)

        return self._merge_features(features)

    def _extract_mentor_features(
        self,
        mentor: dict[str, Any],
        mentee_budget: int | None,
        known_terms: dict[str, tuple[str, str]],
    ) -> list[Feature]:
        features: list[Feature] = []

        self._add_profile_value(features, "mentor_role", mentor.get("role"), 4.5)
        self._add_profile_value(
            features,
            "role_family",
            self._role_family(mentor.get("role")),
            6.0,
        )
        self._add_profile_value(features, "mentor_level", mentor.get("level"), 1.0)
        self._add_profile_values(features, "industry", self._as_list(mentor.get("industries")), 3.0)
        self._add_profile_values(features, "format", self._as_list(mentor.get("formats")), 2.5)
        self._add_profile_values(features, "language", self._split_values(mentor.get("language")), 2.0)
        self._add_profile_value(features, "timezone", mentor.get("timezone"), 1.0)
        self._add_profile_value(features, "availability", mentor.get("availability"), 1.5)
        self._add_profile_values(features, "level", self._as_list(mentor.get("works_with_levels")), 2.5)

        for skill in self._as_list(mentor.get("expertise")):
            self._add_profile_value(features, "skill", skill, 5.0)

        text_parts = [
            mentor.get("bio"),
            mentor.get("test_role"),
            mentor.get("mentoring_style"),
        ]
        features.extend(self._extract_text_features(text_parts, known_terms, 1.8))

        if self._rating_value(mentor) >= 4.8:
            self._add_profile_value(features, "quality", "high_rating", 1.2)
        if self._sessions_value(mentor) >= 100:
            self._add_profile_value(features, "quality", "many_sessions", 1.0)
        if mentor.get("profile_completeness") == "high":
            self._add_profile_value(features, "quality", "complete_profile", 0.8)

        reliability = mentor.get("reliability") or {}
        if reliability.get("response_rate") == "высокий":
            self._add_profile_value(features, "quality", "high_response_rate", 0.8)
        if reliability.get("cancellation_rate") == "низкий":
            self._add_profile_value(features, "quality", "low_cancellation_rate", 0.6)

        mentor_price = self._parse_money(mentor.get("price"))
        if mentee_budget is not None and mentor_price is not None and mentor_price <= mentee_budget:
            self._add_profile_value(features, "budget", "fits_budget", 2.0)

        return self._merge_features(features)

    @staticmethod
    def _default_quality_preferences() -> list[Feature]:
        return [
            Feature("feature:quality:high_rating", "высокий рейтинг", 0.7),
            Feature("feature:quality:many_sessions", "много проведенных сессий", 0.5),
            Feature("feature:quality:complete_profile", "полный профиль", 0.4),
            Feature("feature:quality:high_response_rate", "высокая скорость ответа", 0.4),
            Feature("feature:quality:low_cancellation_rate", "низкая доля отмен", 0.3),
        ]

    def _build_match_report(
        self,
        mentee: dict[str, Any],
        mentor: dict[str, Any],
        mentee_budget: int | None,
    ) -> tuple[list[str], list[str]]:
        factors: list[str] = []
        risks: list[str] = []

        skill_matches = self._overlap(
            self._as_list(mentee.get("skills_to_develop")),
            self._as_list(mentor.get("expertise")),
        )
        if skill_matches:
            factors.append("совпали навыки: " + ", ".join(skill_matches[:5]))

        mentee_role_family = self._role_family(mentee.get("target_role"))
        mentor_role_family = self._role_family(mentor.get("role"))
        if mentee_role_family and mentee_role_family == mentor_role_family:
            factors.append("роль ментора совпадает с целевой ролью менти")
        elif self._role_mentions_text(mentee_role_family, mentor):
            factors.append("целевая роль менти встречается в описании опыта ментора")

        if self._is_transition(mentee) and self._mentor_has_signal(mentor, "career transition"):
            factors.append("есть релевантность к карьерному переходу")

        industry_matches = self._overlap(
            [mentee.get("industry")],
            self._as_list(mentor.get("industries")),
        )
        if industry_matches:
            factors.append("совпала индустрия: " + ", ".join(industry_matches))

        format_matches = self._overlap(
            [mentee.get("preferred_format")],
            self._as_list(mentor.get("formats")),
        )
        if format_matches:
            factors.append("совпал формат работы: " + ", ".join(format_matches))

        language_matches = self._overlap(
            self._split_values(mentee.get("language")),
            self._split_values(mentor.get("language")),
        )
        if language_matches:
            factors.append("совпал язык: " + ", ".join(language_matches))
        elif mentee.get("language"):
            risks.append("язык ментора не совпал или не указан")

        if mentee.get("timezone") and mentee.get("timezone") == mentor.get("timezone"):
            factors.append("совпал часовой пояс")
        elif mentee.get("timezone") and not mentor.get("timezone"):
            risks.append("у ментора не указан часовой пояс")

        if mentee.get("availability") and mentee.get("availability") == mentor.get("availability"):
            factors.append("совпала доступность")

        if mentee.get("level") and mentee.get("level") in self._as_list(mentor.get("works_with_levels")):
            factors.append("ментор работает с уровнем менти")

        mentor_price = self._parse_money(mentor.get("price"))
        if mentee_budget is not None and mentor_price is not None:
            if mentor_price <= mentee_budget:
                factors.append("стоимость укладывается в бюджет")
            else:
                risks.append("стоимость выше бюджета менти")
        elif mentee_budget is not None:
            risks.append("у ментора не указана стоимость")

        rating = self._rating_value(mentor)
        if rating >= 4.8:
            factors.append(f"высокий рейтинг: {rating:g}")

        sessions = self._sessions_value(mentor)
        if sessions >= 100:
            factors.append(f"много проведенных сессий: {sessions}")

        if mentor.get("profile_completeness") in {"low", "medium"}:
            risks.append(
                "профиль ментора заполнен неполно"
                if mentor.get("profile_completeness") == "medium"
                else "профиль ментора заполнен слабо"
            )

        reliability = mentor.get("reliability") or {}
        if reliability.get("response_rate") == "низкий":
            risks.append("у ментора низкая скорость ответа")
        if reliability.get("cancellation_rate") == "высокий":
            risks.append("у ментора высокая доля отмен")
        if reliability.get("no_show_rate") == "высокий":
            risks.append("у ментора высокий no-show риск")

        not_ready = self._as_list(mentor.get("not_ready_to_help_with"))
        if self._is_transition(mentee) and "смена профессии" in not_ready:
            risks.append("ментор указал, что не готов помогать со сменой профессии")

        if mentee.get("profile_completeness") == "low":
            risks.append("у менти мало структурированных данных, рекомендация менее уверенная")

        if not factors:
            factors.append("PageRank нашел косвенную близость через общие признаки графа")

        return factors[:8], risks[:6]

    @staticmethod
    def _build_explanation(
        mentor: dict[str, Any],
        matched_factors: list[str],
        possible_risks: list[str],
    ) -> str:
        mentor_name = mentor.get("name") or "Ментор"
        base = (
            f"{mentor_name} высоко поднялся в personalized PageRank за счет связей "
            f"с персональными признаками выбранного менти: {', '.join(matched_factors[:3])}."
        )
        if possible_risks:
            return base + " Перед выбором стоит проверить риск: " + possible_risks[0] + "."
        return base

    def _build_known_terms(self, mentors: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
        terms: dict[str, tuple[str, str]] = {}

        for mentor in mentors:
            for skill in self._as_list(mentor.get("expertise")):
                self._register_known_term(terms, "skill", skill)
            for role in [mentor.get("role"), self._role_family(mentor.get("role"))]:
                self._register_known_term(terms, "role_family", role)
            for value in self._as_list(mentor.get("formats")):
                self._register_known_term(terms, "format", value)

        for phrase, target in {
            "собесед": ("skill", "interview preparation"),
            "интерв": ("skill", "interview preparation"),
            "резюме": ("skill", "resume"),
            "оффер": ("skill", "job search"),
            "ваканс": ("skill", "job search"),
            "портфоли": ("skill", "portfolio review"),
            "переход": ("skill", "career transition"),
            "перейти": ("skill", "career transition"),
            "roadmap": ("skill", "roadmap"),
            "исследован": ("skill", "user research"),
        }.items():
            terms[phrase] = target

        return terms

    def _extract_text_features(
        self,
        text_parts: list[Any],
        known_terms: dict[str, tuple[str, str]],
        weight: float,
    ) -> list[Feature]:
        text = self._normalize(" ".join(str(part) for part in text_parts if part))
        if not text:
            return []

        features: list[Feature] = []
        for term, (kind, value) in known_terms.items():
            if term and term in text:
                self._add_profile_value(features, kind, value, weight)

        return features

    @staticmethod
    def _add_edge(graph: Graph, source: str, target: str, weight: float) -> None:
        if weight <= 0:
            return

        graph[source][target] = graph[source].get(target, 0.0) + weight
        graph[target][source] = graph[target].get(source, 0.0) + weight

    @staticmethod
    def _ensure_node(graph: Graph, node: str) -> None:
        graph.setdefault(node, {})

    def _add_profile_value(
        self,
        features: list[Feature],
        kind: str,
        value: Any,
        weight: float,
    ) -> None:
        normalized_value = self._normalize(value)
        if not normalized_value:
            return

        features.append(
            Feature(
                node=f"feature:{kind}:{normalized_value}",
                label=str(value).strip(),
                weight=weight,
            )
        )

    def _add_profile_values(
        self,
        features: list[Feature],
        kind: str,
        values: list[Any],
        weight: float,
    ) -> None:
        for value in values:
            self._add_profile_value(features, kind, value, weight)

    @staticmethod
    def _merge_features(features: list[Feature]) -> list[Feature]:
        merged: dict[str, Feature] = {}

        for feature in features:
            if feature.node in merged:
                existing = merged[feature.node]
                merged[feature.node] = Feature(
                    node=feature.node,
                    label=existing.label,
                    weight=existing.weight + feature.weight,
                )
            else:
                merged[feature.node] = feature

        return list(merged.values())

    @staticmethod
    def _register_known_term(
        terms: dict[str, tuple[str, str]],
        kind: str,
        value: Any,
    ) -> None:
        normalized_value = PersonalizedPageRankRecommendationService._normalize(value)
        if normalized_value:
            terms[normalized_value] = (kind, str(value))

    @staticmethod
    def _normalize_score(raw_score: float, max_score: float) -> float:
        if max_score <= 0:
            return 0.0
        return round(min(100.0, max(0.0, raw_score / max_score * 100.0)), 2)

    @staticmethod
    def _normalize(value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value).strip().lower())

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _split_values(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            raw_values = value
        else:
            raw_values = re.split(r"[,;/]|\s+и\s+", str(value))
        return [str(item).strip() for item in raw_values if str(item).strip()]

    @staticmethod
    def _parse_money(value: Any) -> int | None:
        if value is None:
            return None

        numbers = re.findall(r"\d[\d\s]*", str(value))
        parsed = [int(number.replace(" ", "")) for number in numbers]
        return max(parsed) if parsed else None

    @staticmethod
    def _role_family(value: Any) -> str:
        normalized = PersonalizedPageRankRecommendationService._normalize(value)
        if not normalized:
            return ""

        replacements = [
            (r"\bsenior\b", ""),
            (r"\bmiddle\b", ""),
            (r"\bjunior\b", ""),
            (r"\blead\b", ""),
            (r"\bprincipal\b", ""),
            (r"\bstaff\b", ""),
            (r"\bhead of\b", ""),
        ]
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)

        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _overlap(left: list[Any], right: list[Any]) -> list[str]:
        right_by_normalized = {
            PersonalizedPageRankRecommendationService._normalize(item): str(item)
            for item in right
            if PersonalizedPageRankRecommendationService._normalize(item)
        }

        matches: list[str] = []
        for item in left:
            normalized = PersonalizedPageRankRecommendationService._normalize(item)
            if normalized and normalized in right_by_normalized:
                matches.append(right_by_normalized[normalized])

        return matches

    @staticmethod
    def _rating_value(mentor: dict[str, Any]) -> float:
        try:
            return float(mentor.get("rating") or 0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _sessions_value(mentor: dict[str, Any]) -> int:
        try:
            return int(mentor.get("sessions_count") or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _is_transition(mentee: dict[str, Any]) -> bool:
        current_role = PersonalizedPageRankRecommendationService._role_family(
            mentee.get("current_role")
        )
        target_role = PersonalizedPageRankRecommendationService._role_family(
            mentee.get("target_role")
        )
        return bool(current_role and target_role and current_role != target_role)

    def _mentor_has_signal(self, mentor: dict[str, Any], signal: str) -> bool:
        signal_normalized = self._normalize(signal)
        text = self._normalize(
            " ".join(
                [
                    str(mentor.get("role") or ""),
                    str(mentor.get("bio") or ""),
                    str(mentor.get("test_role") or ""),
                    " ".join(self._as_list(mentor.get("expertise"))),
                ]
            )
        )
        return signal_normalized in text

    def _role_mentions_text(self, role_family: str, mentor: dict[str, Any]) -> bool:
        if not role_family:
            return False

        text = self._normalize(
            " ".join(
                [
                    str(mentor.get("role") or ""),
                    str(mentor.get("bio") or ""),
                    str(mentor.get("test_role") or ""),
                    " ".join(self._as_list(mentor.get("expertise"))),
                ]
            )
        )
        return role_family in text

    @staticmethod
    def _mentee_node(mentee: dict[str, Any]) -> str:
        return f"mentee:{mentee.get('id')}"

    @staticmethod
    def _mentor_node(mentor: dict[str, Any]) -> str:
        return f"mentor:{mentor.get('id')}"

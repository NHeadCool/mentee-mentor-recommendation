import json
from typing import Any


DISPLAY_ONLY_MENTEE_FIELDS = {
    "request_scenario",
    "help_format",
    "urgency",
    "motivation",
    "expected_result",
}
DISPLAY_ONLY_MENTOR_FIELDS = {
    "request_scenarios",
    "help_formats",
}


def _without_fields(profile: dict[str, Any], excluded_fields: set[str]) -> dict[str, Any]:
    return {
        key: value
        for key, value in profile.items()
        if key not in excluded_fields
    }


def build_yandex_gpt_recommendation_prompt(
    mentee: dict[str, Any],
    mentors: list[dict[str, Any]],
    top_n: int,
) -> str:
    algorithm_mentee = _without_fields(mentee, DISPLAY_ONLY_MENTEE_FIELDS)
    algorithm_mentors = [
        _without_fields(mentor, DISPLAY_ONLY_MENTOR_FIELDS)
        for mentor in mentors
    ]

    return f"""
Подбери top-{top_n} менторов для одного менти.

У тебя есть предварительно отобранный список кандидатов. Выбирай только менторов из этого списка.
Ты сам определяешь критерии рекомендации на основе данных профиля.
Учитывай цель, роль, уровень, навыки, описание проблемы, формат, язык, часовой пояс, доступность, бюджет, опыт ментора, экспертизу, отзывы и полноту профиля.

Верни только JSON строго такого вида:
{{
  "mentee_id": "{mentee.get("id")}",
  "recommendation_logic": "строка с описанием логики подбора",
  "recommendations": [
    {{
      "mentor_id": "string",
      "mentor_name": "string",
      "rank": 1,
      "score": 95,
      "explanation": "почему этот ментор подходит",
      "matched_factors": ["совпавший фактор 1", "совпавший фактор 2"],
      "possible_risks": ["риск или ограничение"]
    }}
  ],
  "general_explanation": "общий вывод"
}}

Правила:
- Не добавляй markdown.
- Не добавляй текст вне JSON.
- Не придумывай факты, которых нет в данных.
- Если данных мало, укажи это в possible_risks.
- recommendations должен содержать не больше {top_n} объектов.
- mentor_id и mentor_name бери только из списка кандидатов ниже.
- Не оборачивай JSON в markdown-блоки ```json или ```.

Менти:
{json.dumps(algorithm_mentee, ensure_ascii=False, indent=2)}

Кандидаты-менторы:
{json.dumps(algorithm_mentors, ensure_ascii=False, indent=2)}
""".strip()

import json
from typing import Any


def build_yandex_gpt_recommendation_prompt(
    mentee: dict[str, Any],
    mentors: list[dict[str, Any]],
    top_n: int,
) -> str:
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
{json.dumps(mentee, ensure_ascii=False, indent=2)}

Кандидаты-менторы:
{json.dumps(mentors, ensure_ascii=False, indent=2)}
""".strip()

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent
MENTEES_PATH = DATA_DIR / "mentees.json"
MENTORS_PATH = DATA_DIR / "mentors.json"

MENTEE_FIELDS = (
    "request_scenario",
    "help_format",
    "urgency",
    "motivation",
    "expected_result",
)
MENTOR_FIELDS = (
    "request_scenarios",
    "help_formats",
)

REQUEST_SCENARIOS = (
    "онбординг",
    "решение рабочей задачи",
    "развитие компетенций",
    "подготовка к переходу роли",
    "подготовка к собеседованию",
    "проектное сопровождение",
)
HELP_FORMATS = (
    "прямой ответ и конкретные рекомендации",
    "сопровождение через вопросы",
    "работа через документацию",
    "проектное сопровождение с обратной связью",
)
URGENCY_LEVELS = (
    "высокая: результат нужен в ближайшие дни",
    "средняя: результат нужен в течение месяца",
    "низкая: жесткого срока нет",
)
MOTIVATION_LEVELS = (
    "высокая: готов регулярно работать между встречами",
    "средняя: готов выполнять отдельные рекомендации",
    "не указана",
)


def stable_index(profile_id: Any, salt: str, modulo: int) -> int:
    raw = f"{profile_id}:{salt}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % modulo


def profile_text(profile: dict[str, Any], fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    for field in fields:
        value = profile.get(field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts).lower()


def infer_request_scenario(mentee: dict[str, Any]) -> str:
    intent_text = profile_text(
        mentee,
        ("goal",),
    )
    primary_text = profile_text(
        mentee,
        ("goal", "problem_description"),
    )
    full_text = " ".join(
        part
        for part in (
            primary_text,
            str(mentee.get("expected_result") or "").lower(),
        )
        if part
    )
    if re.search(r"онборд|адаптац", intent_text):
        return "онбординг"
    if re.search(r"собесед|интервью", intent_text):
        return "подготовка к собеседованию"
    current_role = str(mentee.get("current_role") or "").strip().lower()
    target_role = str(mentee.get("target_role") or "").strip().lower()
    if (current_role and target_role and current_role != target_role) or re.search(
        r"переход|перейти|сменить профес", intent_text
    ):
        return "подготовка к переходу роли"
    if re.search(r"собесед|интервью|резюме|оффер", full_text):
        return "подготовка к собеседованию"
    if re.search(r"рабоч|задач|ошибк|почин|решить", full_text):
        return "решение рабочей задачи"
    if re.search(r"проект|портфолио|roadmap", full_text):
        return "проектное сопровождение"
    return "развитие компетенций"


def infer_help_format(mentee: dict[str, Any]) -> str:
    text = profile_text(
        mentee,
        ("communication_preferences", "preferred_format", "problem_description"),
    )
    if re.search(r"прям|конкрет|быстр|готов.*ответ", text):
        return HELP_FORMATS[0]
    if re.search(r"вопрос|коуч|мягк|самостоят", text):
        return HELP_FORMATS[1]
    if re.search(r"документ|материал|шаблон", text):
        return HELP_FORMATS[2]
    if re.search(r"проект|практи|домаш|ревью|регуляр", text):
        return HELP_FORMATS[3]
    return HELP_FORMATS[stable_index(mentee.get("id"), "help-format", len(HELP_FORMATS))]


def inferred_expected_result(mentee: dict[str, Any]) -> str:
    goal = str(mentee.get("goal") or "").strip()
    target_role = str(mentee.get("target_role") or "").strip()
    if goal:
        return f"Получить понятный план действий и следующий шаг по цели: {goal}."
    if target_role:
        return f"Получить план подготовки и дальнейших действий для роли {target_role}."
    return "Получить практический план действий по запросу менти."


def enrich_mentee(mentee: dict[str, Any]) -> dict[str, Any]:
    completeness = mentee.get("profile_completeness")
    profile_id = mentee.get("id")

    values: dict[str, Any] = {
        "request_scenario": None,
        "help_format": None,
        "urgency": None,
        "motivation": None,
        "expected_result": mentee.get("expected_result"),
    }

    if completeness == "high":
        values.update(
            request_scenario=infer_request_scenario(mentee),
            help_format=infer_help_format(mentee),
            urgency=URGENCY_LEVELS[stable_index(profile_id, "urgency", len(URGENCY_LEVELS))],
            motivation=MOTIVATION_LEVELS[stable_index(profile_id, "motivation", 2)],
            expected_result=mentee.get("expected_result") or inferred_expected_result(mentee),
        )
    elif completeness == "medium":
        values["request_scenario"] = infer_request_scenario(mentee)
        if stable_index(profile_id, "medium-help-format", 4) != 0:
            values["help_format"] = infer_help_format(mentee)
        if stable_index(profile_id, "medium-urgency", 2) == 0:
            values["urgency"] = URGENCY_LEVELS[stable_index(profile_id, "urgency", len(URGENCY_LEVELS))]
        if stable_index(profile_id, "medium-motivation", 2) == 0:
            values["motivation"] = MOTIVATION_LEVELS[stable_index(profile_id, "motivation", 2)]
    elif completeness == "low":
        text = profile_text(mentee, ("problem_description",))
        if text and stable_index(profile_id, "low-scenario", 3) == 0:
            values["request_scenario"] = infer_request_scenario(mentee)
        if text and stable_index(profile_id, "low-help-format", 5) == 0:
            values["help_format"] = infer_help_format(mentee)

    enriched: dict[str, Any] = {}
    inserted = False
    for key, value in mentee.items():
        if key in MENTEE_FIELDS:
            if not inserted:
                enriched.update(values)
                inserted = True
            continue
        enriched[key] = value
        if key == "profile_completeness" and not inserted:
            enriched.update(values)
            inserted = True
    if not inserted:
        enriched.update(values)
    return enriched


def infer_mentor_scenarios(mentor: dict[str, Any]) -> list[str]:
    text = profile_text(
        mentor,
        ("role", "expertise", "formats", "bio", "test_role", "mentoring_style"),
    )
    scenarios: list[str] = []
    if re.search(r"career|job search|резюме|собесед|interview", text):
        scenarios.extend(["подготовка к переходу роли", "подготовка к собеседованию"])
    if re.search(r"lead|manager|head|onboard|онборд", text):
        scenarios.append("онбординг")
    if re.search(r"проект|roadmap|портфолио|review|ревью", text):
        scenarios.append("проектное сопровождение")
    if re.search(r"задач|debug|architecture|разработ|аналит|design|marketing", text):
        scenarios.append("решение рабочей задачи")
    scenarios.append("развитие компетенций")
    return list(dict.fromkeys(scenarios))


def infer_mentor_help_formats(mentor: dict[str, Any]) -> list[str]:
    text = profile_text(mentor, ("formats", "mentoring_style", "bio"))
    formats: list[str] = []
    if re.search(r"директив|прям|конкрет|консультац", text):
        formats.append(HELP_FORMATS[0])
    if re.search(r"коуч|вопрос|поддерж|сократ", text):
        formats.append(HELP_FORMATS[1])
    if re.search(r"документ|материал|roadmap|план", text):
        formats.append(HELP_FORMATS[2])
    if re.search(r"проект|практи|домаш|ревью|регуляр", text):
        formats.append(HELP_FORMATS[3])
    if not formats:
        start = stable_index(mentor.get("id"), "mentor-help-format", len(HELP_FORMATS))
        formats.append(HELP_FORMATS[start])
    return list(dict.fromkeys(formats))


def enrich_mentor(mentor: dict[str, Any]) -> dict[str, Any]:
    completeness = mentor.get("profile_completeness")
    profile_id = mentor.get("id")
    scenarios = infer_mentor_scenarios(mentor)
    formats = infer_mentor_help_formats(mentor)

    if completeness == "high":
        request_scenarios = scenarios[:3]
        help_formats = formats[:3]
    elif completeness == "medium":
        request_scenarios = scenarios[:2]
        help_formats = formats[:1]
    else:
        request_scenarios = scenarios[:1] if stable_index(profile_id, "low-mentor-scenario", 3) == 0 else []
        help_formats = formats[:1] if stable_index(profile_id, "low-mentor-help", 4) == 0 else []

    values = {
        "request_scenarios": request_scenarios,
        "help_formats": help_formats,
    }
    enriched: dict[str, Any] = {}
    inserted = False
    for key, value in mentor.items():
        if key in MENTOR_FIELDS:
            if not inserted:
                enriched.update(values)
                inserted = True
            continue
        enriched[key] = value
        if key == "test_role" and not inserted:
            enriched.update(values)
            inserted = True
    if not inserted:
        enriched.update(values)
    return enriched


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    mentees = json.loads(MENTEES_PATH.read_text(encoding="utf-8"))
    mentors = json.loads(MENTORS_PATH.read_text(encoding="utf-8"))
    write_json(MENTEES_PATH, [enrich_mentee(profile) for profile in mentees])
    write_json(MENTORS_PATH, [enrich_mentor(profile) for profile in mentors])
    print(f"Updated mentees: {len(mentees)}")
    print(f"Updated mentors: {len(mentors)}")


if __name__ == "__main__":
    main()

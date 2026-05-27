import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException
#

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

MENTEES_PATH = DATA_DIR / "mentees.json"
MENTORS_PATH = DATA_DIR / "mentors.json"


def load_json_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Файл не найден: {path}",
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка чтения JSON-файла {path.name}: {str(error)}",
        ) from error

    if not isinstance(data, list):
        raise HTTPException(
            status_code=500,
            detail=f"Файл {path.name} должен содержать список объектов",
        )

    return data


def load_mentees() -> list[dict[str, Any]]:
    return load_json_file(MENTEES_PATH)


def load_mentors() -> list[dict[str, Any]]:
    return load_json_file(MENTORS_PATH)


def get_mentee_by_id(mentee_id: str) -> dict[str, Any]:
    mentees = load_mentees()

    for mentee in mentees:
        if mentee.get("id") == mentee_id:
            return mentee

    raise HTTPException(
        status_code=404,
        detail=f"Менти с id={mentee_id} не найден",
    )


def get_mentor_by_id(mentor_id: str) -> dict[str, Any] | None:
    mentors = load_mentors()

    for mentor in mentors:
        if mentor.get("id") == mentor_id:
            return mentor

    return None
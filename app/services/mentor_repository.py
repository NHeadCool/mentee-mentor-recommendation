import json
from pathlib import Path
from typing import Dict, List, Any


MENTORS_PATH = Path("app/data/mentors.json")


def load_mentors() -> List[Dict[str, Any]]:
    with open(MENTORS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        return data.get("mentors", [])

    return data
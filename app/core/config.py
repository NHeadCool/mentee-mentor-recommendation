from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    yandex_cloud_folder: str
    yandex_cloud_api_key: str
    yandex_cloud_model: str = "yandexgpt-5-lite/latest"
    yandex_cloud_base_url: str = "https://ai.api.cloud.yandex.net/v1"
    llm_candidate_selector: Literal["filters", "pagerank"] = "filters"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

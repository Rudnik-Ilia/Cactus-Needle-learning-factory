from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    path_to_model: str | Path = ""
    path_to_tools_file: str | Path = ""


SETTINGS = Settings()

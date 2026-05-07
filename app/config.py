from pydantic import PostgresDsn, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    postgres_dsn_async: PostgresDsn = Field(
        default='postgresql+asyncpg://user:pass@localhost:5432/foobar'
    )

    model_config = SettingsConfigDict(env_file="app/.env")


def load_config() -> Config:
    return Config()

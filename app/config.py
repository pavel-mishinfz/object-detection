from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    postgres_dsn_async: PostgresDsn = Field(
        default='postgresql+asyncpg://user:pass@localhost:5432/foobar'
    )

    sentinel_client_id: str = Field(default='client_id')
    sentinel_client_secret: SecretStr = Field(default='client_secret')

    redis_host: str = Field(default='localhost')
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)

    sentinel_temp_dir: str = Field(default='sentinel_temp')
    sentinel_images_dir: str = Field(default='sentinel_images')

    model_config = SettingsConfigDict(env_file="app/.env")


def load_config() -> Config:
    return Config()

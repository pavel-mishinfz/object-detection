from typing import Tuple, Type
from pydantic import Field, PostgresDsn, SecretStr, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource


class Config(BaseSettings):
    postgres_dsn_async: PostgresDsn = Field(
        default='postgresql+asyncpg://user:pass@localhost:5432/foobar',
        env='POSTGRES_DSN_ASYNC',
        alias='POSTGRES_DSN_ASYNC'
    )

    client_id: str = Field(
        default='client_id',
        env='CLIENT_ID',
        alias='CLIENT_ID'
    )

    client_secret: SecretStr = Field(
        default='client_secret',
        env='CLIENT_SECRET',
        alias='CLIENT_SECRET'
    )

    redis_host: str = Field(
        default='localhost',
        env='REDIS_HOST',
        alias='REDIS_HOST'
    )

    redis_port: int = Field(
        default=6379,
        env='REDIS_PORT',
        alias='REDIS_PORT'
    )

    redis_db: int = Field(
        default=0,
        env='REDIS_DB',
        alias='REDIS_DB'
    )

    model_config = SettingsConfigDict(env_file=".env", extra='allow')

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return dotenv_settings, env_settings, init_settings


def load_config() -> Config:
    return Config()

from typing import Tuple, Type
from pydantic import Field, PostgresDsn, SecretStr, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource


class Config(BaseSettings):
    postgres_dsn_async: PostgresDsn = Field(
        default='postgresql+asyncpg://user:pass@localhost:5432/foobar',
        env='POSTGRES_DSN_ASYNC',
        alias='POSTGRES_DSN_ASYNC'
    )

    default_objects_config_path: FilePath = Field(
        default='default-objects.json',
        env='DEFAULT_OBJECTS_CONFIG_PATH',
        alias='DEFAULT_OBJECTS_CONFIG_PATH'
    )

    width_image: int = Field(
        default=800,
        env='WIDTH_IMAGE',
        alias='WIDTH_IMAGE'
    )

    height_image: int = Field(
        default=800,
        env='HEIGHT_IMAGE',
        alias='HEIGHT_IMAGE'
    )

    model_path: str = Field(
        default='path/to/model',
        env='MODEL_PATH',
        alias='MODEL_PATH'
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

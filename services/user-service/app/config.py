from typing import Tuple, Type
from pydantic import Field, PostgresDsn, SecretStr, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource


class Config(BaseSettings):
    postgres_dsn_async: PostgresDsn = Field(
        default='postgresql+asyncpg://user:pass@localhost:5432/foobar',
        env='POSTGRES_DSN_ASYNC',
        alias='POSTGRES_DSN_ASYNC'
    )

    jwt_secret: SecretStr = Field(
        default='jwt_secret',
        env='JWT_SECRET',
        alias='JWT_SECRET'
    )

    reset_password_token_secret: SecretStr = Field(
        default='reset_password_token_secret',
        env='RESET_PASSWORD_TOKEN_SECRET',
        alias='RESET_PASSWORD_TOKEN_SECRET'
    )

    verification_token_secret: SecretStr = Field(
        default='verification_token_secret',
        env='VERIFICATION_TOKEN_SECRET',
        alias='VERIFICATION_TOKEN_SECRET'
    )

    default_groups_config_path: FilePath = Field(
        default='default-groups.json',
        env='DEFAULT_GROUPS_CONFIG_PATH',
        alias='DEFAULT_GROUPS_CONFIG_PATH'
    )

    own_email: str = Field(
        default='user@example.com',
        env='OWN_EMAIL',
        alias='OWN_EMAIL'
    )

    own_email_password: str = Field(
        default='password',
        env='OWN_EMAIL_PASSWORD',
        alias='OWN_EMAIL_PASSWORD'
    )

    smtp_server: str = Field(
        default='smtp.mail.ru',
        env='SMTP_SERVER',
        alias='SMTP_SERVER'
    )

    smtp_port: int = Field(
        default='465',
        env='smtp_port',
        alias='smtp_port'
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

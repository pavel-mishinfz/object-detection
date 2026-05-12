from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    postgres_dsn_async: PostgresDsn = Field(
        default='postgresql+asyncpg://user:pass@localhost:5432/foobar'
    )

    sentinel_client_id: str = Field(default='client_id')
    sentinel_client_secret: SecretStr = Field(default='client_secret')

    sentinel_temp_dir: str = Field(default='sentinel_temp')
    sentinel_images_dir: str = Field(default='sentinel_images')

    jwt_secret: SecretStr = Field(default='jwt_secret')
    reset_password_token_secret: SecretStr = Field(default='reset_password_token_secret')
    verification_token_secret: SecretStr = Field(default='verification_token_secret')

    sender_email: str = Field(default='noreply@example.com')
    sender_password: SecretStr = Field(default='password')
    smtp_server: str = Field(default='smtp.example.com')
    smtp_port: int = Field(default=465)

    default_groups_config_path: str = Field(default='default-groups.json')
    default_objects_config_path: str = Field(default='default-objects.json')
    model_path: str = Field(default='best.pt')

    model_config = SettingsConfigDict(env_file="app/.env")


def load_config() -> Config:
    return Config()

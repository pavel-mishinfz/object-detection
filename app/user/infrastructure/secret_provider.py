from fastapi import Depends

from app.config import Config, load_config


class SecretProvider:
    def __init__(self, cfg: Config) -> None:
        self.jwt_secret = cfg.jwt_secret.get_secret_value()
        self.reset_password_token_secret = cfg.reset_password_token_secret.get_secret_value()
        self.verification_token_secret = cfg.verification_token_secret.get_secret_value()


def get_secret_provider(cfg: Config = Depends(load_config)) -> SecretProvider:
    return SecretProvider(cfg)

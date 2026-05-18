import uuid
from typing import Optional

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, InvalidPasswordException, UUIDIDMixin

from app.user.exceptions import UserValidationError
from app.user.infrastructure.email_gateway import SmtpEmailSender, get_email_sender
from app.user.infrastructure.secret_provider import SecretProvider, get_secret_provider
from app.user.infrastructure.user_db import get_user_db
from app.user.models.user import User
from app.user.services import validators


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    def __init__(self, user_db, email_sender: SmtpEmailSender) -> None:
        super().__init__(user_db)
        self._email_sender = email_sender

    async def validate_password(self, password: str, user) -> None:
        try:
            validators.validate_password(password, user.email)
        except UserValidationError as e:
            raise InvalidPasswordException(reason=str(e))

    async def on_after_register(
        self, user: User, request: Optional[Request] = None
    ) -> None:
        print(f"User {user.id} has registered.")

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ) -> None:
        await self._email_sender.send_reset_password(user.email, token)

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ) -> None:
        await self._email_sender.send_verification(user.email, token)


async def get_user_manager(
    user_db=Depends(get_user_db),
    secret_provider: SecretProvider = Depends(get_secret_provider),
    email_sender: SmtpEmailSender = Depends(get_email_sender),
):
    manager = UserManager(user_db, email_sender)
    manager.reset_password_token_secret = secret_provider.reset_password_token_secret
    manager.verification_token_secret = secret_provider.verification_token_secret
    yield manager

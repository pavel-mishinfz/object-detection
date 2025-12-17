import uuid
from typing import Optional, Union

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, UUIDIDMixin, InvalidPasswordException

from .database import database, models
from . import secretprovider, schemas
from app import config

from email.mime.text import MIMEText
from smtplib import SMTP_SSL


app_config: config.Config = config.load_config()


class UserManager(UUIDIDMixin, BaseUserManager[models.User, uuid.UUID]):

    async def validate_password(
        self,
        password: str,
        user: Union[schemas.user.UserCreate, models.User],
    ) -> None:
        if len(password) < 8:
            raise InvalidPasswordException(
                reason="Пароль должен содержать не менее 8 символов"
            )
        if user.email in password:
            raise InvalidPasswordException(
                reason="Пароль не должен содержать e-mail"
            )

    async def on_after_register(
            self, user: models.User, request: Optional[Request] = None
    ):
        print(f"User {user.id} has registered.")

    async def on_after_forgot_password(
            self, user: models.User, token: str, request: Optional[Request] = None
    ):
        message = make_reset_password_template(token)
        await send_email(message, "Сброс пароля", user.email)
        print(f"User {user.id} has forgot their password. Reset token: {token}")

    async def on_after_request_verify(
            self, user: models.User, token: str, request: Optional[Request] = None
    ):
        print(f"Verification requested for user {user.id}. Verification token: {token}")


async def get_user_manager(
        user_db=Depends(database.get_user_db),
        secret_provider: secretprovider.SecretProvider = Depends(secretprovider.get_secret_provider)
):
    user_manager = UserManager(user_db)
    user_manager.reset_password_token_secret = secret_provider.reset_password_token_secret
    user_manager.verification_token_secret = secret_provider.verification_token_secret
    yield user_manager

def make_reset_password_template(token: str):
    return f"""
    <html>
        <body>
            <div style="background-color:#fff;padding:20px">
            <h1>Забыли пароль?</h1>
            <p style="display:block;font-size:18px">
                Для сброса пароля нажмите:
            </p>
            <button type="button" style="margin-top:10px;padding:10px 18px;background-color:blue;border:none;border-radius:15px">
                <a href="http://127.0.0.1:3000/reset-password?token={token}" style="text-decoration:none;color:#fff;font-weight:700">
                    Сбросить пароль 
                </a>
            </button>
            </div>
        </body>
    </html>
    """

async def send_email(message: str, subject: str, to: str):
    msg = MIMEText(message, "html")
    msg['Subject'] = subject
    msg['From'] = f'<{app_config.own_email}>'
    msg['To'] = to
    
    with SMTP_SSL(app_config.smtp_server, port=app_config.smtp_port) as server:
        server.login(app_config.own_email, app_config.own_email_password)
        server.send_message(msg)
        server.quit()
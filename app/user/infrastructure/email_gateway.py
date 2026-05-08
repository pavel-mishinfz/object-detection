import asyncio
from email.mime.text import MIMEText
from smtplib import SMTP_SSL

from fastapi import Depends

from app.config import Config, load_config
from app.user.application.interfaces import IEmailSender


class SmtpEmailSender(IEmailSender):
    def __init__(
        self,
        sender_email: str,
        sender_password: str,
        smtp_server: str,
        smtp_port: int,
    ) -> None:
        self._sender_email = sender_email
        self._sender_password = sender_password
        self._smtp_server = smtp_server
        self._smtp_port = smtp_port

    async def send_reset_password(self, to_email: str, token: str) -> None:
        html = _render_reset_password(token)
        await asyncio.to_thread(
            self._send_smtp, html, "Сброс пароля", to_email
        )

    async def send_verification(self, to_email: str, token: str) -> None:
        html = _render_verification(token)
        await asyncio.to_thread(
            self._send_smtp, html, "Подтверждение аккаунта", to_email
        )

    def _send_smtp(self, html: str, subject: str, to: str) -> None:
        msg = MIMEText(html, "html")
        msg["Subject"] = subject
        msg["From"] = f"<{self._sender_email}>"
        msg["To"] = to
        with SMTP_SSL(self._smtp_server, port=self._smtp_port) as server:
            server.login(self._sender_email, self._sender_password)
            server.send_message(msg)


def _render_reset_password(token: str) -> str:
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


def _render_verification(token: str) -> str:
    return f"""
    <html>
        <body>
            <div style="background-color:#fff;padding:20px">
            <h1>Подтверждение аккаунта</h1>
            <p style="display:block;font-size:18px">
                Для подтверждения аккаунта нажмите:
            </p>
            <button type="button" style="margin-top:10px;padding:10px 18px;background-color:blue;border:none;border-radius:15px">
                <a href="http://127.0.0.1:3000/verify?token={token}" style="text-decoration:none;color:#fff;font-weight:700">
                    Подтвердить аккаунт
                </a>
            </button>
            </div>
        </body>
    </html>
    """


def get_email_sender(cfg: Config = Depends(load_config)) -> SmtpEmailSender:
    return SmtpEmailSender(
        sender_email=cfg.sender_email,
        sender_password=cfg.sender_password.get_secret_value(),
        smtp_server=cfg.smtp_server,
        smtp_port=cfg.smtp_port,
    )

from app.user.domain.errors import GroupValidationError, UserValidationError


def validate_password(password: str, email: str) -> None:
    if len(password) < 8:
        raise UserValidationError("Пароль должен содержать не менее 8 символов")
    if email in password:
        raise UserValidationError("Пароль не должен содержать e-mail")


def validate_group_name(name: str) -> None:
    if not name or not name.strip():
        raise GroupValidationError("Название группы не может быть пустым")

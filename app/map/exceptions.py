from app.shared.exceptions import AccessDeniedError, NotFoundError


class DomainError(Exception):
    pass


class PolygonValidationError(DomainError):
    pass


class PolygonNotFoundError(DomainError, NotFoundError):
    pass


class PolygonAccessDeniedError(DomainError, AccessDeniedError):
    pass


class PolygonNameConflictError(DomainError):
    pass


class PolygonLimitExceededError(DomainError):
    pass

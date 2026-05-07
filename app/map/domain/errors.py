class DomainError(Exception):
    pass


class PolygonValidationError(DomainError):
    pass


class PolygonNotFoundError(DomainError):
    pass


class PolygonAccessDeniedError(DomainError):
    pass


class PolygonNameConflictError(DomainError):
    pass


class PolygonLimitExceededError(DomainError):
    pass

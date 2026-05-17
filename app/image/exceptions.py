from app.shared.exceptions import NotFoundError


class DomainError(Exception):
    pass


class InvalidDateRangeError(DomainError):
    pass


class NoPreviewAvailableError(DomainError):
    pass


class ImageNotFoundError(DomainError, NotFoundError):
    pass

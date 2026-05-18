class DomainError(Exception):
    pass


class UserValidationError(DomainError):
    pass


class GroupNotFoundError(DomainError):
    pass


class GroupValidationError(DomainError):
    pass

class ImageDomainError(Exception):
    pass


class InvalidDateRangeError(ImageDomainError):
    pass


class NoPreviewAvailableError(ImageDomainError):
    pass


class ImageNotFoundError(ImageDomainError):
    pass

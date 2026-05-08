class AnalysisDomainError(Exception):
    pass


class AreaNotFoundError(AnalysisDomainError):
    pass


class AreaAccessDeniedError(AnalysisDomainError):
    pass


class NoImagesError(AnalysisDomainError):
    pass

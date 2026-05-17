from datetime import date

from app.image.exceptions import InvalidDateRangeError


def validate_date_range(date_start: date, date_end: date, today: date) -> None:
    if date_start > today or date_end > today:
        raise InvalidDateRangeError("Дата не может быть в будущем")
    if date_start > date_end:
        raise InvalidDateRangeError(
            "Дата начала должна быть раньше или равна дате окончания"
        )

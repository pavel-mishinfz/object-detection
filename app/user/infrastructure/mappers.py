from app.user.domain.user import Group, User
from app.user.infrastructure.models import Group as GroupRecord, User as UserRecord


def to_domain_group(record: GroupRecord) -> Group:
    return Group(id=record.id, name=record.name)


def to_domain_user(record: UserRecord) -> User:
    return User(
        id=record.id,
        email=record.email,
        username=record.username,
        group_id=record.group_id,
        is_active=record.is_active,
        is_verified=record.is_verified,
        is_superuser=record.is_superuser,
    )

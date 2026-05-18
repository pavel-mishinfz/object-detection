from app.user.entity.group import Group
from app.user.entity.user import User
from app.user.models.group import Group as GroupRecord
from app.user.models.user import User as UserRecord


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

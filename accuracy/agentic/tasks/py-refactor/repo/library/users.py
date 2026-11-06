from datetime import date

from . import db


class UserNotFound(LookupError):
    pass


def get_user(id: int) -> db.User | None:
    return db.USERS.get(id)


def require_user(id: int) -> db.User:
    user = get_user(id)
    if user is None:
        raise UserNotFound(id)
    return user


def create_user(name: str, email: str, joined: date | None = None, roles: set[str] | None = None) -> db.User:
    user = db.User(db.next_id(db.USERS), name, email, joined or date.today(), roles=set(roles or ()))
    db.USERS[user.id] = user
    return user


def delete_user(id: int, when: date | None = None) -> None:
    require_user(id).deleted_at = when or date.today()


def is_staff(id: int) -> bool:
    user = get_user(id)
    return user is not None and "staff" in user.roles

"""Reports for staff. These must cover every account, including deleted ones."""
from datetime import date

from . import db
from .fines import outstanding
from .users import fetch_user


def user_summary(user_id: int, today: date) -> str:
    user = fetch_user(user_id, include_deleted=True)
    if user is None:
        return f"#{user_id}: unknown"
    state = f"deleted {user.deleted_at:%Y-%m-%d}" if user.deleted_at else "active"
    n_loans = sum(1 for l in db.LOANS.values() if l.user_id == user_id)
    return f"#{user.id} {user.name} <{user.email}> {state}, {n_loans} loans, owes {outstanding(user_id, today)} cents"


def all_summaries(today: date) -> list[str]:
    return [user_summary(uid, today) for uid in sorted(db.USERS)]

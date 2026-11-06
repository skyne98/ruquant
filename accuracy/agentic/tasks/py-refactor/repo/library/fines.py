from datetime import date

from . import db
from .users import get_user

CENTS_PER_DAY = 25
CAP_CENTS = 1000


def fine_for(loan: db.Loan, today: date) -> int:
    end = loan.returned or today
    late = (end - loan.due).days
    return min(CAP_CENTS, max(0, late) * CENTS_PER_DAY)


def outstanding(user_id: int, today: date) -> int:
    user = get_user(user_id)
    if user is None:
        return 0
    return sum(fine_for(l, today) for l in db.LOANS.values() if l.user_id == user.id)

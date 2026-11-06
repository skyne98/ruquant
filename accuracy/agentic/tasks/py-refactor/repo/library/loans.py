from datetime import date, timedelta

from . import db
from .books import available
from .users import get_user, is_staff

LOAN_DAYS = 21
STAFF_LOAN_DAYS = 60
MAX_LOANS = 5


class LoanError(Exception):
    pass


def open_loans(user_id: int) -> list[db.Loan]:
    return [l for l in db.LOANS.values() if l.user_id == user_id and l.returned is None]


def lend(user_id: int, book_id: int, today: date | None = None) -> db.Loan:
    today = today or date.today()
    user = get_user(user_id)
    if user is None:
        raise LoanError(f"no such user {user_id}")
    if len(open_loans(user.id)) >= MAX_LOANS:
        raise LoanError(f"{user.name} already has {MAX_LOANS} books")
    if available(book_id) <= 0:
        raise LoanError("no copy available")
    days = STAFF_LOAN_DAYS if is_staff(user.id) else LOAN_DAYS
    loan = db.Loan(db.next_id(db.LOANS), user.id, book_id, today, today + timedelta(days=days))
    db.LOANS[loan.id] = loan
    return loan


def give_back(loan_id: int, today: date | None = None) -> db.Loan:
    loan = db.LOANS[loan_id]
    loan.returned = today or date.today()
    return loan

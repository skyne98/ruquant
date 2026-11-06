from datetime import date

from . import db
from .users import fetch_user


def overdue_emails(today: date) -> list[tuple[str, str]]:
    """(email address, message) for every overdue loan."""
    out = []
    for loan in sorted(db.LOANS.values(), key=lambda l: l.id):
        if loan.returned is None and loan.due < today:
            user = fetch_user(loan.user_id)
            if user is None:
                continue
            book = db.BOOKS[loan.book_id]
            out.append((user.email, f"Dear {user.name}, '{book.title}' was due on {loan.due:%Y-%m-%d}."))
    return out

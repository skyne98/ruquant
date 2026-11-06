from datetime import date

import pytest

from library.admin import user_summary
from library.books import add_book
from library.fines import outstanding
from library.loans import LoanError, give_back, lend
from library.notify import overdue_emails
from library.users import create_user, fetch_user

D = date(2024, 3, 1)


def test_get_user():
    u = create_user("Ann", "ann@x.org", D)
    assert fetch_user(u.id) is u
    assert fetch_user(999) is None


def test_lend_and_fine():
    u = create_user("Bob", "bob@x.org", D)
    b = add_book("Dune", "Herbert")
    loan = lend(u.id, b.id, D)
    assert loan.due == date(2024, 3, 22)
    give_back(loan.id, date(2024, 3, 25))
    assert outstanding(u.id, date(2024, 4, 1)) == 75


def test_staff_loan_longer():
    u = create_user("Cy", "cy@x.org", D, {"staff"})
    loan = lend(u.id, add_book("Emma", "Austen").id, D)
    assert (loan.due - D).days == 60


def test_no_copy():
    u = create_user("Dee", "dee@x.org", D)
    b = add_book("Ulysses", "Joyce")
    lend(u.id, b.id, D)
    with pytest.raises(LoanError):
        lend(u.id, b.id, D)


def test_overdue_emails():
    u = create_user("Eve", "eve@x.org", D)
    lend(u.id, add_book("Ubik", "Dick").id, D)
    assert overdue_emails(date(2024, 4, 1)) == [("eve@x.org", "Dear Eve, 'Ubik' was due on 2024-03-22.")]


def test_summary():
    u = create_user("Fay", "fay@x.org", D)
    assert user_summary(u.id, D) == f"#{u.id} Fay <fay@x.org> active, 0 loans, owes 0 cents"

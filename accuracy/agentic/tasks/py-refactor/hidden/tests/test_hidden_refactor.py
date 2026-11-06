import pathlib
import re
from datetime import date

import pytest

from library import cli, users
from library.admin import all_summaries, user_summary
from library.books import add_book
from library.fines import outstanding
from library.loans import LoanError, give_back, lend
from library.notify import overdue_emails
from library.users import UserNotFound, create_user, delete_user, fetch_user, is_staff, require_user

D = date(2024, 3, 1)


def test_get_user_is_gone():
    assert not hasattr(users, "get_user")
    for path in pathlib.Path("library").rglob("*.py"):
        assert not re.search(r"\bget_user\b", path.read_text()), path


def test_fetch_user_signature():
    u = create_user("Ann", "ann@x.org", D)
    assert fetch_user(u.id) is u
    assert fetch_user(user_id=u.id) is u
    assert fetch_user(12345) is None
    with pytest.raises(TypeError):
        fetch_user(u.id, True)  # include_deleted is keyword-only


def test_fetch_deleted():
    u = create_user("Bob", "bob@x.org", D)
    delete_user(u.id, D)
    assert fetch_user(u.id) is None
    assert fetch_user(u.id, include_deleted=True) is u


def test_require_and_staff_skip_deleted():
    u = create_user("Cy", "cy@x.org", D, {"staff"})
    assert is_staff(u.id)
    delete_user(u.id, D)
    assert not is_staff(u.id)
    with pytest.raises(UserNotFound):
        require_user(u.id)


def test_deleted_cannot_borrow():
    u = create_user("Dee", "dee@x.org", D)
    b = add_book("Dune", "Herbert")
    delete_user(u.id, D)
    with pytest.raises(LoanError):
        lend(u.id, b.id, D)


def test_no_email_for_deleted_but_fines_count():
    a = create_user("Eve", "eve@x.org", D)
    z = create_user("Zed", "zed@x.org", D)
    la = lend(a.id, add_book("Ubik", "Dick").id, D)
    lend(z.id, add_book("Emma", "Austen").id, D)
    delete_user(a.id, date(2024, 3, 30))
    assert overdue_emails(date(2024, 4, 1)) == [("zed@x.org", "Dear Zed, 'Emma' was due on 2024-03-22.")]
    assert outstanding(a.id, date(2024, 4, 1)) == 250
    give_back(la.id, date(2024, 4, 2))
    assert outstanding(a.id, date(2024, 5, 1)) == 275


def test_admin_lists_deleted():
    a = create_user("Fay", "fay@x.org", D)
    create_user("Gus", "gus@x.org", D)
    delete_user(a.id, date(2024, 3, 5))
    assert user_summary(a.id, D) == f"#{a.id} Fay <fay@x.org> deleted 2024-03-05, 0 loans, owes 0 cents"
    assert len(all_summaries(D)) == 2


def test_whois(capsys):
    a = create_user("Hal", "hal@x.org", D)
    assert cli.main(["whois", str(a.id)]) == 0
    delete_user(a.id, D)
    assert cli.main(["whois", str(a.id)]) == 1
    assert capsys.readouterr().out.splitlines() == ["Hal", "unknown"]

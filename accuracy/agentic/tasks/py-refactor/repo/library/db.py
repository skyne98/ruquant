"""In-memory tables. Tests and the CLI seed them through the functions below."""
from dataclasses import dataclass, field
from datetime import date


@dataclass
class User:
    id: int
    name: str
    email: str
    joined: date
    deleted_at: date | None = None
    roles: set[str] = field(default_factory=set)


@dataclass
class Book:
    id: int
    title: str
    author: str
    copies: int = 1


@dataclass
class Loan:
    id: int
    user_id: int
    book_id: int
    start: date
    due: date
    returned: date | None = None


USERS: dict[int, User] = {}
BOOKS: dict[int, Book] = {}
LOANS: dict[int, Loan] = {}


def reset() -> None:
    USERS.clear()
    BOOKS.clear()
    LOANS.clear()


def next_id(table: dict) -> int:
    return max(table, default=0) + 1

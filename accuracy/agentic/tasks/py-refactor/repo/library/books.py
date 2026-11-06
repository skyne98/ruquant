from . import db


def add_book(title: str, author: str, copies: int = 1) -> db.Book:
    book = db.Book(db.next_id(db.BOOKS), title, author, copies)
    db.BOOKS[book.id] = book
    return book


def available(book_id: int) -> int:
    book = db.BOOKS[book_id]
    out = sum(1 for l in db.LOANS.values() if l.book_id == book_id and l.returned is None)
    return book.copies - out

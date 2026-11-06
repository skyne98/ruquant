import pytest

from shop.checkout import receipt, total
from shop.discounts import InvalidCoupon
from shop.models import Order, Product

PEN = Product("PEN-1", "Pen", 150)
NOVEL = Product("BK-7", "Novel", 1299, "books")


def test_simple_total():
    o = Order("ann")
    o.add(PEN, 3)
    assert total(o) == 450


def test_add_merges_lines():
    o = Order("ann")
    o.add(PEN, 2)
    o.add(PEN, 1)
    assert len(o.items) == 1 and o.items[0].qty == 3


def test_books_promo():
    o = Order("bob")
    o.add(NOVEL, 2)
    assert total(o) == 2598 - 130


def test_fixed_coupon():
    o = Order("cy")
    o.add(PEN, 2)
    assert total(o, "welcome5") == 0


def test_bad_coupon():
    o = Order("cy")
    o.add(PEN)
    with pytest.raises(InvalidCoupon):
        total(o, "NOPE")


def test_receipt_lists_items():
    o = Order("dee")
    o.add(PEN, 2)
    text = receipt(o)
    assert "2 x Pen" in text and "$3.00" in text

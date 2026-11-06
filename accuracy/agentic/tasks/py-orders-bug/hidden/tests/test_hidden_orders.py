from shop.checkout import receipt, total
from shop.models import Order, Product

NB = Product("NB-A5", "Notebook", 250)
NOVEL = Product("BK-7", "Novel", 1299, "books")
PEN = Product("PEN-1", "Pen", 150)


def test_bulk_charged_once():
    o = Order("eve")
    o.add(NB, 12)
    assert total(o) == 2700


def test_bulk_receipt_adds_up():
    o = Order("eve")
    o.add(NB, 12)
    text = receipt(o)
    assert "12 x Notebook" in text
    assert "$30.00" in text  # line and subtotal at full price
    assert "Bulk discount (Notebook)" in text and "-$3.00" in text
    assert text.rstrip().endswith("$27.00")


def test_bulk_book_with_promo_and_coupon():
    o = Order("fay")
    o.add(NOVEL, 10)
    o.add(PEN, 1)
    # 12990 + 150; bulk 1299, books 5% 650 -> 11191; SAVE15 1679 -> 9512
    assert total(o, "SAVE15") == 9512


def test_under_bulk_threshold():
    o = Order("gus")
    o.add(NB, 9)
    assert total(o) == 2250
    assert "Bulk" not in receipt(o)


def test_merged_lines_reach_bulk():
    o = Order("hal")
    o.add(NB, 6)
    o.add(NB, 4)
    assert total(o) == 2250

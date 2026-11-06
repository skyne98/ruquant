from .models import LineItem, Order
from .money import percent_of

BULK_MIN_QTY = 10
BULK_PERCENT = 10


def line_total(item: LineItem) -> int:
    """Price of one line before order-level discounts."""
    total = item.product.price_cents * item.qty
    if item.qty >= BULK_MIN_QTY:
        total -= percent_of(total, BULK_PERCENT)
    return total


def subtotal(order: Order) -> int:
    return sum(line_total(item) for item in order.items)

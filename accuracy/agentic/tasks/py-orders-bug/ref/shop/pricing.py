from .models import LineItem, Order

BULK_MIN_QTY = 10
BULK_PERCENT = 10


def line_total(item: LineItem) -> int:
    """Price of one line before order-level discounts."""
    return item.product.price_cents * item.qty


def subtotal(order: Order) -> int:
    return sum(line_total(item) for item in order.items)

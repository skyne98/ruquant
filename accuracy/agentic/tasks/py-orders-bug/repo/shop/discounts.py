"""Order-level discounts. Each rule returns (label, amount in cents) pairs; amounts are positive."""
from .models import Order
from .money import percent_of
from .pricing import BULK_MIN_QTY, BULK_PERCENT

CATEGORY_PROMOS = {"books": 5}

COUPONS = {
    "WELCOME5": ("fixed", 500),
    "SAVE15": ("percent", 15),
}


class InvalidCoupon(ValueError):
    pass


def bulk(order: Order) -> list[tuple[str, int]]:
    out = []
    for item in order.items:
        if item.qty >= BULK_MIN_QTY:
            gross = item.product.price_cents * item.qty
            out.append((f"Bulk discount ({item.product.name})", percent_of(gross, BULK_PERCENT)))
    return out


def category(order: Order) -> list[tuple[str, int]]:
    out = []
    for cat, percent in CATEGORY_PROMOS.items():
        gross = sum(i.product.price_cents * i.qty for i in order.items if i.product.category == cat)
        if gross:
            out.append((f"{cat.title()} promo {percent}%", percent_of(gross, percent)))
    return out


def coupon(code: str | None, amount_after_other_discounts: int) -> list[tuple[str, int]]:
    if code is None:
        return []
    try:
        kind, value = COUPONS[code.upper()]
    except KeyError:
        raise InvalidCoupon(code) from None
    if kind == "fixed":
        off = min(value, amount_after_other_discounts)
    else:
        off = percent_of(amount_after_other_discounts, value)
    return [(f"Coupon {code.upper()}", off)]

from .discounts import bulk, category, coupon
from .models import Order
from .money import fmt
from .pricing import line_total, subtotal

TAX_PERCENT = 0  # prices include tax


def discounts(order: Order, coupon_code: str | None = None) -> list[tuple[str, int]]:
    found = bulk(order) + category(order)
    after = subtotal(order) - sum(amount for _, amount in found)
    return found + coupon(coupon_code, after)


def total(order: Order, coupon_code: str | None = None) -> int:
    return max(0, subtotal(order) - sum(amount for _, amount in discounts(order, coupon_code)))


def receipt(order: Order, coupon_code: str | None = None) -> str:
    lines = [f"Receipt for {order.customer}"]
    for item in order.items:
        lines.append(f"  {item.qty} x {item.product.name:<20} {fmt(line_total(item)):>10}")
    lines.append(f"  {'Subtotal':<24} {fmt(subtotal(order)):>10}")
    for label, amount in discounts(order, coupon_code):
        lines.append(f"  {label:<24} {fmt(-amount):>10}")
    lines.append(f"  {'Total':<24} {fmt(total(order, coupon_code)):>10}")
    return "\n".join(lines)

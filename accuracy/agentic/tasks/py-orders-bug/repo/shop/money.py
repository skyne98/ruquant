def percent_of(cents: int, percent: int) -> int:
    """percent% of an amount in cents, rounded half up."""
    return (cents * percent + 50) // 100


def fmt(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100}.{cents % 100:02d}"

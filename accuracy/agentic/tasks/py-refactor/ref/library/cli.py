import sys
from datetime import date

from .admin import all_summaries
from .users import fetch_user


def main(argv: list[str]) -> int:
    if argv[:1] == ["whois"]:
        user = fetch_user(int(argv[1]))
        print(user.name if user else "unknown")
        return 0 if user else 1
    if argv[:1] == ["report"]:
        print("\n".join(all_summaries(date.today())))
        return 0
    print("usage: library whois ID | report", file=sys.stderr)
    return 2

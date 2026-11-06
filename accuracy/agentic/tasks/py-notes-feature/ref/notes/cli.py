import argparse
import sys
from datetime import datetime

from .model import Note
from .storage import Store


def cmd_add(store: Store, args) -> int:
    notes = store.load()
    note = Note(store.next_id(notes), args.title, args.body or "", sorted(set(args.tag or [])))
    if args.date:
        note.created = datetime.fromisoformat(args.date)
    notes.append(note)
    store.save(notes)
    print(note.id)
    return 0


def cmd_list(store: Store, args) -> int:
    notes = sorted(store.load(), key=lambda n: n.created, reverse=True)
    for note in notes:
        print(note.line())
    return 0


def cmd_delete(store: Store, args) -> int:
    notes = store.load()
    kept = [n for n in notes if n.id != args.id]
    if len(kept) == len(notes):
        print(f"no note {args.id}", file=sys.stderr)
        return 1
    store.save(kept)
    return 0


def cmd_tag(store: Store, args) -> int:
    notes = store.load()
    for note in notes:
        if note.id == args.id:
            note.tags = sorted(set(note.tags) | set(args.tags))
            store.save(notes)
            return 0
    print(f"no note {args.id}", file=sys.stderr)
    return 1


def cmd_search(store: Store, args) -> int:
    q = args.query.lower()
    since = datetime.fromisoformat(args.since).date() if args.since else None
    found = [n for n in store.load()
             if (q in n.title.lower() or q in n.body.lower())
             and set(args.tag or []) <= set(n.tags)
             and (since is None or n.created.date() >= since)]
    if not found:
        print("no matches", file=sys.stderr)
        return 1
    for note in sorted(found, key=lambda n: n.created, reverse=True):
        print(note.line())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="notes")
    p.add_argument("--file", help="notes file (default $NOTES_FILE or ~/.notes.json)")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="add a note")
    a.add_argument("title")
    a.add_argument("--body")
    a.add_argument("--tag", action="append")
    a.add_argument("--date", help="creation time, ISO format (default now)")
    a.set_defaults(func=cmd_add)
    sub.add_parser("list", help="list notes, newest first").set_defaults(func=cmd_list)
    d = sub.add_parser("delete", help="delete a note")
    d.add_argument("id", type=int)
    d.set_defaults(func=cmd_delete)
    t = sub.add_parser("tag", help="add tags to a note")
    t.add_argument("id", type=int)
    t.add_argument("tags", nargs="+")
    t.set_defaults(func=cmd_tag)
    s = sub.add_parser("search", help="search notes")
    s.add_argument("query")
    s.add_argument("--tag", action="append")
    s.add_argument("--since")
    s.set_defaults(func=cmd_search)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(Store(args.file), args)


if __name__ == "__main__":
    sys.exit(main())

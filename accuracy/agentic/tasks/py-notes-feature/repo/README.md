# notes

A small command-line notebook. Notes live in a JSON file (`$NOTES_FILE`, default `~/.notes.json`).

    python -m notes add "Buy milk" --tag errands
    python -m notes list
    python -m notes tag 1 urgent
    python -m notes delete 1

Tests: `python -m pytest`.

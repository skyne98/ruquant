import subprocess
import sys

import pytest

from notes.cli import main


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_FILE", str(tmp_path / "n.json"))
    main(["add", "Groceries", "--body", "milk, EGGS", "--tag", "home", "--tag", "errands", "--date", "2024-05-01T09:00:00"])
    main(["add", "Fix bike", "--body", "chain", "--tag", "home", "--date", "2024-05-03T18:30:00"])
    main(["add", "Quarterly report", "--body", "Numbers for Q2", "--tag", "work", "--date", "2024-06-10T08:00:00"])
    main(["add", "egg recipe", "--date", "2024-06-11T07:00:00"])


def run(capsys, *argv):
    capsys.readouterr()
    rc = main(["search", *argv])
    c = capsys.readouterr()
    return rc, c.out.splitlines(), c.err


def test_title_or_body_case_insensitive(db, capsys):
    rc, out, _ = run(capsys, "egg")
    assert rc == 0
    assert out == ["  4  2024-06-11  egg recipe", "  1  2024-05-01  Groceries  [errands, home]"]


def test_tags_all_required(db, capsys):
    rc, out, _ = run(capsys, "", "--tag", "home", "--tag", "errands")
    assert rc == 0 and out == ["  1  2024-05-01  Groceries  [errands, home]"]
    rc, out, _ = run(capsys, "", "--tag", "home")
    assert [l.split()[0] for l in out] == ["2", "1"]


def test_since_inclusive(db, capsys):
    rc, out, _ = run(capsys, "", "--since", "2024-06-10")
    assert rc == 0 and [l.split()[0] for l in out] == ["4", "3"]


def test_combined(db, capsys):
    rc, out, _ = run(capsys, "BIKE", "--tag", "home", "--since", "2024-05-02")
    assert rc == 0 and out == ["  2  2024-05-03  Fix bike  [home]"]


def test_no_matches(db, capsys):
    rc, out, err = run(capsys, "zebra")
    assert rc == 1 and out == [] and "no matches" in err


def test_module_entry_point(db, tmp_path):
    r = subprocess.run([sys.executable, "-m", "notes", "search", "report"], capture_output=True, text=True)
    assert r.returncode == 0 and "Quarterly report" in r.stdout


def test_readme_mentions_search():
    assert "search" in open("README.md").read()

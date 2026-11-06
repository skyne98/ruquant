import pytest

from notes.cli import main


@pytest.fixture
def notes_file(tmp_path, monkeypatch):
    path = tmp_path / "notes.json"
    monkeypatch.setenv("NOTES_FILE", str(path))
    return path


def test_add_and_list(notes_file, capsys):
    assert main(["add", "first", "--tag", "x", "--date", "2024-01-02T10:00:00"]) == 0
    assert main(["add", "second", "--date", "2024-03-04T10:00:00"]) == 0
    capsys.readouterr()
    assert main(["list"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["  2  2024-03-04  second", "  1  2024-01-02  first  [x]"]


def test_delete(notes_file, capsys):
    main(["add", "a"])
    assert main(["delete", "1"]) == 0
    assert main(["delete", "1"]) == 1


def test_tag(notes_file, capsys):
    main(["add", "a", "--tag", "b"])
    assert main(["tag", "1", "a", "c"]) == 0
    capsys.readouterr()
    main(["list"])
    assert capsys.readouterr().out.strip().endswith("[a, b, c]")

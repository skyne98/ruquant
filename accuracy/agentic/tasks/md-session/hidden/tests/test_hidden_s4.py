import subprocess
import sys

from mdlite import to_html


def test_link():
    assert to_html("[site](https://x.org/?a=1&b=2)") == '<p><a href="https://x.org/?a=1&amp;b=2">site</a></p>'
    assert to_html("see [**bold** link](u) now") == '<p>see <a href="u"><strong>bold</strong> link</a> now</p>'


def test_no_link_in_code():
    assert to_html("`[not](link)`") == "<p><code>[not](link)</code></p>"


def test_link_in_item():
    assert to_html("- [a](b)") == '<ul>\n<li><a href="b">a</a></li>\n</ul>'


def test_heading_escaped():
    assert to_html("# a < b") == "<h1>a &lt; b</h1>"
    assert to_html("- x & y") == "<ul>\n<li>x &amp; y</li>\n</ul>"


def test_cli_file_and_stdin(tmp_path):
    src = "# Hi\n\n- *a*\n"
    f = tmp_path / "in.md"
    f.write_text(src)
    want = to_html(src) + "\n"
    r = subprocess.run([sys.executable, "-m", "mdlite", str(f)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout == want
    r = subprocess.run([sys.executable, "-m", "mdlite"], input=src, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout == want

from mdlite import to_html


def test_inline():
    assert to_html("**b** and *i* and `c`") == "<p><strong>b</strong> and <em>i</em> and <code>c</code></p>"


def test_code_span_is_literal():
    assert to_html("`*not* <em>`") == "<p><code>*not* &lt;em&gt;</code></p>"


def test_spaced_stars_stay():
    assert to_html("2 * 3 * 4") == "<p>2 * 3 * 4</p>"


def test_unclosed():
    assert to_html("an *unclosed marker") == "<p>an *unclosed marker</p>"


def test_heading_inline():
    assert to_html("# A **bold** title") == "<h1>A <strong>bold</strong> title</h1>"

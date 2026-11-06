from mdlite import to_html


def test_empty():
    assert to_html("") == ""


def test_example():
    assert to_html("# Title\nSome\ntext\n\nMore") == "<h1>Title</h1>\n<p>Some text</p>\n<p>More</p>"


def test_heading_levels():
    assert to_html("###### Six") == "<h6>Six</h6>"
    assert to_html("####### Seven") == "<p>####### Seven</p>"
    assert to_html("#NoSpace") == "<p>#NoSpace</p>"


def test_heading_breaks_paragraph():
    assert to_html("Para\n# Head\nAfter") == "<p>Para</p>\n<h1>Head</h1>\n<p>After</p>"


def test_blank_lines_and_whitespace():
    assert to_html("a\n   \n\n\nb") == "<p>a</p>\n<p>b</p>"
    assert to_html("  x  \n  y\n") == "<p>x y</p>"


def test_escaping():
    assert to_html("a < b & c > d") == "<p>a &lt; b &amp; c &gt; d</p>"

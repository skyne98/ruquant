from mdlite import to_html


def test_unordered():
    assert to_html("- a\n- *b*\n* c") == "<ul>\n<li>a</li>\n<li><em>b</em></li>\n<li>c</li>\n</ul>"


def test_ordered_split_by_blank():
    assert to_html("1. one\n2. two\n\n10. ten") == "<ol>\n<li>one</li>\n<li>two</li>\n</ol>\n<ol>\n<li>ten</li>\n</ol>"


def test_list_boundaries():
    assert to_html("intro\n- item\nafter") == "<p>intro</p>\n<ul>\n<li>item</li>\n</ul>\n<p>after</p>"
    assert to_html("- a\n1. b") == "<ul>\n<li>a</li>\n</ul>\n<ol>\n<li>b</li>\n</ol>"


def test_fence():
    src = "```python\nx = 1 < 2\n\n**no**\n```\nafter"
    assert to_html(src) == '<pre><code class="language-python">x = 1 &lt; 2\n\n**no**</code></pre>\n<p>after</p>'


def test_fence_plain_and_unclosed():
    assert to_html("```\n# not a heading\n  indented\n```") == "<pre><code># not a heading\n  indented</code></pre>"
    assert to_html("```\na\nb") == "<pre><code>a\nb</code></pre>"

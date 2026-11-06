import re

CODE = re.compile(r"`([^`]+)`")
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
STRONG = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
EM = re.compile(r"\*(?=\S)(.+?)(?<=\S)\*")
HEADING = re.compile(r"^(#{1,6}) (.*)$")
FENCE = re.compile(r"^```\s*(\S*)\s*$")
LISTS = (("ul", re.compile(r"^[-*] (.*)$")), ("ol", re.compile(r"^\d+\. (.*)$")))


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def emph(s):
    return EM.sub(r"<em>\1</em>", STRONG.sub(r"<strong>\1</strong>", s))


def links(t):
    out, pos = [], 0
    for m in LINK.finditer(t):
        url = m.group(2).replace("&", "&amp;").replace('"', "&quot;")
        out += [emph(esc(t[pos:m.start()])), f'<a href="{url}">{emph(esc(m.group(1)))}</a>']
        pos = m.end()
    return "".join(out + [emph(esc(t[pos:]))])


def inline(text):
    out, pos = [], 0
    for m in CODE.finditer(text):
        out += [links(text[pos:m.start()]), f"<code>{esc(m.group(1))}</code>"]
        pos = m.end()
    return "".join(out + [links(text[pos:])])


def to_html(text):
    lines = text.splitlines()
    out, para, i = [], [], 0

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    while i < len(lines):
        s = lines[i].strip()
        if not s:
            flush()
            i += 1
            continue
        if m := FENCE.match(lines[i]):
            flush()
            body, i = [], i + 1
            while i < len(lines) and lines[i] != "```":
                body.append(lines[i])
                i += 1
            i += 1
            cls = f' class="language-{esc(m.group(1))}"' if m.group(1) else ""
            out.append(f"<pre><code{cls}>{esc(chr(10).join(body))}</code></pre>")
            continue
        if m := HEADING.match(s):
            flush()
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2).strip())}</h{n}>")
            i += 1
            continue
        for kind, rx in LISTS:
            if rx.match(s):
                flush()
                items = []
                while i < len(lines) and (m := rx.match(lines[i].strip())):
                    items.append(f"<li>{inline(m.group(1).strip())}</li>")
                    i += 1
                out.append(f"<{kind}>\n" + "\n".join(items) + f"\n</{kind}>")
                break
        else:
            para.append(s)
            i += 1
    flush()
    return "\n".join(out)

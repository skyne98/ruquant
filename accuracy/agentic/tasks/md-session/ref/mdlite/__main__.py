import sys

from . import to_html

src = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
print(to_html(src))

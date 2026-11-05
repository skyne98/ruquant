#!/usr/bin/env python3
# Writes a C stub library for the symbols LIB imports but no library in PROVIDERS defines. Every stub function aborts
# with its own name. Used for hipBLASLt (rocBLAS imports it, but never calls it on gfx906) and rocSOLVER (hipBLAS
# imports it for LAPACK-style calls that ggml never makes).
#   mkstub.py LIB OUT.c PROVIDERS...
import subprocess, sys


def syms(lib, undefined):
    flag = "--undefined-only" if undefined else "--defined-only"
    out = subprocess.run(["nm", "-D", flag, lib], capture_output=True, text=True, check=True).stdout
    return {l.split()[-1].split("@")[0] for l in out.splitlines() if l.strip()}


lib, out, providers = sys.argv[1], sys.argv[2], sys.argv[3:]
have = set()
for p in providers:
    have |= syms(p, False)
need = sorted(s for s in syms(lib, True) - have if not s.startswith(("__gmon_start__", "_ITM_")))
with open(out, "w") as f:
    f.write("#include <stdio.h>\n#include <stdlib.h>\n")
    f.write("static void die(const char * s) { fprintf(stderr, \"stub library: %s called\\n\", s); abort(); }\n")
    for s in need:
        if s.startswith(("_ZTI", "_ZTV", "_ZTS")):
            f.write("char %s[64];\n" % s)
        else:
            f.write("void %s(void) { die(\"%s\"); }\n" % (s, s))
print("%s: %d stubs -> %s" % (lib, len(need), out))

#!/usr/bin/env python3
# Sets string metadata keys of a GGUF and copies the tensor data byte for byte (gguf-py's editor can't read the W4A4
# tensor types):
#   python3 rig/gguf_set_kv.py IN.gguf OUT.gguf general.quantized_by=ruquant [key=value ...]
# A key that is absent is added; an existing key must hold a string. Tensor offsets are relative to the data
# section, so only the header is rewritten.
import os, shutil, struct, sys

GGUF_STRING = 8
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

src, dst = sys.argv[1], sys.argv[2]
updates = dict(a.split("=", 1) for a in sys.argv[3:])
f = open(src, "rb")

def read(n):
    b = f.read(n)
    assert len(b) == n, "truncated header"
    return b

def raw_string():
    n = read(8)
    return n + read(struct.unpack("<Q", n)[0])

def raw_value(t):
    if t == GGUF_STRING:
        return raw_string()
    if t == 9:  # array: element type, count, elements
        head = read(12)
        et, n = struct.unpack("<IQ", head)
        if et == GGUF_STRING:
            return head + b"".join(raw_string() for _ in range(n))
        assert et != 9, "nested arrays"
        return head + read(FIXED[et] * n)
    return read(FIXED[t])

def enc_string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b

assert read(4) == b"GGUF"
version, n_tensors, n_kv = struct.unpack("<IQQ", read(20))
kvs = []
for _ in range(n_kv):
    key_raw = raw_string()
    key = key_raw[8:].decode()
    t = struct.unpack("<I", read(4))[0]
    kvs.append([key, t, raw_value(t)])
infos = read(0)
tensor_info = []
for _ in range(n_tensors):
    name = raw_string()
    nd = read(4)
    dims = read(8 * struct.unpack("<I", nd)[0])
    tensor_info.append(name + nd + dims + read(12))  # type (4) + offset (8)
alignment = 32
for key, t, v in kvs:
    if key == "general.alignment":
        alignment = struct.unpack("<I", v)[0]
data_start = (f.tell() + alignment - 1) // alignment * alignment

for key, value in updates.items():
    hit = [kv for kv in kvs if kv[0] == key]
    if hit:
        assert hit[0][1] == GGUF_STRING, key + " is not a string"
        hit[0][2] = enc_string(value)
    else:
        kvs.append([key, GGUF_STRING, enc_string(value)])

header = b"GGUF" + struct.pack("<IQQ", version, n_tensors, len(kvs))
header += b"".join(enc_string(k) + struct.pack("<I", t) + v for k, t, v in kvs)
header += b"".join(tensor_info)
header += b"\0" * ((-len(header)) % alignment)
with open(dst, "wb") as out:
    out.write(header)
    f.seek(data_start)
    shutil.copyfileobj(f, out, 64 << 20)
print("%s: %d keys, header %d -> %d bytes, data %d bytes" %
      (dst, len(kvs), data_start, len(header), os.path.getsize(src) - data_start))

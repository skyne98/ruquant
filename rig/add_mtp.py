#!/usr/bin/env python3
# Gives a ruquant-converted (rotated) Qwen3.5/3.8 GGUF a working MTP drafter for mx-llama.cpp's --spec-type draft-mtp.
# `ruquant convert` copies the MTP block unrotated, while the trunk's hidden state and the token embedding are in the
# rotated basis (R1 folded into the weights). This appends three tensors and leaves every other byte as it was:
#   blk.N.nextn.h_unrotate.weight        F16 [n_embd, n_embd]: diag(output_norm) R1^T, maps the trunk's normed
#                                        hidden state back to the MTP block's basis (from UNROT, raw row-major f16)
#   blk.N.nextn.embed_tokens.weight      the unrotated token embedding, copied from SRC
#   blk.N.nextn.shared_head_head.weight  the unrotated lm_head, copied from SRC
# mx-llama.cpp needs the nextn.h_unrotate patch (patches/mx-llama.cpp-ruquant.patch).
#   python3 rig/add_mtp.py MODEL SRC UNROT.f16 OUT
# Pure python (no numpy), so it runs on the MI50 boxes; the model is streamed, not loaded.
import os
import struct
import sys

# (block elements, block bytes) of the types read from SRC or written here
TYPE_SIZE = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
             12: (256, 144), 13: (256, 176), 14: (256, 210), 30: (1, 2)}
VALUE_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def rd_str(f):
    (n,) = rd(f, "Q")
    return f.read(n).decode()


def skip_value(f, t):
    if t in VALUE_SIZE:
        f.read(VALUE_SIZE[t])
    elif t == 8:
        rd_str(f)
    elif t == 9:
        at, n = rd(f, "IQ")
        if at in VALUE_SIZE:
            f.read(VALUE_SIZE[at] * n)
        else:
            for _ in range(n):
                skip_value(f, at)
    else:
        raise ValueError("unknown GGUF value type %d" % t)


def parse(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", path
    (version,) = rd(f, "I")
    n_tensors, n_kv = rd(f, "QQ")
    kv_start = f.tell()
    align, kv = 32, {}
    for _ in range(n_kv):
        key = rd_str(f)
        (t,) = rd(f, "I")
        if t == 4 and key == "general.alignment":
            (align,) = rd(f, "I")
        elif t == 4 and key.endswith(".block_count"):
            (kv[key],) = rd(f, "I")
        else:
            skip_value(f, t)
    infos_start = f.tell()
    tensors = {}
    for _ in range(n_tensors):
        name = rd_str(f)
        (nd,) = rd(f, "I")
        dims = rd(f, "Q" * nd)
        t, off = rd(f, "IQ")
        tensors[name] = (dims, t, off)
    infos_end = f.tell()
    data_start = (infos_end + align - 1) // align * align
    return dict(f=f, version=version, n_tensors=n_tensors, n_kv=n_kv, kv_start=kv_start, infos_start=infos_start,
                infos_end=infos_end, data_start=data_start, align=align, tensors=tensors, kv=kv)


def nbytes(dims, t):
    n = 1
    for d in dims:
        n *= d
    blk, size = TYPE_SIZE[t]
    assert n % blk == 0
    return n // blk * size


def tensor_info(name, dims, t, off):
    b = name.encode()
    return struct.pack("<Q", len(b)) + b + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims) + \
        struct.pack("<IQ", t, off)


def main():
    model_path, src_path, unrot_path, out_path = sys.argv[1:5]
    m, s = parse(model_path), parse(src_path)
    n_layer = next(v for k, v in m["kv"].items() if k.endswith(".block_count"))
    mtp = n_layer - 1
    emb = s["tensors"]["token_embd.weight"]
    head = s["tensors"]["output.weight"]
    n_embd = emb[0][0]
    assert os.path.getsize(unrot_path) == n_embd * n_embd * 2, "UNROT must be n_embd x n_embd f16"
    new = [  # name, dims, type, (file, absolute offset)
        ("blk.%d.nextn.h_unrotate.weight" % mtp, (n_embd, n_embd), 1, (open(unrot_path, "rb"), 0)),
        ("blk.%d.nextn.embed_tokens.weight" % mtp, emb[0], emb[1], (s["f"], s["data_start"] + emb[2])),
        ("blk.%d.nextn.shared_head_head.weight" % mtp, head[0], head[1], (s["f"], s["data_start"] + head[2])),
    ]
    for name, *_ in new:
        assert name not in m["tensors"], name + " is already in the model"
    align = m["align"]
    pad = lambda x: (x + align - 1) // align * align
    data_len = os.path.getsize(model_path) - m["data_start"]
    off = pad(data_len)
    infos = b""
    offsets = []
    for name, dims, t, _ in new:
        infos += tensor_info(name, dims, t, off)
        offsets.append(off)
        off = pad(off + nbytes(dims, t))

    f = m["f"]
    with open(out_path, "wb") as out:
        out.write(b"GGUF" + struct.pack("<IQQ", m["version"], m["n_tensors"] + len(new), m["n_kv"]))
        f.seek(m["kv_start"])
        out.write(f.read(m["infos_end"] - m["kv_start"]))  # KV pairs and the existing tensor infos, unchanged
        out.write(infos)
        out.write(b"\0" * (pad(out.tell()) - out.tell()))
        base = out.tell()
        f.seek(m["data_start"])
        while True:
            chunk = f.read(64 << 20)
            if not chunk:
                break
            out.write(chunk)
        for (name, dims, t, (sf, soff)), o in zip(new, offsets):
            out.write(b"\0" * (base + o - out.tell()))
            sf.seek(soff)
            left = nbytes(dims, t)
            while left > 0:
                chunk = sf.read(min(left, 64 << 20))
                out.write(chunk)
                left -= len(chunk)
            print("%s %s type %d at %d" % (name, dims, t, o))
    print("wrote %s (%d tensors)" % (out_path, m["n_tensors"] + len(new)))


if __name__ == "__main__":
    main()

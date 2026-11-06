#!/usr/bin/env python3
"""A vision projector (mmproj GGUF) for our rotated W4A4 model.

    PYTHONPATH=build-llama/src/gguf-py .venv/bin/python accuracy/rotate_mmproj.py IN.gguf R1.f32 OUT.gguf

Our model's residual stream is rotated: its token embeddings are the original's times R1 (src/convert.rs, R1 from
--ost DIR/r1.f32, y = x R1 on rows of 5120). The image embeddings the projector feeds into the same stream must be
rotated too, or the model sees noise. Qwen3.8's projector (qwen3vl_merger, no DeepStack layers) writes the stream
through one linear layer, mm.2 (4608 -> 5120): e = W x + b, so e R1 = (R1^T W) x + b R1. This copies IN with
mm.2.weight <- R1^T W and mm.2.bias <- b R1, both kept in f32, and every other tensor unchanged.

IN is the original model's mmproj (convert_hf_to_gguf.py Qwen/Qwen3.8-27B --mmproj).
"""
import sys

import gguf
import numpy as np


def main():
    src, r1_path, out = sys.argv[1:4]
    reader = gguf.GGUFReader(src)
    r1 = np.fromfile(r1_path, dtype=np.float32).astype(np.float64)
    n = int(round(len(r1) ** 0.5))
    r1 = r1.reshape(n, n)
    assert np.abs(r1 @ r1.T - np.eye(n)).max() < 1e-2, "R1 is not orthogonal"

    tensors = {t.name: t for t in reader.tensors}
    w = tensors["mm.2.weight"].data.astype(np.float64)  # [out = n, in]
    b = tensors["mm.2.bias"].data.astype(np.float64)    # [n]
    assert w.shape[0] == n and b.shape == (n,), (w.shape, b.shape)
    new = {
        "mm.2.weight": (r1.T @ w).astype(np.float32),
        "mm.2.bias": (b @ r1).astype(np.float32),
    }
    x = np.random.default_rng(0).standard_normal(w.shape[1])
    err = np.abs((w @ x + b) @ r1 - (new["mm.2.weight"] @ x + new["mm.2.bias"])).max()
    print(f"R1 {n}x{n}; mm.2 {w.shape[1]} -> {n}; max |(Wx+b)R1 - (W'x+b')| = {err:.2e}")

    writer = gguf.GGUFWriter(out, arch=reader.fields[gguf.Keys.General.ARCHITECTURE].contents())
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if field.name == gguf.Keys.General.NAME:
            value = "Qwen3.8-27B W4A4 (ruquant) mmproj"
        elif field.name == gguf.Keys.General.DESCRIPTION:
            value = ("Vision encoder + projector of Qwen/Qwen3.8-27B for raspbfox/Qwen3.8-27B-W4A4-GGUF, its output "
                     "rotated by the model's residual rotation R1 (mm.2: W <- R1^T W, b <- b R1)")
        writer.add_key_value(field.name, value, vtype, sub_type=sub)
    if gguf.Keys.General.DESCRIPTION not in reader.fields:
        writer.add_description("Vision encoder + projector of Qwen/Qwen3.8-27B for raspbfox/Qwen3.8-27B-W4A4-GGUF, its "
                               "output rotated by the model's residual rotation R1 (mm.2: W <- R1^T W, b <- b R1)")

    for t in reader.tensors:
        if t.name in new:
            d = new[t.name]
            writer.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(new[t.name] if t.name in new else t.data, tensor_endianess=reader.endianess)
    writer.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

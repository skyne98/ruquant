# Measured on gfx906

The JSON behind the README's "Measured on gfx906" section, from `rig/` runs on two borrowed
boxes (probe dumps of the hosts left out):

- `mi50x4-rocm57/`: four MI50 32 GB, ROCm 5.7.1, PCIe gen3 through the CPU, P2P on.
  `skinny-gpu0.json` is the kernel with activation prefetch and the shared LDS buffer;
  `skinny-gpu0-before-fixes.json` is the same GPU before both.
- `provii-x2-rocm101/`: two Radeon Pro VII 16 GB (gfx906, 60 CUs), ROCm 10.1, P2P off.
  `skinny-gpu0.json` (current kernel), `ggml-gpu0.json` (our vendored ggml) and
  `ggml-mx-gpu0.json` (the box owner's llama.cpp build) all ran on the same GPU.

```sh
python3 rig/calibrate.py rig/measured/provii-x2-rocm101
python3 gfx906/model.py <flags it prints> --measured rig/measured/provii-x2-rocm101
```

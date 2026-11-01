#!/usr/bin/env python3
"""Static instruction mix of AMDGPU kernels, read from compiler asm (hipcc -S).

For every kernel: resource usage (VGPRs, SGPRs, LDS, occupancy, from the compiler's own
comments) and, for every loop, the instruction mix of one iteration, split by the pipe it
issues to. Loops are found from back edges in the block layout (LLVM emits loops
contiguously), so a loop's blocks are the ones between the branch target and the branch.
In branchy code outside loops a jump to an earlier block can show up as a "loop" too;
model.py only uses the loops that hold the dot instructions.

    isa.py file.s [--kernel REGEX] [--json]
"""

import json
import re
import sys
from dataclasses import dataclass, field

LANES = 64  # gfx906 wave64

# instruction -> pipe. Order matters: first match wins.
CLASSES = [
    ("dot8", re.compile(r"^v_dot8_[iu]32_[iu]4$")),
    ("dot4", re.compile(r"^v_dot4_[iu]32_[iu]8$")),
    ("dot2", re.compile(r"^v_dot2_")),
    ("valu", re.compile(r"^v_")),
    ("smem", re.compile(r"^s_(load|buffer_load|store|buffer_store|dcache|memtime|memrealtime)")),
    ("wait", re.compile(r"^s_(waitcnt|barrier|nop|sleep|setprio|sethalt|endpgm|trap)")),
    ("branch", re.compile(r"^s_(branch|cbranch_|setpc|swappc|getpc)")),
    ("salu", re.compile(r"^s_")),
    ("lds", re.compile(r"^ds_")),
    ("vmem", re.compile(r"^(global|buffer|flat|scratch|image|tbuffer)_")),
]
MACS = {"dot8": 8, "dot4": 4, "dot2": 2}

# bytes per lane moved by a memory instruction, from its suffix
SIZES = [
    (re.compile(r"read2st64_b64|read2_b64|write2_b64|write2st64_b64"), 16),
    (re.compile(r"read2st64_b32|read2_b32|write2_b32|write2st64_b32"), 8),
    (re.compile(r"dwordx4|_b128"), 16),
    (re.compile(r"dwordx3|_b96"), 12),
    (re.compile(r"dwordx2|_b64"), 8),
    (re.compile(r"dword|_b32"), 4),
    (re.compile(r"short|_b16|_u16|_i16"), 2),
    (re.compile(r"byte|_b8|_u8|_i8"), 1),
]

BRANCH_TARGET = re.compile(r"^s_(?:branch|cbranch_\w+)\s+(\.\w+)")
LABEL = re.compile(r"^(\.LBB\w+|\S+):(\s*;.*)?$")


def mem_bytes(op: str) -> int:
    for pat, n in SIZES:
        if pat.search(op):
            return n
    return 4


@dataclass
class Mix:
    counts: dict = field(default_factory=dict)
    load_bytes: dict = field(default_factory=lambda: {"vmem": 0, "lds": 0, "smem": 0})
    store_bytes: dict = field(default_factory=lambda: {"vmem": 0, "lds": 0})

    def add(self, op: str):
        cls = next((c for c, pat in CLASSES if pat.match(op)), "other")
        self.counts[cls] = self.counts.get(cls, 0) + 1
        if cls in ("vmem", "lds"):
            is_store = "store" in op or "write" in op
            is_atomic = "atomic" in op or op.startswith("ds_add")
            n = mem_bytes(op)
            if is_store or is_atomic:
                self.store_bytes[cls] += n
            else:
                self.load_bytes[cls] += n
        elif cls == "smem" and "load" in op:
            m = re.search(r"dwordx(\d+)", op)
            self.load_bytes["smem"] += 4 * (int(m.group(1)) if m else 1)

    def macs(self) -> int:
        """Multiply-adds per wave per pass."""
        return LANES * sum(MACS[c] * self.counts.get(c, 0) for c in MACS)

    def summary(self) -> dict:
        c = self.counts
        dots = sum(c.get(k, 0) for k in MACS)
        macs = self.macs()
        out = {
            "counts": dict(sorted(c.items())),
            "macs_per_wave": macs,
            "lds_bytes_per_wave": LANES * (self.load_bytes["lds"] + self.store_bytes["lds"]),
            "vmem_bytes_per_wave": LANES * (self.load_bytes["vmem"] + self.store_bytes["vmem"]),
            "smem_bytes_per_wave": self.load_bytes["smem"],
        }
        if dots:
            valu = c.get("valu", 0)
            out["non_dot_valu_per_dot"] = valu / dots
            # wave-level VALU issue slots per 8 multiply-adds per lane: 1.0 = pure dot8
            out["valu_slots_per_8mac"] = (valu + dots) / (macs / LANES / 8)
        return out


@dataclass
class Block:
    label: str
    ops: list
    targets: list


@dataclass
class Kernel:
    name: str
    blocks: list
    meta: dict


def parse(path: str) -> list:
    kernels_declared = set()
    text = open(path).read().splitlines()
    for line in text:
        m = re.match(r"\s*\.amdhsa_kernel\s+(\S+)", line)
        if m:
            kernels_declared.add(m.group(1))

    kernels, cur, block = [], None, None
    for raw in text:
        line = raw.split(";", 1)[0].strip() if not raw.lstrip().startswith(";") else ""
        comment = raw.strip()
        m = LABEL.match(raw.strip())
        if m and m.group(1) in kernels_declared:
            cur = Kernel(m.group(1), [], {})
            block = Block("entry", [], [])
            cur.blocks.append(block)
            kernels.append(cur)
            continue
        if cur is None:
            continue
        if comment.startswith(".Lfunc_end"):
            cur, block = None, None
            continue
        if m and m.group(1).startswith(".LBB"):
            block = Block(m.group(1), [], [])
            cur.blocks.append(block)
            continue
        if not line or line.startswith("."):
            continue
        op = line.split()[0]
        block.ops.append(op)
        t = BRANCH_TARGET.match(line)
        if t:
            block.targets.append(t.group(1))

    # resource comments follow the function body
    meta_pat = re.compile(r";\s*(NumVgprs|NumSgprs|Occupancy|ScratchSize|LDSByteSize|NumVGPRsForWavesPerEU|NumSGPRsForWavesPerEU|codeLenInByte):\s*(\d+)")
    by_name = {k.name: k for k in kernels}
    current = None
    for raw in text:
        m = re.match(r"\s*\.amdhsa_kernel\s+(\S+)", raw)
        hdr = re.match(r"; Kernel info:", raw.strip())
        fn = re.match(r";\s*-+\s*$", raw.strip())
        mm = re.match(r"^(\S+):(\s*;.*)?$", raw.strip())
        if mm and mm.group(1) in by_name:
            current = by_name[mm.group(1)]
        mt = meta_pat.search(raw)
        if mt and current is not None:
            current.meta.setdefault(mt.group(1), int(mt.group(2)))
    return kernels


def loops(k: Kernel) -> list:
    """(head_index, tail_index) for every back edge, innermost first."""
    index = {b.label: i for i, b in enumerate(k.blocks)}
    found = []
    for i, b in enumerate(k.blocks):
        for t in b.targets:
            j = index.get(t)
            if j is not None and j <= i:
                found.append((j, i))
    return sorted(set(found), key=lambda r: r[1] - r[0])


def analyse(k: Kernel) -> dict:
    total = Mix()
    for b in k.blocks:
        for op in b.ops:
            total.add(op)
    out = {"kernel": k.name, "meta": k.meta, "whole_kernel": total.summary(), "loops": []}
    for head, tail in loops(k):
        mix = Mix()
        for b in k.blocks[head : tail + 1]:
            for op in b.ops:
                mix.add(op)
        s = mix.summary()
        s["blocks"] = [k.blocks[head].label, k.blocks[tail].label]
        s["range"] = [head, tail]
        out["loops"].append(s)
    return out


def main():
    args = sys.argv[1:]
    as_json = "--json" in args
    pat = None
    if "--kernel" in args:
        pat = re.compile(args[args.index("--kernel") + 1])
    path = [a for a in args if a.endswith(".s")][0]
    results = [analyse(k) for k in parse(path) if pat is None or pat.search(k.name)]
    if as_json:
        json.dump(results, sys.stdout, indent=1)
        return
    for r in results:
        m = r["meta"]
        print(f"\n{r['kernel']}")
        print(f"  vgpr {m.get('NumVgprs')}  sgpr {m.get('NumSgprs')}  lds {m.get('LDSByteSize')}  "
              f"occupancy {m.get('Occupancy')} waves/SIMD  scratch {m.get('ScratchSize')}")
        for i, lp in enumerate(r["loops"]):
            c = lp["counts"]
            dots = {k: c[k] for k in MACS if k in c}
            line = "  ".join(f"{k} {v}" for k, v in sorted(c.items()))
            print(f"  loop {i} {lp['blocks'][0]}..{lp['blocks'][1]}: {line}")
            if dots:
                print(f"      non-dot VALU per dot {lp['non_dot_valu_per_dot']:.2f}   "
                      f"VALU slots per 8 MAC {lp['valu_slots_per_8mac']:.2f}   "
                      f"LDS B/wave {lp['lds_bytes_per_wave']}  VMEM B/wave {lp['vmem_bytes_per_wave']}")


if __name__ == "__main__":
    main()

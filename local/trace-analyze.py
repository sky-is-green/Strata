#!/usr/bin/env python3
"""local/trace-analyze.py trace_short.csv trace_long.csv N

Two rocprofv3 kernel traces of the same prompt that differ only in how many
tokens were decoded (``-n 3`` vs ``-n 7``, say).  The *difference in dispatch
count* tells us how many decode steps are in the long trace's tail; everything
in that tail is then broken down directly, without subtracting one run from the
other (subtraction mixes in the model load and the graph-capture warmup, and it
disagreed with itself when tried).

Prints: dispatch count and kernel-busy time per trace and per token, the
per-kernel and per-family breakdown of the trailing decode steps, and the gap
structure.  Two caveats, both from measurement: kernel tracing slows this engine
about 20% and adds ~2 us between dispatches, so *gaps* in a trace are not the
engine's real gaps; and every row is a graph node, not a host launch (this build
captures its graphs), so the row count is not a launch count.

Timestamps in rocprofv3 CSVs are nanoseconds.
"""
import csv
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
            except (KeyError, ValueError):
                continue
            rows.append((s, e, r["Kernel_Name"]))
    rows.sort()
    return rows


def family(name):
    if "native_" in name:
        return "tier fused expert kernels"
    if "copyBuffer" in name:
        return "PCIe copies (tier)"
    if "mul_mat_vec" in name:
        return "GEMV"
    if "flash_attn" in name:
        return "attention"
    if "hc_" in name:
        return "hyper-connections"
    if any(k in name for k in ("rms_norm", "rope", "k_bin_bcast", "scale_f32",
                               "unary", "concat", "cpy_", "get_rows", "set_rows",
                               "reduce_rows", "add")):
        return "elementwise/norm/rope"
    if any(k in name for k in ("quantize", "swiglu")):
        return "quantize/swiglu"
    if any(k in name for k in ("argsort", "topk", "top_k")):
        return "top-k/argsort"
    return "other"


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ntok = int(sys.argv[3])
    a, b = load(sys.argv[1]), load(sys.argv[2])

    ca, cb = len(a), len(b)
    ta = sum(e - s for s, e, _ in a) / 1e6
    tb = sum(e - s for s, e, _ in b) / 1e6
    print(f"trace A: {ca:7d} rows  {ta:9.2f} ms kernel-busy")
    print(f"trace B: {cb:7d} rows  {tb:9.2f} ms kernel-busy")
    if cb <= ca:
        sys.exit("trace B must contain the longer decode")
    window = cb - ca
    print(f"decode rows in B's tail: {window} ({window / ntok:.0f}/token), "
          f"{ntok} requested tokens -> {(tb - ta) / ntok:.2f} ms kernel-busy/token")

    tail = b[-window:]
    busy = sum(e - s for s, e, _ in tail) / 1e6
    print(f"trailing window: {busy:.1f} ms kernel-busy = {busy / ntok:.2f} ms/token")

    per = defaultdict(float)
    cnt = defaultdict(int)
    for s, e, n in tail:
        per[n] += (e - s) / 1e6
        cnt[n] += 1
    print("\ntop kernels (per token):")
    print(f"  {'ms':>7} {'calls':>7} {'us each':>8}  kernel")
    for n, v in sorted(per.items(), key=lambda kv: -kv[1])[:14]:
        print(f"  {v / ntok:7.2f} {cnt[n] / ntok:7.1f} {1e3 * v / cnt[n]:8.1f}  {n[:62]}")

    fam = defaultdict(float)
    for n, v in per.items():
        fam[family(n)] += v
    print("\nby family (ms/token):")
    for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
        print(f"  {v / ntok:6.2f}  {k}")

    gaps = [tail[i + 1][0] - tail[i][1] for i in range(len(tail) - 1)]
    span = (tail[-1][1] - tail[0][0]) / 1e6
    print(f"\nidle between rows: {sum(gaps) / 1e6:.1f} ms over {len(gaps)} gaps "
          f"(trace inflation included; span {span:.1f} ms)")
    for lo, hi, label in ((0, 10_000, "<10us"), (10_000, 100_000, "10-100us"),
                          (100_000, 1 << 62, ">100us")):
        g = [x for x in gaps if lo <= x < hi]
        print(f"  {label:9s} {len(g):6d} gaps  {sum(g) / 1e6:7.1f} ms")


if __name__ == "__main__":
    main()

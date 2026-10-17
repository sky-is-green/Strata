#!/usr/bin/env python3
"""local/engram_index.py - merge the per-layer Engram indexes into the one index the engine reads.

  local/engram_index.py DIR_WITH_TABLES...

`tools/v41/extract_engram.py --to layers.L=DIR` writes one engram-L*.safetensors and an index.json per destination
holding only that layer's tensors.  Ds41Engram::open() reads the FIRST model.safetensors.index.json it finds among the
--engram directories and resolves every tensor's filename against all of them, so with the tables split over two
drives the index has to name all four tensors at once.  This writes that merged index into every directory given
(the filenames differ per layer, so the per-directory lookup stays unambiguous).
"""
import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(2)
dirs = [Path(d) for d in sys.argv[1:]]
merged, size = {}, 0
for d in dirs:
    p = d / "model.safetensors.index.json"
    if not p.exists():
        sys.exit(f"no index in {d}")
    ix = json.loads(p.read_text())
    merged.update(ix["weight_map"])
    size += int(ix.get("metadata", {}).get("total_size", 0))
for d in dirs:
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": size}, "weight_map": merged}, indent=1))
print(f"merged index ({len(merged)} tensors, {size / 1e9:.1f} GB) written to " + ", ".join(str(d) for d in dirs))
for k in sorted(merged):
    print(f"  {k:40s} {merged[k]}")

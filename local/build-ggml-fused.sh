#!/usr/bin/env bash
# local/build-ggml-fused.sh - build ggml-hip from the PATCHED llama.cpp tree in third_party/llama.cpp, then the
# engine with the fork's two fused dense ops on (STRATA_DSV41_FQ_OP + STRATA_DSV41_HC_MIX).
#
# Why: on this box a decode token's dense half runs ~3300 tiny ggml ops, and the fork's fused ops collapse the
# activation rounding (GGML_OP_DSV41_FAKE_QUANT) and the two hyper-connection mixes (GGML_OP_DSV41_HC_MIXES) into
# one node each.  The author measured +5% decode on CUDA (26.3 -> 27.7 tok/s); HIP launch overhead is higher, so it
# is worth re-measuring here.  The patches only apply to the patched tree, so ggml has to be built once from it
# (~40+ min, and it must NOT run while an engine holds its 75 GiB arena: 20 hipcc jobs want ~80 GB).
#
# Usage: local/build-ggml-fused.sh [jobs]        (default 16)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS="${1:-16}"
ROCMP="${ROCM_PATH:-/opt/rocm}"
LLAMA="$PWD/third_party/llama.cpp"
GB="$LLAMA/build-hip"

[ -f "$LLAMA/ggml/include/ggml.h" ] || { echo "no llama.cpp tree in $LLAMA" >&2; exit 1; }
grep -q "GGML_OP_DSV41_FAKE_QUANT" "$LLAMA/ggml/include/ggml.h" || {
    echo "the tree is not patched: run /usr/bin/patch -p1 -d $LLAMA < tools/v41/ggml/dsv41-fake-quant.patch" >&2; exit 1; }
grep -q "GGML_OP_DSV41_HC_MIXES" "$LLAMA/ggml/include/ggml.h" || {
    echo "the tree is not patched: run /usr/bin/patch -p1 -d $LLAMA < tools/v41/ggml/dsv41-fuse.patch" >&2; exit 1; }

echo "== ggml-hip (patched tree) -> $GB"
cmake -S "$LLAMA" -B "$GB" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER="$ROCMP/lib/llvm/bin/clang++" \
      -DCMAKE_PREFIX_PATH="$ROCMP" -DGGML_STATIC=OFF -DBUILD_SHARED_LIBS=OFF \
      -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF > local/build-ggml-fused-cfg.log 2>&1
cmake --build "$GB" --target ggml ggml-cpu ggml-hip -j "$JOBS"

echo "== engine with the fused ops -> build-hip-fused"
cmake -S . -B build-hip-fused -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_HIP_COMPILER="$ROCMP/lib/llvm/bin/clang++" -DCMAKE_PREFIX_PATH="$ROCMP" \
      -DSTRATA_GGML_DIR="$LLAMA" -DSTRATA_GGML_PREBUILT="$GB" \
      -DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON \
      -DBUILD_SHARED_LIBS=ON -DSTRATA_BUILD_TESTS=OFF > local/build-hip-fused-cfg.log 2>&1
make -C build-hip-fused -j "$JOBS" ds41_generate
echo "done: build-hip-fused/ds41_generate (compare it against build-hip-pre/ds41_generate with local/bench_ds41.py)"

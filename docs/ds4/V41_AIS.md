# NVMe to VRAM on this box (AIS / hipFile)

Status: **working as of 2026-10-23.** The kernel supports it, the HIP runtime supports it, the cards qualify, and
the direct path measures **1.5-1.7x** the staged path. What is not done yet is the engine using it.

## What was missing, exactly

AMD's AIS (AMD Infinity Storage) is the mechanism: at driver load it registers the whole VRAM region as
`MEMORY_DEVICE_PCI_P2PDMA` pages, and a new KFD ioctl (`AMDKFD_IOC_AIS_OP`, `0x87`) has the driver fill a `bio_vec`
with VRAM pages so the **NVMe controller DMAs straight into VRAM** with no host bounce. First commit 2025-07-22
(+438 lines, 6 files); 26 commits later it is 1,446 lines in `kfd_ais.c` plus hooks.

It was **never upstreamed** - `kfd_ais.c` is not in torvalds/linux - so kernel 7.3.0-cachyos does not have it. AMD
ships it only in `ROCm/amdgpu` (a full kernel tree, base 7.1.0).

Three checks decide whether a box can use it (`tools/ais-check`), and this box failed only the third:

| check | before | after |
|---|---|---|
| kernel `CONFIG_PCI_P2PDMA` | True | True |
| HIP runtime `hipAmdFileRead`/`Write` | True (ROCm 7.2.4 exports them) | True |
| amdgpu `kfd_ais_rw_file` in kallsyms | **False** | **True** |

The hardware gate in `kfd_ais_init` is: not SRIOV, not an APU, not XGMI-connected, and **large BAR**
(`visible_vram == real_vram`). This box passes all of them (20.0 GiB visible == 20.0 GiB real, Re-Size BAR is on).

## The port

Rather than swap kernels, the AIS patch set was ported onto the **exact running kernel source** (CachyOS
`cachyos-7.3.0-4`, fetched from their release tarball) so that only `amdgpu.ko` changes. 14 files, ~1,700 lines.

Drift that had to be handled (all of it small, none of it conceptual):

| issue | fix |
|---|---|
| 7.3's handle idr stores `kgd_mem *`; ROCm wraps it in `struct kfd_bo` | handler uses `kfd_process_device_translate_handle` + `mem->bo` / `mem->domain` |
| `amdgpu_amdkfd_gpuvm_pin_bo` / `unpin_bo` are `static` in mainline | exported + prototyped |
| capability query used ROCm's `amdgpu_no_evict` (not in mainline) | reports no pin-mem flag; hipFile does not read it |
| sysfs transfer counters need `struct ais_counter_entry`, `kobj_ais`, `ais_counters_xa` | added to `kfd_priv.h` |
| `kfd_process_device_find_bo` | not needed on 7.3, dropped |
| `AMDKFD_IOCTL_DEF` takes 3 args in 7.3, 4 in 7.1 | entry corrected |

Dropped as unnecessary: NFS, NVMeoF-RDMA, LVM/device-mapper (plain local NVMe only). The NFS code compiles
dormant behind `CONFIG_NFS_FS`.

## The gotcha that cost a boot: module BTF

The first install loaded the stock `simpledrm` instead of amdgpu and ROCm saw no devices at all. `dmesg`:

```
amdgpu: module verification failed: signature ... tainting kernel     (harmless)
failed to validate module [amdgpu] BTF: -22                           (fatal)
```

Module BTF is **split**: its type IDs point into the running kernel's BTF. Ours were generated against *our*
vmlinux, and the running kernel is the stock CachyOS build, so the link cannot resolve and the kernel refuses the
module. The fix is to ship the module **without `.BTF`** - BTF is optional for loading and amdgpu has no kfuncs
that need it:

    objcopy --remove-section=.BTF --remove-section=.BTF.ext --strip-debug amdgpu.ko

That took 846 MB of debug info down to 35 MB and it loads. (An unsigned module only taints the kernel;
`module.sig_enforce` is off here.)

## Install and rollback

`local/install-ais-module.sh` (root) has three modes. Our module goes to `/lib/modules/<rel>/updates/`, which
`depmod` prefers over `kernel/`, so the stock module is never touched:

- `stage1` - install and boot with `amdgpu.ais_disabled=1`, so a driver fault cannot be confused with an AIS fault.
- `stage2` - drop the override, `limine-mkinitcpio`, reboot, and AIS initialises.
- `rollback` - remove `updates/amdgpu.ko.zst`, `depmod`, `limine-mkinitcpio`, reboot.

On this box `mkinitcpio -P` does nothing useful (no `/etc/mkinitcpio.d` presets); the tool is `limine-mkinitcpio`,
which builds the initramfs with explicit arguments. amdgpu loads early for KMS, so a stale initramfs boots the
*stock* module and nothing appears to change - that step is not optional.

Success looks like this in `dmesg`, once per card:

    amdgpu 0000:03:00.0: AIS: registered 20464MB device memory

and `ais-check` prints True / True / True.

## The two constraints that matter for where files live

1. **Filesystem: ext4 or xfs only.** hipFile refuses anything else in its backend scoring (`m_on_xfs`,
   `m_on_ext4_ordered`). This is policy, not physics: reads only need O_DIRECT. Our build carries a local patch
   (`src/amd_detail/mountinfo.cpp`) that classifies **btrfs as xfs**; that is what let the working numbers below be
   measured on the work drive. Never used for writes.
2. **P2P distance is real.** With the same xfs file on the **Micron** (`13:00.0`, six hops behind the chipset) the
   kernel's AIS ioctl fails and HIP reports `hipErrorUnknown`. The **P310** at `00:01.2` -> `04:00.0`, on the same
   root-complex class as both GPUs, works. So the model has to live on the **P310**, not the Micron.

## Measured (2026-10-23, P310, XFS-compatible path, `HIPFILE_ALLOW_COMPAT_MODE=0`)

`kbuild/aistest.cpp`: 4 KiB-aligned reads of random offsets into one `hipMalloc` buffer, 24 samples, p50.
`hipfile` = `hipFileRead` (drive DMAs into VRAM); `staged` = `O_DIRECT` pread into host + `hipMemcpy`.

| block | hipfile | staged | gain |
|---|---:|---:|---:|
| 4 MiB | 0.64 ms = **6.24 GB/s** | 1.10 ms = 3.63 GB/s | 1.72x |
| 8 MiB | 1.27 ms = **6.31 GB/s** | 1.93 ms = 4.14 GB/s | 1.53x |
| 32 MiB | 4.24 ms = **7.54 GB/s** | 7.13 ms = 4.49 GB/s | 1.68x |

**Watch the env var syntax**: `HIPFILE_ALLOW_COMPAT_MODE=0` disables the bounce path; `=false` is not parsed and
silently leaves the fallback enabled, which fails inside hipFile (`-5030`) and hides the real error.

## What this is worth to the engine, and what it still needs

The engine's file tier currently costs 3.5 ms/token *after* `STRATA_ADVISE_AHEAD`, but that prefetch costs a predict
graph per layer (~5.6 ms of `predict`). A direct read at 6.3 GB/s is fast enough to sit on the critical path
(1.3 ms for a 7.75 MB blob), which would let the prefetch and its predict graphs go: about **10% of a token** at
today's 75 ms.

Two things it needs first:

- **Alignment.** hipFile requires 4 KiB-aligned file *and* memory offsets. GGUF blob offsets are not aligned, so
  either the engine reads aligned windows and uses the sub-range, or it uses the blob mirror
  (`--build-mirror`, which pads every blob to 4096). The P310 has only 55 GB free, so a partial mirror
  (`--mirror-gib`) is the realistic start.
- **The read must land in the VRAM slot itself.** `ds41_moe.cpp`'s file-tier read currently goes O_DIRECT into a
  pinned host scratch and then DMAs; the win is only real if the expert blob goes straight into its cache slot.

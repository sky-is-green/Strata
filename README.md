<h1 align="center">Strata-DSV4.1 - AMD (gfx1100) branch</h1>

<p align="center"><b>DeepSeek-V4.1-Flash (40 layers x 384 experts) on ONE RX 7900 XTX - 152 GiB of experts across VRAM, pinned RAM and NVMe, plus 189 GiB of Engram on disk</b><br>
The AMD port and tuning of <a href="https://github.com/Indras-Mirror/Strata-DSV4.1">Strata-DSV4.1</a>, itself built on <a href="https://github.com/Niko1221/Strata">Strata</a> - Linux - ROCm/HIP - MIT</p>

<p align="center"><b>RX 7900 XTX (gfx1100, ROCm 7.14) + 128 GiB RAM:&nbsp; prefill 240-363 tok/s &nbsp;·&nbsp; DSpark decode 24.3 tok/s &nbsp;·&nbsp; 256K context, 8-bit KV</b></p>

## What this branch is

This is the **AMD (gfx1100)** branch of the fork `vincalkr/Strata-DSV4.1`: the same engine and serve as upstream, ported
and tuned to run DeepSeek-V4.1-Flash on **one** RX 7900 XTX.  Every number on this page was measured on that machine;
the engine runs on a single card (`"gpu": 0` in [local/ds41-local.json](local/ds41-local.json)).

The platform is not ours.  The engine, the serve and the earlier DeepSeek-V4-Flash (0731) path come from
[Indras-Mirror/Strata-DSV4.1](https://github.com/Indras-Mirror/Strata-DSV4.1), which builds on
[Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT; `LICENSE` is kept as it is).  Their README - the NVIDIA
measurements, the 512K-context settings, the API and the V4-Flash engine - is kept verbatim in
**[docs/upstream/README.md](docs/upstream/README.md)**, with the localized copies next to it; read it for anything that
is not AMD-specific.  Our contribution is the AMD path: the port, the tuning, the measurements and these notes.

## The machine, and what is on disk

Two RX 7900 XTX are installed, but **one** is used.  128 GiB of RAM, one 2 TB NVMe, ROCm 7.14.  On disk:
`models/V41-GSQ/V41-GSQ.gguf` (152 GiB), the 5.3 GiB DSpark draft with its small files, and the two Engram tables in
`models/engram/` (189 GiB, layers 1 and 14).  The second card was tried as an expert tier and dropped: behind its
PCIe 4.0 **x2** bridge it is worth +10-20% as extra cache, not 2x (see the negatives below).

| measured here, one sequence | |
| --- | ---: |
| decode, 4K state, DSpark off | 13.7 tok/s |
| decode, 64K state, DSpark off | 14.0 tok/s |
| decode through the server, DSpark depth 4, ggml graphs off | 22.8 -> 24.3 tok/s |
| prompt, 5787 tokens | 240-245 tok/s |
| prompt, 63,617 tokens, fresh, through the server | 363 tok/s |

## The optimizations, each with what it was worth

| change | measured effect |
| --- | --- |
| ggml's fused dense ops (`local/build-ggml-fused.sh`: `dsv41-fake-quant.patch` + `dsv41-fuse.patch`) | +13% decode, +23% with DSpark; last-row logits top-1 identical, mean absolute difference 0.156 |
| `--arena-gib 75` (from 60) | +40% decode (16.4-17.4 -> 23.8-25.2 tok/s), prompt file wait 16.6 -> 11.0 s; 80 and above are worse or unstable |
| `GGML_CUDA_DISABLE_GRAPHS=1` | +6.6% decode, +3% prompt in the server |
| `STRATA_PRESTAGE_GIB=1.5` | +5.8% prompt (231.2 -> 244.5 tok/s) |
| `--kv-q8` (8-bit KV, not 4-bit) | lossless here: last-row cosine 0.998987, top-5 5/5, +0.37 GiB |
| DSpark depth 4 with `KEEP 64` | depth 5, depth 6, `KEEP 32` and `KEEP 128` all measured worse |
| `STRATA_SERVE_PASS_BIAS=0` | the upstream default (0.5/0.3) gives 2.2x the expert misses here: 12.1 against 25.1 tok/s |
| the v41 serve's thinking budget (`reasoning_budget_tokens 8192`, `reasoning_close_retry`) | a runaway reasoning loop is now closed at the budget and the model answers (tested with a 128-token budget) |
| the effort mapping aligned with the reference implementation | `low`/`medium`/`xhigh` resolve the way the reference server resolves them for V4.1 (75/75/75) |
| DSML tool-call parsing in the local path | tool calls reach the client, streaming and not |
| prompt batching | a fresh 63K prompt at `--prompt-batch 12288` can run out of memory and segfault the one-shot CLI; the server sizes the batch itself and does 363 tok/s |

## Where the time goes, and what is left

A runtime trace at 17K state: the GPU kernels are 32.2 ms of a 139 ms profiled token, and 215 `hipStreamSynchronize`
calls a token account for 105 ms - 95.3 of them after a device-to-host readback and 76.7 after the dense graph's
launch - with 4,238 kernel launches a token.  Decode is therefore **host/sync-bound, not GPU-bound**, and the
synchronizations are mostly real work being waited for (the experts' DMA and the launch latency), not idle overhead.
The lever that is left is the number of kernel launches.  The per-layer host handshake itself was investigated to the
end and closed: see [docs/ds4/V41_PIPELINE_OVERLAP.md](docs/ds4/V41_PIPELINE_OVERLAP.md).

- **Older graphics cards** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, built from source on Linux: [Intel Arc](docs/INTEL_ARC.md).
- **Apple Silicon Macs (M1 to M5)**, built from source by `./setup.sh` (the Metal engine): [Mac](docs/MAC.md).
- **AMD Ryzen AI Max (Strix Halo)**, built from source on Linux: [Strix Halo](docs/STRIX_HALO.md).
- **Older processors without AVX2**: they work, but slowly. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

With the numbers, so they are not repeated: `--pf-b` below 1.0 (-19% decode; the default 1.43 sits just above that
cliff), DSpark depth 5 and 6, `KEEP 32` and `128`, `STRATA_DSPARK_ADAPT`, the adaptive arena, `STRATA_PROM=1`, the
peer expert tier on the second card (correct and bit-identical rows, 91.5 rows a token moved off card 0, but 11.8
against 12.6 tok/s), the transcode decode, `STRATA_HIP_WMMA` (the +5.4% was noise: v41 never calls that kernel), a
cache for `ggml_cuda_get_device` (120 calls a token, 0.06 ms) and the 685 per-token error queries (below 0.3 ms).

## Run it

```sh
./run-v41-gsq.sh                 # the server on port 8150, with the configuration in local/ds41-local.json
./run-v41-gsq.sh --port 8151     # extra server arguments are passed straight through
```

It first checks that the model and the two Engram tables are in `models/` and says what is missing if they are not.
It picks the interpreter itself: `STRATA_PY` if set, then a `.venv` in the repository, then `venv/`, then `python3` -
the first one that imports **pillow, psutil, regex and jinja2**, which are the only packages the server needs
(`python3 -m venv .venv && .venv/bin/pip install pillow psutil regex jinja2`; `.venv/` is already gitignored).
Loading takes a couple of minutes (152 GiB are mapped, the RAM arena is filled, the Engram tables are opened); once
it is up, `curl -s http://127.0.0.1:8150/v1/models` answers `"status": {"value": "loaded"}`, and the OpenAI- and
Anthropic-compatible API is on that port.  For a single generation without the server, use `local/run-one-shot.sh`
(with `EXE=$PWD/build-hip-fused/ds41_generate` for the fused build the server uses).

## Getting the model files

The model is not in git (355 GiB in total).  [local/fetch-model.sh](local/fetch-model.sh) builds exactly the layout
this branch uses:

**Mac (Apple Silicon, M1 to M5, macOS 15 or newer):** install Apple's Command Line Tools once
(`xcode-select --install` in Terminal; Xcode itself is not needed). Then open Terminal in the Strata folder and run
**`./setup.sh`**. It builds the Mac engine on your Mac (about 7 minutes), asks the same questions, downloads the model
and starts it. Next time, `./setup.sh` starts it right away. Which size fits your Mac and how fast it is:
[docs/MAC.md](docs/MAC.md).

The steps are the same for NVIDIA and AMD. The installer finds your card and sets up the right engine for it. It
asks you a few questions:

Space: 163 GB for the GGUF, then the two Engram tables (101 GB each) plus the two 101 GB shards they are extracted
from.  `extract_engram.py` works on the whole set - it validates every shard the index names and needs a `--to` for
every Engram layer - so both shards must be on disk at once and both destinations are given in one call: there is no
one-table-at-a-time order (measured: "shard not found: model-00048-of-00048.safetensors" with only 47 present, and
"no --to for layers.14" with a single `--to`).  Peak for that phase is therefore 163 + 203 + 202 GB, which is why
`WORK` and `DEST_LAYERS1`/`DEST_LAYERS14` exist: put the shards on one disk and the tables on another, and the peak
split across the two is what the machine has to hold.

```sh
local/fetch-model.sh /path/to/models     # DEST defaults to <repo>/models
```

It is resumable, it uses `aria2c` when it is installed (8 connections per file: measured 270 MiB/s against 5.7 MB/s for a single `curl`), and it checks: every expected size comes from the Hugging Face API, a file counts as complete only
when its size matches (so a partial download is resumed with `curl -C -`, never trusted), the parts are joined by the
repo's own `reassemble.sh` (which verifies their sha256 first and the joined file at the end), and an interrupted join
is reported instead of being joined over.  Every parameter:

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again. It starts right away and downloads nothing twice. Close
its window to stop the model. `UPDATE.bat` (`./update.sh`) updates Strata without starting it. Updating, Docker,
several cards, where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

**Is it fast on your PC?** Run `BENCH.bat` (`./bench.sh`): about 5-10 minutes, it measures chat, 4K/16K prompts and an agent-style session with your own settings and writes a report to share ([how](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

## Which model should I pick?

The installer recommends one for your RAM. The same model comes in several sizes, compressed more or less. Smaller
sizes are faster. Larger sizes are a bit smarter.

| Your RAM | Take | Why |
| --- | --- | --- |
| first argument | `<repo>/models` | where the trees go |
| `GSQ_REPO`, `FP8_REPO` | the two repositories linked above | where to fetch from |
| `JOBS` | 1 | how many files are fetched at once (one by default) |
| `CONN` | 4 | connections per file with `aria2c`.  The Hugging Face CDN throttles each connection: measured on this repo, 1 gives 5.7 MB/s, 4 gives ~27 MB/s and 8 gives **140 MiB/s** |
| `ARIA2` | 1 when `aria2c` is installed | `0` forces `curl` |
| `DRY_RUN=1` | 0 | print the whole plan (what is complete, what would be fetched, the join, the Engram steps) and change nothing |
| `DELETE_PARTS` | 1 | drop each GGUF part as it is appended; `0` keeps them and needs the space twice |
| `DEST_LAYERS1`, `DEST_LAYERS14` | `<models>/engram/layers1`, `layers14` | put the two Engram tables on different disks |
| `KEEP_FP8` | 0 | keep the FP8 shards after the extraction (101 GB each) |
| `FORCE_JOIN` | 0 | remove an interrupted `V41-GSQ.gguf` and join again (by default it stops and says so) |
| `VERIFY_ROWS` | 200 | how many random Engram rows are checked against the source |
| `WORK` | `DEST` | scratch for the FP8 shards and the size lists |
| `HF_TOKEN` | `~/.cache/huggingface/token` | needed only for gated repositories |

- **[Coder](docs/MODELS.md#coder):** a coding version with half of the experts removed. It reaches 91% of the full
  model's SWE-bench Verified score (measured by its authors) and fits 32 GB of RAM. It is weaker outside code,
  including Chinese and other CJK text (#438). For those, take Q2_0, IQ2_XS or IQ3_S, which keep every expert.
- **[Swift 1.5](docs/MODELS.md#swift-15):** a fine-tune that thinks for a much shorter time before it answers. You
  get the answer sooner, at about the same quality.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** Unsloth's ~4-bit version, between IQ3_S and
  UD-Q4_K_XL in quality. A 94 GB download. With less than ~80 GB of RAM, Strata reads part of it from the SSD
  while it answers, so it is slower there (an NVMe SSD helps).
- **[Unsloth UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL and Q8_0](docs/MODELS.md#unsloths-files-ud-iq4_xs-to-q8_0):**
  4- to 8-bit, closer to the full model (93-95% the same next token as Q8_0). Fast where every expert fits in RAM
  (~88 GB for UD-Q4_K_XL up to ~136 GB for Q8_0); with less RAM Strata reads part of them from the SSD while it
  answers, which is much slower (7-8.5 tokens/s for UD-Q4_K_XL on a 64 GB PC).
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** you set it up by hand. It is
  not in the installer's menu.

Sizes, downloads and what fits where: [docs/MODELS.md](docs/MODELS.md). To add another model later, run
`SETUP.bat` (Linux: `./setup.sh --setup`).

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** open `http://127.0.0.1:8080`. It has **Chat**, a live **Monitor** of the model and your
  GPU/CPU/RAM, and **About** with the settings and addresses.
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8080/v1`**. Any API key and any model name work.
  - Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI and other apps that use the OpenAI Responses API: `/v1/responses`
    ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Thinking:** choose **off, low, medium or high** in the chat menu or in your app's "reasoning effort". Off is the
  fastest. High is best for hard questions.
- **Pictures:** say yes to "Images?" in setup. Then click **Picture** in the chat, or attach pictures in your app.
  AMD cards read pictures on Linux through the processor; on Windows they can't yet.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **One request at a time:** by default Strata answers one request, and the others wait. To answer several at once,
  set `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). On a 12 GB card this makes each answer slower.
- **Long prompts:** Strata reads the first message of a chat in full, about 1 minute per 30,000 tokens. Follow-up
  messages start in seconds.

More: [where your chats are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

### Optional: more speed on one card when the whole model fits in RAM

Two opt-in settings (by RubenTorney, #1904 and #1905). They help when you run one graphics card and your RAM holds
every expert (Strata's arena mode, for example Q2_0 or IQ3_XXS on a 64 GB PC). Open your config,
`strata-<model>.json` in the Strata data folder ([where](docs/INSTALL.md#where-things-are-stored)), and add:

- **Faster decode:** `"--adapt-async", "1"` to the `"args"` list. The expert cache then updates on a helper thread
  instead of pausing the answer. Measured on an RTX 5070 (Windows, Q2_0): +4-5% decode, +7-10% on code and long
  documents. Answers can differ slightly from the default (they stay the same from run to run).
- **Faster prompt reading on Windows:** `"STRATA_PIN_BY_NEED": "1"` in the `"env"` block. Windows can lock only about
  half the RAM for the graphics card; this locks the experts the card actually needs first. Measured on the RTX 5070:
  +4.4% prompt reading.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Restart Strata (`START-HERE.bat` or `./setup.sh`) after the change. To undo, remove the two entries.

## Something went wrong?

- **My PC froze the first time Strata started.** This is normal while it loads the model. Wait, and don't close the
  window. Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Run `START-HERE.bat` (or `./setup.sh`) again. It continues where
  it stopped.
- **It's very slow and the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does
  not have enough free RAM. Close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It says port 8080 is already in use.** Strata is already running. Look for its window.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/Niko1221/Strata/issues) and attach `strata-<model>.log` from the Strata folder. Found a
security problem? Report it privately: [SECURITY.md](SECURITY.md).

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
you use all the time stay on the counter, and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts").** Each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are used most often. **Your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words. The big model checks them all at once. You get
  the same answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens at a time), at over 1,000 tokens per second.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was
compressed by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE). A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Support Strata

Strata is free and open source. If it is useful to you, you can support its development:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>

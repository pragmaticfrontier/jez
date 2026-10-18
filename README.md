# Jez

An open source non-generative AI decision model that returns typed answers with calibrated probabilities.

Jez is a pair of decision models that answer [Jev's](https://docs.typesafe.ai/api) HTTP API field for
field: **Jez-small** (from `Qwen/Qwen3.5-0.8B`) and **Jez-large** (from `google/gemma-4-E4B-it`). A client
written against TypeSafe's SDK moves to Jez by changing two environment variables, `TYPESAFE_BASE_URL`
and `TYPESAFE_DEFAULT_MODEL`. Both models read the probabilities of the declared answer labels and never
generate text, so an answer outside the declared set cannot occur.

```
state + typed questions  ──►  prefix once, one suffix per question and order  ──►  label log-probabilities  ──►  Jev answers
```

The repository also keeps the earlier `sysone/` prototype (its own `/v1/decide` API) that Jez grew out of.

## Quick start

```bash
uv venv --python 3.12 .venv
# aarch64 + CUDA 13 (DGX Spark): torch comes from the PyTorch index
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "torch==2.10.0+cu130"
uv pip install --python .venv/bin/python -e ".[dev]"

JEZ_API_KEYS=my-token JEZ_MODELS=jez-small-1.0.0 .venv/bin/jez-serve        # http://127.0.0.1:8009
```

Then, with TypeSafe's own SDK:

```bash
export TYPESAFE_API_KEY=my-token TYPESAFE_BASE_URL=http://127.0.0.1:8009 TYPESAFE_DEFAULT_MODEL=jez-small-latest
```

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient() as client:
    res = client.system_one(
        state={"message": "I was charged twice. Please help."},
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
            "urgency": Score(instructions="How urgent?", criteria=["Can wait", "This week", "Today"]),
        },
    )
    print(res.model, res.nouls["billing"].noul, res.choices["tone"].choice, res.scores["urgency"].score)
```

Or raw HTTP: `examples/claim.json` is the reference request (a claim record and one question of each type).

```bash
curl -s localhost:8009/v1/systemone -H "authorization: Bearer my-token" -H "content-type: application/json" \
     -d @examples/claim.json
```

### Server environment

| variable | default | |
|---|---|---|
| `JEZ_MODELS` | `jez-small-1.0.0` | comma list of `id` or `id=checkpoint`; a checkpoint puts a stand-in behind an id |
| `JEZ_BACKEND` | `auto` | `torch`, `mlx`, `fake` |
| `JEZ_API_KEYS` | | accepted bearer tokens; `--no-auth` / `JEZ_NO_AUTH=1` skips the check for local work |
| `JEZ_MAX_QUEUE` | `32` | requests in flight before 429 |
| `JEZ_CUDA_MEMORY_FRACTION` | `0.6` | PyTorch allocator cap; over-allocation becomes a 529, never a hung machine |
| `JEZ_CALIBRATION` | `releases/{model}/calibration.json` | fitted temperatures per model |
| `JEZ_HOST`, `JEZ_PORT` | `127.0.0.1`, `8009` | |

## The contract

`POST /v1/systemone`, `GET /v1/models` and `GET /health`, with Jev's request rules and status codes:

| rule | status |
|---|---|
| body over 32 MiB, compressed or not | 413 |
| not UTF-8, a byte order mark, non-finite numbers, more than 64 nested levels | 400 |
| 64k tokens per request, 32k per question, counted with the serving tokenizer | 400 `max_tokens_exceeded` |
| `Content-Encoding` other than one layer of gzip, x-gzip or zstd (window at most 8 MiB) | 415; corrupt body 400 |
| missing field, malformed question, unknown model, image asset, `box` question | 422, naming the field |
| missing or wrong token | 401 |
| queue deeper than `JEZ_MAX_QUEUE` | 429 with `retry-after` |
| model not loaded, out of memory | 529 |

Model ids are `jez-small-1.0.0` and `jez-large-1.0.0`; aliases `jez-small-latest`, `jez-large-latest` and
`jez-latest` (the large model). Jev's own names (`jev-latest`, `jev-1.13.0`, ...) get a 422 so a misconfigured
client fails loudly. Answers follow TypeSafe's formulas (`jez/confidence.py`): choice confidence
`(p_max - 1/n) / (1 - 1/n)`, score `Σ i·p_i` with confidence `1 - spread / even_spread`, noul `P(true)`.
Errors are `{"error": {"type", "message", "field"}}`. Latency and cost go in the `Jez-Latency-Ms` and
`Jez-Cost-Usd` response headers, never in the body.

The conformance suite (`tests/conformance/`) covers every request example in TypeSafe's API reference, each
answer type with plain and structured criteria, choice at 2, 26, 27 and 255 options, score at 2 and 10 levels,
every rule above at its limit and one step past it, both compression formats, and the Python SDK
(`typesafe-sdk` 0.7.5) and JavaScript SDK (`@typesafe-ai/sdk` 0.7.0) against a live server, including their
exception classes for 400, 401, 413, 415, 422, 429 and 529.

## Engine

1. The model scores label tokens at the answer position and never samples. Identical requests return identical bytes.
2. Only the text decoder loads: Gemma's `model.language_model.*` weights are mapped onto `Gemma4ForCausalLM`,
   the vision and audio encoders are never materialised.
3. The output head multiplies the final hidden state by the label rows of the tied embedding only
   (Gemma's soft-cap of 30 applied to those rows), instead of 248,320 or 262,144 rows.
4. Choice labels are A..Z then AA, AB, ... up to 255; score levels are the digits 0..9; noul is Yes/No. Both
   tokenizers supply more than 255 single-token labels (`tests/jez/test_labels.py`); a teacher-forced fallback
   exists for tokenizers that do not.
5. Every question is scored in its given order and in reverse, and the two label log-probability sets are averaged.
6. The state (system prompt plus state block) runs once; the question suffixes run batched against the replicated
   cache, including Gemma 4's sliding-window layers and its 18 shared-KV layers. In fp32 the shared path matches a
   single unpadded forward pass to 2e-4 on both bases; in bf16 the same comparison shows accumulation noise of
   0.1 to 0.2 (Qwen) and more on 30-option questions (Gemma), so backend-agreement checks run in fp32
   (`eval/backend_agreement.py`).
7. Temperatures are fitted per question type and option-count bucket (`jez/calibration.py`) and never change the winner.

Backends: `jez/engine/torch_backend.py` (CUDA, tested) and `jez/engine/mlx_backend.py` (Apple Silicon, written
against mlx-lm and not yet run; it cannot import on the Linux development machine).

## Measured so far (DGX Spark, GB10, bf16)

| | Jez-small base (Qwen3.5-0.8B) | Jez-large base (Gemma 4 E4B) |
|---|---|---|
| reference request `examples/claim.json`, p50 / p99 | 60 / 63 ms | 201 / 209 ms |
| weights resident | 1.5 GB | 15.0 GB |
| GPU power, inference / training | 49 W / 58 W peak | not measured (see below) |
| Tier 1 zero-shot, 1,000-item samples | 57.6% mean (`runs/qwen3.5-0.8b-zeroshot-n1000.md`) | run twice, both times the machine froze |

Qwen3.5-0.8B zero-shot, 1,000 items per task: SST-2 83.1, IMDB 88.7, HellaSwag 44.6, ARC-Easy 80.7, ARC-Challenge 61.2,
BoolQ 74.5, Belebele 49.5 (12-language mean), AG News 76.0, Emotion 55.5, BANKING77 15.0, ToxicChat F1 0.417,
prompt injections 53.4, SST-5 24.3. Pooled choice ECE 0.078 raw and 0.050 after temperature fitting; order stability on
AG News 88% (gate: 95%). `eval/compare.py` sets any run beside Jev, Microsoft-Decision-1, the Decisions API, Laya and the
targets from `research/decision-model-benchmarks.csv`.

### The machine powers itself off under sustained heavy GPU load

Twice on 2026-10-16 the DGX Spark went down silently within two minutes of Gemma 4 E4B starting sustained forward
passes: the journal stops mid-stream with no Xid, OOM or panic, and the machine stays off until power-cycled. Memory
was not the cause (92 GB free, 16 GB model). The mechanism showed itself later the same day during the Jez-small
fine-tune: at a steady 59 W the GPU heat-soaked to 90°C within ten minutes, `nvidia-smi` reported a 2°C margin to
the slowdown point and 7°C to the shutdown point, and the SoC thermal zones read 97°C, the band the community
reports for this GB10 defect. The documented workaround is a clock cap, `sudo nvidia-smi -pm 1 && sudo nvidia-smi
-lgc 300,2200` (`tools/gb10-clock-cap.service` makes it persistent), which needs root. Until then
`tools/gpu_guard.py` governs from user space: it samples the GPU and the SoC every two seconds and pauses the
job's process group (SIGSTOP) at GPU 82°C, SoC 88°C or an 8°C margin to slowdown, resuming (SIGCONT) once every
reading is 6°C cooler; the die cools in seconds but the heat-soaked chassis does not, so under a steady 60 W load
the job runs at roughly 50 to 65% duty and readings overshoot the limits by up to 3°C with half-second sampling. Paused time
does not count against the job's timeout, a pause longer than ten minutes kills the job (exit 125), and the
fsync'd telemetry in `runs/telemetry/` (power, temperatures, clocks, MemAvailable, paused flag) lets the moment
of any future outage be read back. Jez-large work stays parked until the clock cap is in place.

## Fast kernels

Qwen3.5's Gated DeltaNet layers run through `flash-linear-attention` (Triton). Triton compiles a small C helper
against the running Python's headers; this machine's system Python has no `python3-dev`, so the matching headers
were unpacked from `libpython3.12-dev` into `.venv/include/python3.12`, and `tools/gpu_guard.py` adds that
directory to `CPATH`. Without the kernels the same request takes 467 ms instead of 60.

## Running anything on the GPU

Every GPU job goes through `tools/gpu_guard.py`:

```bash
tools/gpu_guard.py --timeout 3600 -- .venv/bin/python -m eval.run_eval --checkpoint google/gemma-4-E4B-it --label gemma-zeroshot --tasks tier1 --n 1000
```

On a DGX Spark the GPU shares the 128 GB of system memory, and CUDA allocations are not charged to the process
cgroup, so an over-allocating job is not killed; the machine hangs. The guard closes that hole three ways: a cgroup
cap on CPU-side memory (`systemd-run --scope`), PyTorch's allocator cap on the GPU (`JEZ_CUDA_MEMORY_FRACTION`), and a
watchdog that kills the process group when `MemAvailable` drops below a floor. Keep one model resident per process
and call `backend.release()` before loading another.

## Evaluation

```bash
tools/gpu_guard.py -- .venv/bin/python -m eval.run_eval --checkpoint Qwen/Qwen3.5-0.8B --label qwen-zeroshot --tasks tier1 --n 1000
.venv/bin/python -m eval.compare runs/qwen-zeroshot.json
tools/gpu_guard.py -- .venv/bin/python -m eval.order_stability --checkpoint ... --task withdrawal_routing
```

`eval/tasks.py` maps the thirteen Tier 1 datasets onto Jev questions (Belebele as twelve language tasks), and reads
the Tier 2 sets (typed-decisions, Decision Bench, the Laya battery, withdrawal routing) from `eval/data/tier2/`,
which the repository does not ship; see the README there. Every run reports accuracy (or F1 for ToxicChat), ECE raw
and after temperature fitting on the other half of the sample, Brier, and p50/p99 latency.

## Training

```bash
.venv/bin/python -m train.build_mix --total 300000 --out train/data/mix-v2            # manifest with licences, overlap drops, yes/no balance
tools/gpu_guard.py --cuda-fraction 0.7 -- .venv/bin/python -m train.teacher_label --teacher <27B teacher> --data train/data/mix-v2/train.jsonl --licence-checked
tools/gpu_guard.py -- .venv/bin/python -m train.finetune --model jez-small-1.0.0 --data train/data/mix-v2 --sweep 1e-5,2e-5,4e-5
tools/gpu_guard.py -- .venv/bin/python -m train.finetune --model jez-small-1.0.0 --data train/data/mix-v2 --lr 1e-5 --out releases/jez-small-1.0.0
```

The mix builder balances every yes/no question family (one source, one instruction text, at least 200 examples) to
about as many "true" as "false" answers. mix-v1 did not: its Jigsaw toxicity questions were 94% "false" and the NLI
entailment question 2:1 "false", and the Jez-small trained on it answered "false" to 98% of ToxicChat and every
prompt-injection item with 0.98 confidence (ToxicChat F1 0.42 → 0.18, BoolQ −8 points). mix-v2 is the same build with
the balance; `--no-balance-noul` reproduces the old behaviour and the manifest records the caps and drops.

The objective is cross-entropy on the gold label plus 0.5 × KL from the teacher's label distribution when an example
carries teacher probabilities, read at the answer position through the same label-only head as serving; option order
and label letters are reshuffled per example. Jez-small is a full fine-tune (fp32 master weights, bf16 autocast),
Jez-large trains LoRA (rank 64, alpha 128) with embeddings frozen. Each 64-sequence optimizer step is taken in
forward passes bounded by both a sequence count and a padded-token budget (64 sequences, 32,768 tokens), so a batch
of 4,096-token states runs as several passes and a batch of short utterances as one wide pass, and activations are
recomputed in the backward pass: on Qwen3.5-0.8B they cost about 2.9 MB per token (51 GiB for a 16k-token pass),
checkpointing brings that to 7 GiB for about 14% more compute. A 27B teacher is a heavier sustained load than
Gemma on this machine, so the first Jez-small run trains on gold labels only; teacher labels can be added later. Sources
whose licence forbids commercial use (XNLI, Yelp Polarity, SciQ, RACE) stay out unless `--allow-noncommercial` is
passed; `train/sources.py` records every licence, and the manifest records the 13-token overlap drops per source.
The builder reads each source from the Hub's parquet conversion (`train/parquet.py`: one download per shard, cached
under `train/data/parquet/`, rows spread evenly over the split and read one row group at a time), which takes seconds
per source where the paged rows API took five minutes; the rows API remains the fallback for datasets without a
parquet conversion. `train/pipeline_small.sh` runs the whole Jez-small release (sweep, full run, temperatures, Tier 1,
order stability, latency, model card) with every GPU step under the guard, skipping steps whose output exists. A
release run writes its training state (fp32 trainable weights, optimizer, step, best-so-far) to `<release>/state.pt`
every 200 steps and resumes from it when restarted, so a power-off costs at most 200 steps; the file goes when the run
completes. The serving weights are saved as a bf16 copy so the fp32 master weights keep accumulating small updates.
`train/pipeline_large.sh` is the Gemma counterpart: it adds the Gemma 4 E4B zero-shot baseline that gate 2 compares
against and `train/merge_adapter.py`, which folds the LoRA adapter into the base weights so the engine serves a plain
checkpoint. Both scripts take `GUARD`, `PY` and `SKIP_LATENCY` from the environment, which is how `train/modal_app.py`
runs them unchanged on a rented GPU: Gemma and a 27B teacher are beyond this machine's cooling, and an H100 does the
Jez-small run in about an hour and the Jez-large run in three to four. The latency gate stays on the target machines;
the Modal runs skip it and the model card picks the figure up locally. Setup and the job list are in that file's docstring.

## Layout

```
jez/            schema, prompt planner, labels, confidence, calibration, engine backends, decider, server
eval/           datasets-server loader, Tier 1 tasks, run_eval, compare, order_stability, backend_agreement
train/          sources with licences, parquet reader, build_mix, teacher_label, finetune, merge_adapter, fit_calibration, model_card,
                pipeline_small.sh and pipeline_large.sh, modal_app.py (the same pipelines on a rented GPU)
tools/          gpu_guard (memory limits, thermal governor, telemetry), no_guard (its stand-in on cloud GPUs), engine_check,
                latency, gb10-clock-cap.service
tests/          conformance suite, SDK tests (Python, JavaScript), unit tests, GPU tests
research/       published figures (CSV) and their provenance
examples/       claim.json (reference request), sysone-era scripts
sysone/, bench/ the earlier prototype, unchanged
```

## Status against the milestones

| milestone | state |
|---|---|
| M0 Jev schema, server, conformance suite, SDK tests | done; gate 1 passes with a stand-in behind every id |
| M1 both bases on torch, 255-label tables, Gemma cache sharing, latency | done on CUDA; MLX backend written, untested |
| M2 the thirteen Tier 1 loaders, compare columns | loaders and compare done; Tier 2 sets need the data files |
| M3 manifest, overlap filter, teacher labels | scripts done; the 300k mix and teacher run have not been executed |
| M4 fine-tunes | script done, not run |
| M5 full-split runs, model cards, tagged weights | not started |

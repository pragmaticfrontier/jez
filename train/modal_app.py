"""Run the Jez pipelines on a rented GPU through Modal (https://modal.com), for the jobs this machine cannot sustain.

Setup, once:

    uv pip install --python .venv/bin/python -e ".[cloud]"    # the modal client
    modal setup                                               # browser login
    modal secret create huggingface HF_TOKEN=hf_...           # an account that has accepted the Gemma licence
    modal volume put jez-data train/data/mix-v2 /mix-v2       # the exact mix this machine built (268 MB); JEZ_MIX picks it

Jobs, each `modal run --detach train/modal_app.py --step <name> --no-wait`; the card is JEZ_MODAL_GPU (default H100, which needs
a payment method on the account; `JEZ_MODAL_GPU=A10G` runs on the free credit and fits the small model with
`--micro-tokens 16384`):

    small    the Jez-small pipeline on the volume's mix; release under /releases/jez-small-1.0.0
    large    the Jez-large pipeline: Gemma zero-shot baseline, sweep, LoRA run, merge, temperatures, Tier 1, stability
    tests    the CPU test suite inside the image (no GPU)
    cmd      any command in the repository, e.g. --cmd "python3 -m eval.run_eval --checkpoint ... --full"

JEZ_MIX names the mix directory under the jez-data volume (default mix-v2) and JEZ_RUN_ROOT the directory on the
jez-releases volume that holds this round's releases and its runs/ (default /releases/mixv2; the mix-v1 round wrote to
/releases itself). Files a pipeline expects to find under runs/ (a baseline Tier 1 run, a sweep.json to reuse a learning
rate) are copied into <root>/runs with `modal volume cp jez-releases <from> <to>` before launching.

Results stay on the jez-releases volume: `modal volume get jez-releases /jez-large-1.0.0 releases/` brings a release
home and `modal volume get jez-releases /runs runs/` the evaluation JSON. The latency gate is measured on the target
machines, not here, so the pipelines run with SKIP_LATENCY=1 and the model card gets its latency figure locally. A
preempted run is simply started again: the pipelines skip finished steps and the trainer resumes from state.pt.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
GPU = os.environ.get("JEZ_MODAL_GPU", "H100")  # H100 needs a payment method on the Modal account; A10G and L4 run on the free credit
MIX = os.environ.get("JEZ_MIX", "mix-v2")  # the mix directory under the jez-data volume
# Where this round's outputs live on the jez-releases volume: <ROOT>/jez-*-1.0.0 and <ROOT>/runs (the repository's
# runs/ directory is a symlink to it inside the container). The mix-v1 round wrote to the volume root; every later
# round gets its own root so that nothing is deleted or overwritten on the volume.
ROOT = os.environ.get("JEZ_RUN_ROOT", "/releases/mixv2")
STEPS = {
    "small": f"train/pipeline_small.sh /data/{MIX} {ROOT}/jez-small-1.0.0",
    "large": f"train/pipeline_large.sh /data/{MIX} {ROOT}/jez-large-1.0.0",
    "tests": "python3 -m pytest tests -q --ignore=tests/jez/test_engine_gpu.py",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_pyproject(str(REPO / "pyproject.toml"), optional_dependencies=["train", "dev"])  # dev: pytest and the SDK for the tests step
    .pip_install("transformers==5.19.0", "flash-linear-attention==0.5.2")  # the versions the engine was built against;
    # without fla, Qwen3.5's linear-attention layers fall back to a torch loop and an H100 trains at a third of its pace
    .env({"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "PY": "python3", "GUARD": "python3 tools/no_guard.py", "SKIP_LATENCY": "1"})
    .add_local_dir(str(REPO), remote_path="/root/jez",
                   ignore=[".venv", ".git", "train/data", "runs", "releases", ".pytest_cache", "**/__pycache__",
                           "**/node_modules", "*.egg-info"])
    # releases/ is ignored above (weights live on the volume), but train.model_card reads its template from there
    .add_local_file(str(REPO / "releases" / "MODEL_CARD_TEMPLATE.md"), remote_path="/root/jez/releases/MODEL_CARD_TEMPLATE.md")
)
data = modal.Volume.from_name("jez-data", create_if_missing=True)
releases = modal.Volume.from_name("jez-releases", create_if_missing=True)
hf_cache = modal.Volume.from_name("jez-hf-cache", create_if_missing=True)
VOLUMES = {"/data": data, "/releases": releases, "/hf": hf_cache}
app = modal.App("jez")


def _run(cmd: str, root: str = "/releases") -> int:
    import os
    import subprocess

    os.chdir("/root/jez")
    runs = f"{root}/runs"
    os.makedirs(runs, exist_ok=True)
    if os.path.lexists("runs"):
        os.remove("runs")
    os.symlink(runs, "runs")  # the pipelines write evaluation results under runs/
    rc = subprocess.call(["bash", "-c", cmd])
    for vol in VOLUMES.values():
        vol.commit()
    return rc


if GPU != "none":  # JEZ_MODAL_GPU=none leaves the GPU function out, so the CPU steps work on an account without a payment method

    @app.function(image=image, gpu=GPU, timeout=24 * 3600, volumes=VOLUMES, secrets=[modal.Secret.from_name("huggingface")])
    def run_gpu(cmd: str, root: str = "/releases") -> int:
        return _run(cmd, root)

else:
    run_gpu = None


@app.function(image=image, timeout=4 * 3600, volumes=VOLUMES, secrets=[modal.Secret.from_name("huggingface")])
def run_cpu(cmd: str, root: str = "/releases") -> int:  # tests, and CPU-only jobs such as uploading a release (--gpu none)
    return _run(cmd, root)


@app.local_entrypoint()
def main(step: str = "", cmd: str = "", gpu: str = GPU, wait: bool = True) -> None:
    command = STEPS.get(step) or cmd
    if not command:
        raise SystemExit(__doc__)
    fn = run_cpu if step == "tests" or gpu == "none" else run_gpu
    if fn is None:
        raise SystemExit("JEZ_MODAL_GPU=none: only the tests step and --gpu none commands run without a GPU")
    if fn is run_gpu and gpu != GPU:
        fn = run_gpu.with_options(gpu=gpu)
    print(f"modal: {command}  (runs/ -> {ROOT}/runs)")
    if not wait:  # with `modal run --detach ... --no-wait` the job keeps running after this client exits
        call = fn.spawn(command, ROOT)
        print(f"spawned {call.object_id}; follow with: modal app logs <app id printed above>")
        return
    rc = fn.remote(command, ROOT)
    raise SystemExit(rc)

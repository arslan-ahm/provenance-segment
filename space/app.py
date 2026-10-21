"""Hugging Face Space entry point (Gradio SDK on ZeroGPU hardware).

The NLI verifier is placed on CUDA at import time, as ZeroGPU requires, and only the
verifier call runs inside ``@spaces.GPU``; everything else runs on the Space's CPU. Off
ZeroGPU (e.g. locally) the decorator does nothing and the model runs on CPU.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import spaces  # noqa: E402  (must be imported before torch on ZeroGPU)
import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

from provseg.app.gradio_app import build_demo  # noqa: E402
from provseg.app.live import ClaimChecker, with_cpu_fallback  # noqa: E402

RUNS = Path(os.environ.get("PROVSEG_RUNS_DIR") or (ROOT / "runs" if (ROOT / "runs").exists() else ROOT.parent / "runs"))  # Space layout, else the repo
RUN = os.environ.get("PROVSEG_RUN", "main_v1")
v = json.loads((RUNS / RUN / "manifest.json").read_text(encoding="utf-8"))["config"]["verifier"]

ON_GPU = torch.cuda.is_available()  # True on ZeroGPU (CUDA is emulated outside @spaces.GPU)
settings = (v["model"], v["entail_threshold"], v["contradict_threshold"], tuple(v["window_sentences"]))
cpu_checker = ClaimChecker(*settings, encoder=CrossEncoder(v["model"], device="cpu"))
gpu_checker = ClaimChecker(*settings, encoder=CrossEncoder(v["model"], device="cuda")) if ON_GPU else None


@spaces.GPU(duration=30)
def _nli_gpu(claim, chunks, handles):
    return gpu_checker.check(claim, chunks, handles, nli=True)


def check(claim, chunks, handles, nli):
    anchored = cpu_checker.check(claim, chunks, handles, nli=False)  # validates input before asking for a GPU
    if not nli:
        return anchored
    if gpu_checker is None:
        return cpu_checker.check(claim, chunks, handles, nli=True)
    return with_cpu_fallback(lambda: _nli_gpu(claim, chunks, handles), lambda: cpu_checker.check(claim, chunks, handles, nli=True))


sanity = [(f"{label}, validation sample", RUNS / "verifier_dev" / "diagnostics" / f) for label, f in (("3-sentence windows", "verifier_w3.json"), ("1- and 3-sentence windows (adopted)", "verifier_w1_3.json"))]
demo, launch_kwargs = build_demo(RUNS, RUN, check, extra_sanity=sanity)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(**launch_kwargs)

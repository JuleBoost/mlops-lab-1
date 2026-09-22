
## Windows + uv: torch installs the CPU build by default

**Symptom:** `uv add torch torchvision` succeeded, but `torch.cuda.is_available()` was False
and the wheels were `2.14.0+cpu`, despite an NVIDIA RTX 5060 being present.

**Cause:** on Windows, PyPI's default `torch` wheel is CPU-only. CUDA wheels live only on
PyTorch's own index. (Lab notes that say "default install is CUDA" assume Linux.)

**Fix:** add to `pyproject.toml`, then re-run `uv add torch torchvision`:

    [[tool.uv.index]]
    name = "pytorch-cu130"
    url = "https://download.pytorch.org/whl/cu130"
    explicit = true

    [tool.uv.sources]
    torch = { index = "pytorch-cu130" }
    torchvision = { index = "pytorch-cu130" }

**What not to try:** cu128 — it tops out at torch 2.11.0, so it fails against a `>=2.14.0`
pin with "No solution found". Check which torch versions an index actually carries before
picking it. This block is per-project; copy it into each new project or you get CPU again.

## mlflow on Windows: two traps in train.py

**Symptom 1:** training finished and metrics were logged, but the script died with
`UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f3c3'` and the run was left
stuck in RUNNING.
**Cause:** mlflow prints emoji in its "View run ..." links; Windows consoles default to cp1252.
**Fix:** `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at the top of the script.

**Symptom 2:** `mlflow.pytorch.log_model(model, name="model")` failed with "input_example is
required", then with "the input signature must be specified using TensorSpec".
**Cause:** this mlflow defaults to `serialization_format="pt2"`, a traced-graph format.
**Fix:** pass `serialization_format="pickle"`.
**What not to try:** supplying `input_example` alone — it clears the first error and then hits
the TensorSpec one.

**Note:** logged models live at `mlruns/<exp_id>/models/m-<id>/artifacts/`, not inside the
run's own folder, so `mlruns/1/<run_id>/` does not exist in this version.

## Lab 3 serving gotchas (all hit 2026-09-22, fixed)

**Symptom:** `mlflow.pyfunc.load_model("models:/food11@champion")` then `model.eval()`
failed with `'_PyTorchWrapper' object has no attribute 'eval'`.
**Cause:** `_model_impl` is the pyfunc wrapper, not the nn.Module; in mlflow 3.16 the
torch module lives at `wrapper.pytorch_model` (older: `wrapper.model`).
**Fix:** unwrap with getattr fallbacks (`pytorch_model` → `model` → `get_raw_model()`).
**What not to try:** assuming `_model_impl` is directly the ResNet.

**Symptom:** container died with `Attempting to deserialize object on a CUDA device
but torch.cuda.is_available() is False`.
**Cause:** model trained on GPU (CUDA tensors pickled); CPU-only container can't restore them,
and pyfunc load exposes no `map_location`.
**Fix:** default `torch.load` to CPU before loading (`kwargs.setdefault("map_location",
torch.device("cpu"))`) in serve.py.

**Symptom:** API container → mlflow failed with `403 Invalid Host header - possible DNS
rebinding attack`.
**Cause:** mlflow 3.x validates the Host header; `host.docker.internal` isn't allowlisted.
**Fix:** start the tracking server with `--allowed-hosts "*"`.

**Symptom:** `No such artifact: ''` downloading `models:/food11@champion` inside a Linux
container while mlflow runs on Windows.
**Cause:** file-store artifact locations are absolute host paths (`file:///C:/...`);
the client resolves them locally, and the Linux container has no `C:/`.
**Fix (test-only):** bind-mount host `mlruns` at the same Windows path inside the
container. Real fix is a shared/object artifact store. Don't re-register — registry is fine.

## Docker build notes

- `uv sync --frozen --no-dev` in the builder fails with `Expected a Python module at:
src/mlops_lab_1/__init__.py` because only pyproject/uv.lock were copied. Fix:
`uv sync --frozen --no-dev --no-install-project` (project install needs src/).
- Windows `Start-Process`/`Start-Job` background servers are killed when the tool call
ends. For servers, run the command in your own terminal; `docker run -d` (daemon-side)
survives fine.

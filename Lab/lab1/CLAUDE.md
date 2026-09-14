
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

"""Serve the food11 champion model via FastAPI.

Usage:
    uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000
"""

import io
import os

import mlflow
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = "models:/food11@champion"

# Must match ImageFolder alphabetical order (see train.py build_loaders)
CLASS_NAMES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

mlflow.set_tracking_uri(TRACKING_URI)

# Model was trained on CUDA; CPU-only hosts (containers) must remap storages.
# mlflow's pyfunc load doesn't expose map_location, so default torch.load to CPU.
_orig_torch_load = torch.load


def _cpu_torch_load(*args, **kwargs):
    kwargs.setdefault("map_location", torch.device("cpu"))
    return _orig_torch_load(*args, **kwargs)


torch.load = _cpu_torch_load

_pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)

# Unwrap the torch module behind the pyfunc flavor (single load, no .pth path).
_impl = getattr(_pyfunc_model, "_model_impl", _pyfunc_model)
model = getattr(_impl, "pytorch_model", getattr(_impl, "model", _impl))
if not hasattr(model, "eval") and hasattr(model, "get_raw_model"):
    model = model.get_raw_model()
DEVICE = torch.device("cpu")  # trained on CUDA, served on CPU
model.to(DEVICE).eval()

_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

app = FastAPI(title="food11 classifier")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
    return {"category": CLASS_NAMES[idx.item()], "confidence": conf.item()}

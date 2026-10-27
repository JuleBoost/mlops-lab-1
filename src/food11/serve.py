"""Serve the food11 champion model via FastAPI.

Usage:
    uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000
"""

import io
import os
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

import mlflow

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

_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

app = FastAPI(title="food11 classifier")

# Loaded at startup, not at import, so tests can import this module without MLflow.
model = None


@app.on_event("startup")
def load_model():
    global model
    if os.environ.get("SKIP_MODEL"):
        return
    model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]):
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    batch = _transform(image).unsqueeze(0).numpy()
    # the pyfunc returns the network's raw outputs; softmax turns them into probabilities
    logits = torch.as_tensor(np.asarray(model.predict(batch), dtype=np.float32))
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    return {"category": CLASS_NAMES[idx.item()], "confidence": conf.item()}

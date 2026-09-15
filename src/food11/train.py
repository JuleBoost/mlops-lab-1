"""Train a resnet18 on Food-11 and track the run in mlflow.

Usage:
    uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
"""

import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "mini": REPO_ROOT / "data" / "food11_processed_mini",
    "processed": REPO_ROOT / "data" / "food11_processed",
}

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "food11"
NUM_CLASSES = 11

# mlflow prints emoji in its run links; Windows consoles default to cp1252 and crash on them
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# resnet18 was pretrained on 224x224 ImageNet images, normalised with these values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description="Train resnet18 on Food-11")
    p.add_argument("--dataset", choices=DATASETS, default="mini")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def build_loaders(root: Path, batch_size: int):
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    loaders = {}
    for split, folder in (("train", "training"), ("val", "validation"), ("test", "evaluation")):
        ds = datasets.ImageFolder(root / folder, transform=tf)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=split == "train")
    return loaders


def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over loader. Trains when an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += labels.size(0)

    return total_loss / seen, correct / seen


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_loaders(DATASETS[args.dataset], args.batch_size)
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "model": "resnet18",
                "optimizer": "adam",
                "device": device.type,
            }
        )

        for epoch in range(args.epochs):
            train_loss, train_acc = run_epoch(
                model, loaders["train"], criterion, device, optimizer
            )
            val_loss, val_accuracy = run_epoch(model, loaders["val"], criterion, device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(
                f"epoch {epoch + 1}/{args.epochs}  "
                f"train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
                f"val_loss {val_loss:.4f}  val_accuracy {val_accuracy:.4f}"
            )

        test_loss, test_accuracy = run_epoch(model, loaders["test"], criterion, device)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        # pickle stores the model object as-is; the default pt2 format traces the graph
        # and needs a TensorSpec signature we do not need here
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")

        print(f"test_loss {test_loss:.4f}  test_accuracy {test_accuracy:.4f}")


if __name__ == "__main__":
    main()

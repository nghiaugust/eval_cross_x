from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import convnext_tiny, resnet18, resnet50


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_MODELS = {"resnet18", "resnet50", "convnext_tiny"}


class LetterboxResize:
    def __init__(self, size: Iterable[int], fill: int | tuple[int, int, int] = 255) -> None:
        height, width = list(size)
        self.height = int(height)
        self.width = int(width)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        src_w, src_h = image.size
        scale = min(self.width / src_w, self.height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        canvas = Image.new("RGB", (self.width, self.height), color=self.fill)
        left = (self.width - new_w) // 2
        top = (self.height - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas


class CNNFeatureExtractor(nn.Module):
    def __init__(self, trained_model: nn.Module, model_name: str) -> None:
        super().__init__()
        self.model_name = normalize_model_name(model_name)
        if self.model_name in {"resnet18", "resnet50"}:
            self.features = nn.Sequential(*list(trained_model.children())[:-1])
        elif self.model_name == "convnext_tiny":
            self.features = trained_model.features
            self.avgpool = trained_model.avgpool
            self.pre_classifier = nn.Sequential(*list(trained_model.classifier.children())[:-1])
        else:
            raise ValueError(f"Unsupported feature extractor: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name == "convnext_tiny":
            x = self.features(x)
            x = self.avgpool(x)
            return self.pre_classifier(x)
        x = self.features(x)
        return torch.flatten(x, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN or CNN+SVM inference.")
    parser.add_argument("--config", default=None, help="Deploy config. Default: current folder/config.yaml")
    parser.add_argument("--mode", choices=["cnn", "svm"], default="cnn", help="cnn = classifier head, svm = CNN features + SVM.")
    parser.add_argument("--input", required=True, help="Image file or folder.")
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    parser.add_argument("--cnn-checkpoint", default=None, help="Override CNN checkpoint path.")
    parser.add_argument("--svm-model", default=None, help="Override SVM .joblib path when --mode svm.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def normalize_model_name(name: str) -> str:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "resnet_18": "resnet18",
        "resnet_50": "resnet50",
        "convnexttiny": "convnext_tiny",
        "convnext_t": "convnext_tiny",
    }
    return aliases.get(normalized, normalized)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {path}")
    return cfg


def resolve_path(path_text: str | Path, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def label_names(cfg: dict) -> list[str]:
    raw = cfg["labels"]
    return [raw[str(i)] for i in range(len(raw))]


def build_transform(cfg: dict) -> transforms.Compose:
    prep = cfg["preprocess"]
    pad_color = int(prep.get("pad_color", 255))
    fill = (pad_color, pad_color, pad_color)
    return transforms.Compose(
        [
            LetterboxResize(prep["input_size"], fill=fill),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_model(model_name: str, num_classes: int) -> nn.Module:
    name = normalize_model_name(model_name)
    if name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "convnext_tiny":
        model = convnext_tiny(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    supported = ", ".join(sorted(SUPPORTED_MODELS))
    raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_cnn(checkpoint_path: Path, cfg: dict, num_classes: int, device: torch.device) -> tuple[nn.Module, str]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint_path}")

    checkpoint = torch_load(checkpoint_path, device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    checkpoint_name = checkpoint.get("model_name") if isinstance(checkpoint, dict) else None
    model_name = normalize_model_name(checkpoint_name or cfg["model"]["name"])
    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_name


def list_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]
    if input_path.is_dir():
        images = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        return sorted(images)
    raise FileNotFoundError(f"Input not found: {input_path}")


def load_image_batch(paths: list[Path], transform) -> torch.Tensor:
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            tensors.append(transform(image))
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def predict_cnn(model: nn.Module, batch: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    logits = model(batch.to(device))
    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    return predictions, probabilities


@torch.no_grad()
def extract_features(model: nn.Module, model_name: str, batch: torch.Tensor, device: torch.device) -> np.ndarray:
    extractor = CNNFeatureExtractor(model, model_name).to(device)
    extractor.eval()
    features = extractor(batch.to(device))
    return features.detach().cpu().numpy().astype("float32")


def add_probability_columns(rows: list[dict], probabilities: np.ndarray | None, names: list[str]) -> None:
    if probabilities is None:
        return
    for row, probs in zip(rows, probabilities):
        for idx, name in enumerate(names):
            row[f"prob_{name}"] = float(probs[idx])


def run_inference(
    mode: str,
    image_paths: list[Path],
    cnn_model: nn.Module,
    model_name: str,
    svm_model,
    transform,
    device: torch.device,
    batch_size: int,
    names: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch = load_image_batch(batch_paths, transform)

        if mode == "cnn":
            predictions, probabilities = predict_cnn(cnn_model, batch, device)
        else:
            features = extract_features(cnn_model, model_name, batch, device)
            predictions = svm_model.predict(features)
            probabilities = svm_model.predict_proba(features) if hasattr(svm_model, "predict_proba") else None

        batch_rows = []
        for path, pred in zip(batch_paths, predictions):
            pred_int = int(pred)
            batch_rows.append(
                {
                    "path": str(path),
                    "pred_label": pred_int,
                    "pred_name": names[pred_int],
                }
            )
        add_probability_columns(batch_rows, probabilities, names)
        rows.extend(batch_rows)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve() if args.config else script_dir / "config.yaml"
    cfg = load_config(config_path)
    config_dir = config_path.parent

    names = label_names(cfg)
    device_name = args.device or cfg.get("runtime", {}).get("device", "auto")
    batch_size = args.batch_size or int(cfg.get("runtime", {}).get("batch_size", 32))
    device = get_device(device_name)

    checkpoint_path = resolve_path(args.cnn_checkpoint or cfg["paths"]["cnn_checkpoint"], config_dir)
    svm_path = resolve_path(args.svm_model or cfg["paths"]["svm_model"], config_dir)
    input_path = Path(args.input).resolve()

    image_paths = list_images(input_path)
    if not image_paths:
        raise ValueError(f"No images found in: {input_path}")

    transform = build_transform(cfg)
    cnn_model, model_name = load_cnn(checkpoint_path, cfg, num_classes=len(names), device=device)
    svm_model = None
    if args.mode == "svm":
        if not svm_path.exists():
            raise FileNotFoundError(f"SVM model not found: {svm_path}")
        svm_model = joblib.load(svm_path)

    result = run_inference(
        mode=args.mode,
        image_paths=image_paths,
        cnn_model=cnn_model,
        model_name=model_name,
        svm_model=svm_model,
        transform=transform,
        device=device,
        batch_size=batch_size,
        names=names,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved predictions to: {output_path}")
    elif len(result) == 1:
        print(json.dumps(result.iloc[0].to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()

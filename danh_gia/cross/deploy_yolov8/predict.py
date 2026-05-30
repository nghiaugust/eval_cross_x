from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageOps
from torchvision import transforms
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 classification inference.")
    parser.add_argument("--config", default=None, help="Deploy config. Default: deploy_yolov8/config.yaml")
    parser.add_argument("--input", required=True, help="Image file or folder.")
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    parser.add_argument("--checkpoint", default=None, help="Override YOLOv8 .pt checkpoint path.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


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


def config_label_names(cfg: dict) -> list[str]:
    raw = cfg["labels"]
    return [raw[str(i)] for i in range(len(raw))]


def model_label_names(yolo_model: YOLO, cfg: dict) -> list[str]:
    fallback = config_label_names(cfg)
    raw = getattr(yolo_model, "names", None)
    if isinstance(raw, dict) and raw:
        return [str(raw[i]) for i in range(len(raw))]
    if isinstance(raw, list) and raw:
        return [str(name) for name in raw]
    return fallback


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


def normalize_probabilities(outputs) -> torch.Tensor:
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]
    probabilities = outputs
    row_sums = probabilities.sum(dim=1)
    if probabilities.min().item() < 0 or not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
        probabilities = torch.softmax(probabilities, dim=1)
    return probabilities


@torch.no_grad()
def predict_batch(model: torch.nn.Module, batch: torch.Tensor, device: torch.device) -> np.ndarray:
    outputs = model(batch.to(device))
    probabilities = normalize_probabilities(outputs)
    return probabilities.detach().cpu().numpy()


def add_probability_columns(rows: list[dict], probabilities: np.ndarray, names: list[str]) -> None:
    for row, probs in zip(rows, probabilities):
        for idx, name in enumerate(names):
            row[f"prob_{name}"] = float(probs[idx])


def run_inference(
    image_paths: list[Path],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
    names: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch = load_image_batch(batch_paths, transform)
        probabilities = predict_batch(model, batch, device)
        predictions = probabilities.argmax(axis=1)

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

    device_name = args.device or cfg.get("runtime", {}).get("device", "auto")
    batch_size = args.batch_size or int(cfg.get("runtime", {}).get("batch_size", 32))
    checkpoint_path = resolve_path(args.checkpoint or cfg["paths"]["checkpoint"], config_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"YOLOv8 checkpoint not found: {checkpoint_path}")

    input_path = Path(args.input).resolve()
    image_paths = list_images(input_path)
    if not image_paths:
        raise ValueError(f"No images found in: {input_path}")

    device = get_device(device_name)
    yolo = YOLO(str(checkpoint_path))
    names = model_label_names(yolo, cfg)
    model = yolo.model.to(device)
    model.eval()
    transform = build_transform(cfg)

    result = run_inference(
        image_paths=image_paths,
        model=model,
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

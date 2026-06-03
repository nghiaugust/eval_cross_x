from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm.auto import tqdm


BASE_DIR = Path(__file__).resolve().parent
EVAL_ROOT = BASE_DIR.parent
DEFAULT_CONFIG = BASE_DIR / "deploy_yolov8" / "config.yaml"
DEFAULT_OUTPUT_DIR = "ket_qua"
DEFAULT_OUTPUT_FILE = "cross_metrics_yolov8.csv"
DEFAULT_PREDICTIONS_FILE = "cross_predictions_yolov8.csv"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 classification deploy model on data/0 and data/1.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YOLOv8 deploy config path.")
    parser.add_argument("--data", default="data", help="Dataset folder containing class folders 0 and 1.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for output CSV files. Relative paths are created under danh_gia/.",
    )
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="Metrics CSV filename.")
    parser.add_argument(
        "--predictions-file",
        default=None,
        help=f"Optional per-image predictions CSV filename. Example: {DEFAULT_PREDICTIONS_FILE}",
    )
    parser.add_argument("--checkpoint", default=None, help="Override YOLOv8 .pt checkpoint path.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ... Default: deploy config runtime.device.")
    parser.add_argument("--batch-size", type=int, default=None, help="Default: deploy config runtime.batch_size.")
    parser.add_argument(
        "--prediction-map",
        default="auto",
        help=(
            "Map deploy prediction labels to evaluate labels. Use auto, identity, invert, "
            "or comma mapping like 1,0 where index=deploy label and value=evaluate label."
        ),
    )
    parser.add_argument(
        "--python-path",
        action="append",
        default=[],
        help="Optional folder added to sys.path before loading checkpoint. Repeatable.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {path}")
    return cfg


def resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def resolve_data_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()


def resolve_output_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (EVAL_ROOT / path).resolve()


def natural_key(path: Path) -> list[Any]:
    parts: list[Any] = []
    text = path.as_posix()
    current = ""
    for char in text:
        if char.isdigit():
            current += char
        else:
            if current:
                parts.append(int(current))
                current = ""
            parts.append(char)
    if current:
        parts.append(int(current))
    return parts


def list_eval_samples(data_dir: Path) -> tuple[list[Path], np.ndarray]:
    image_paths: list[Path] = []
    labels: list[int] = []

    for label in (0, 1):
        class_dir = data_dir / str(label)
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        paths = [path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        paths = sorted(paths, key=natural_key)
        image_paths.extend(paths)
        labels.extend([label] * len(paths))

    if not image_paths:
        raise ValueError(f"No images found in: {data_dir}")
    return image_paths, np.asarray(labels, dtype=np.int64)


def import_yolo_deploy_module(config_path: Path) -> ModuleType:
    script_path = config_path.parent / "predict.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing YOLOv8 deploy script: {script_path}")

    module_name = "deploy_yolov8_predict_for_eval"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import YOLOv8 deploy script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def config_label_names(cfg: dict) -> list[str]:
    raw = cfg["labels"]
    return [str(raw[str(idx)]) for idx in range(len(raw))]


def eval_label_from_name(name: str) -> int | None:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"ten", "none", "no", "negative", "khong_gach", "khonggach", "0"}:
        return 0
    if normalized in {"gach_ten", "gachten", "co_gach", "cogach", "x_mark", "positive", "1"}:
        return 1
    if "khong" in normalized or "none" in normalized:
        return 0
    if "gach" in normalized or "mark" in normalized or "cross" in normalized:
        return 1
    return None


def build_prediction_map(raw: str, names: Sequence[str]) -> dict[int, int]:
    text = raw.strip().lower()
    if text == "identity":
        return {idx: idx for idx in range(len(names))}
    if text == "invert":
        if len(names) != 2:
            raise ValueError("--prediction-map invert only supports 2 classes.")
        return {0: 1, 1: 0}
    if text != "auto":
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
        if len(values) != len(names):
            raise ValueError(f"--prediction-map has {len(values)} values, expected {len(names)}.")
        return {idx: value for idx, value in enumerate(values)}

    mapping: dict[int, int] = {}
    for idx, name in enumerate(names):
        mapped = eval_label_from_name(str(name))
        mapping[idx] = idx if mapped is None else mapped
    return mapping


def apply_prediction_map(predictions: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.asarray([mapping[int(pred)] for pred in predictions], dtype=np.int64)


def format_prediction_map(mapping: dict[int, int]) -> str:
    return ";".join(f"{key}->{value}" for key, value in sorted(mapping.items()))


def cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, fn) -> tuple[Any, float]:
    cuda_synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        result = fn()
    cuda_synchronize(device)
    return result, time.perf_counter() - start


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average="micro",
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def add_python_paths(paths: Sequence[str | Path], base_dir: Path) -> None:
    for raw_path in paths:
        path = resolve_path(raw_path, base_dir)
        if not path.exists():
            print(f"[WARN] python path does not exist, ignored: {path}")
            continue
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def install_src_compat(yolo_module: ModuleType) -> None:
    if "src.data" in sys.modules:
        return

    src_module = sys.modules.setdefault("src", ModuleType("src"))
    data_module = ModuleType("src.data")
    data_module.LetterboxResize = yolo_module.LetterboxResize
    setattr(src_module, "data", data_module)
    sys.modules["src.data"] = data_module


def evaluate_yolo(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)
    config_dir = config_path.parent
    data_dir = resolve_data_path(args.data)
    image_paths, y_true = list_eval_samples(data_dir)

    config_python_paths = cfg.get("paths", {}).get("python_paths", [])
    if isinstance(config_python_paths, str):
        config_python_paths = [config_python_paths]
    add_python_paths([*config_python_paths, *args.python_path], config_dir)

    module = import_yolo_deploy_module(config_path)
    install_src_compat(module)

    checkpoint_raw = args.checkpoint or cfg["paths"]["checkpoint"]
    checkpoint_path = resolve_path(checkpoint_raw, config_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"YOLOv8 checkpoint not found: {checkpoint_path}")

    device_name = args.device or cfg.get("runtime", {}).get("device", "auto")
    batch_size = args.batch_size or int(cfg.get("runtime", {}).get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    device = module.get_device(device_name)
    names = config_label_names(cfg)
    mapping = build_prediction_map(args.prediction_map, names)
    transform = module.build_transform(cfg)

    print(f"Dataset: {data_dir}")
    print(f"Images: {len(image_paths)} | class 0={int((y_true == 0).sum())}, class 1={int((y_true == 1).sum())}")
    print("Evaluate labels: 0=khong gach, 1=co gach")
    print(f"Deploy labels: {', '.join(f'{idx}={name}' for idx, name in enumerate(names))}")
    print(f"Prediction map: {format_prediction_map(mapping)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print("Timing: model inference only; image loading/preprocessing is outside the timer.")

    yolo = module.YOLO(str(checkpoint_path))
    model = yolo.model.to(device)
    model.eval()

    eval_predictions: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    inference_time = 0.0

    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(image_paths), batch_size), total=total_batches, desc="yolov8_yolo", unit="batch", dynamic_ncols=True):
        batch_paths = image_paths[start : start + batch_size]
        batch = module.load_image_batch(batch_paths, transform)

        probabilities, elapsed = timed_call(device, lambda: module.predict_batch(model, batch, device))
        inference_time += elapsed

        deploy_predictions = np.asarray(probabilities.argmax(axis=1), dtype=np.int64)
        mapped = apply_prediction_map(deploy_predictions, mapping)
        eval_predictions.extend(int(pred) for pred in mapped)

        for offset, (path, deploy_pred, eval_pred, probs) in enumerate(zip(batch_paths, deploy_predictions, mapped, probabilities)):
            row: dict[str, Any] = {
                "model": "yolov8",
                "mode": "yolo",
                "path": str(path),
                "true_label": int(y_true[start + offset]),
                "deploy_pred_label": int(deploy_pred),
                "deploy_pred_name": names[int(deploy_pred)],
                "pred_label": int(eval_pred),
            }
            for idx, name in enumerate(names):
                row[f"prob_{name}"] = float(probs[idx])
            prediction_rows.append(row)

    y_pred = np.asarray(eval_predictions, dtype=np.int64)
    metrics = compute_metrics(y_true, y_pred)
    metrics.update(
        {
            "model": "yolov8",
            "mode": "yolo",
            "status": "ok",
            "num_images": int(len(y_true)),
            "device": str(device),
            "batch_size": int(batch_size),
            "time_total_s": float(inference_time),
            "time_per_image_ms": float((inference_time / max(len(y_true), 1)) * 1000.0),
            "images_per_second": float(len(y_true) / inference_time) if inference_time > 0 else 0.0,
            "prediction_map": format_prediction_map(mapping),
            "checkpoint": str(checkpoint_path),
            "error": "",
        }
    )
    return metrics, pd.DataFrame(prediction_rows)


def ordered_metrics_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "model",
        "mode",
        "status",
        "num_images",
        "device",
        "batch_size",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "time_total_s",
        "time_per_image_ms",
        "images_per_second",
        "tn",
        "fp",
        "fn",
        "tp",
        "prediction_map",
        "checkpoint",
        "error",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


def main() -> None:
    args = parse_args()
    metrics, predictions = evaluate_yolo(args)

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_frame = ordered_metrics_frame([metrics])
    metrics_path = output_dir / args.output_file
    metrics_frame.to_csv(metrics_path, index=False)

    if args.predictions_file:
        predictions_path = output_dir / args.predictions_file
        predictions.to_csv(predictions_path, index=False)
        print(f"\nSaved predictions to: {predictions_path}")

    display_columns = [
        "model",
        "mode",
        "status",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "time_total_s",
        "time_per_image_ms",
        "images_per_second",
        "error",
    ]
    print("\nSummary")
    print(metrics_frame[display_columns].to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm


BASE_DIR = Path(__file__).resolve().parent
EVAL_ROOT = BASE_DIR.parent
DEFAULT_MODEL_NAMES = ("resnet18", "resnet50", "convnext_tiny")
DEFAULT_MODES = ("cnn", "svm")
DEFAULT_OUTPUT_DIR = "ket_qua"
DEFAULT_OUTPUT_FILE = "cross_metrics.csv"
DEFAULT_BATCH_SIZE = 1
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    deploy_dir: Path


MODEL_SPECS = {
    "resnet18": ModelSpec("resnet18", Path("deploy_resnet18")),
    "resnet50": ModelSpec("resnet50", Path("deploy_resnet50")),
    "convnext_tiny": ModelSpec("convnext_tiny", Path("deploy_convnext_tiny")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ResNet/ConvNeXt deploy models on data/0 and data/1. "
            "Each selected model is evaluated in both cnn and svm modes."
        )
    )
    parser.add_argument("--data", default="data", help="Dataset folder containing class folders 0 and 1.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for the single metrics CSV. Relative paths are created under danh_gia/.",
    )
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="Metrics CSV filename.")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODEL_NAMES),
        help="Comma-separated model names to evaluate: resnet18,resnet50,convnext_tiny.",
    )
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ... Default: deploy config runtime.device.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used for every model. Use the same value for x and cross timing comparisons.",
    )
    parser.add_argument(
        "--prediction-map",
        default="auto",
        help=(
            "Map deploy prediction labels to evaluate labels. Use auto, identity, invert, "
            "or comma mapping like 1,0 where index=deploy label and value=evaluate label."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first model error.")
    return parser.parse_args()


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

        paths = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        paths = sorted(paths, key=natural_key)
        image_paths.extend(paths)
        labels.extend([label] * len(paths))

    if not image_paths:
        raise ValueError(f"No images found in: {data_dir}")
    return image_paths, np.asarray(labels, dtype=np.int64)


def import_deploy_module(spec: ModelSpec) -> ModuleType:
    script_path = BASE_DIR / spec.deploy_dir / "predict.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing deploy script: {script_path}")

    module_name = f"deploy_{spec.name}_predict"
    import_spec = importlib.util.spec_from_file_location(module_name, script_path)
    if import_spec is None or import_spec.loader is None:
        raise ImportError(f"Cannot import deploy script: {script_path}")

    module = importlib.util.module_from_spec(import_spec)
    sys.modules[module_name] = module
    import_spec.loader.exec_module(module)
    return module


def parse_model_list(raw: str) -> list[str]:
    models = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in models if name not in MODEL_SPECS]
    if unknown:
        supported = ", ".join(MODEL_SPECS)
        raise ValueError(f"Unsupported model(s): {', '.join(unknown)}. Supported: {supported}")
    return models


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


def normalize_model_name(name: str) -> str:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "resnet_18": "resnet18",
        "resnet_50": "resnet50",
        "convnexttiny": "convnext_tiny",
        "convnext_t": "convnext_tiny",
    }
    return aliases.get(normalized, normalized)


def load_cnn_model(
    module: ModuleType,
    cfg: dict,
    checkpoint_path: Path,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    signature = inspect.signature(module.load_cnn)
    if "cfg" in signature.parameters:
        loaded = module.load_cnn(checkpoint_path, cfg, num_classes=num_classes, device=device)
    else:
        loaded = module.load_cnn(checkpoint_path, num_classes=num_classes, device=device)

    if isinstance(loaded, tuple):
        model, model_name = loaded
        return model, normalize_model_name(str(model_name))

    model_name = normalize_model_name(str(cfg.get("model", {}).get("name", "model")))
    return loaded, model_name


def build_feature_extractor(
    module: ModuleType,
    cnn_model: torch.nn.Module,
    model_name: str,
    device: torch.device,
) -> torch.nn.Module | None:
    if hasattr(module, "CNNFeatureExtractor"):
        extractor = module.CNNFeatureExtractor(cnn_model, model_name)
    else:
        extractor_classes = {
            "resnet18": "ResNet18FeatureExtractor",
            "resnet50": "ResNet50FeatureExtractor",
            "convnext_tiny": "ConvNeXtTinyFeatureExtractor",
        }
        class_name = extractor_classes.get(model_name)
        extractor_cls = getattr(module, class_name, None) if class_name else None
        if extractor_cls is None:
            return None
        extractor = extractor_cls(cnn_model)

    extractor.to(device)
    extractor.eval()
    return extractor


def extract_features_with_extractor(
    extractor: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    features = extractor(batch.to(device))
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    return np.asarray(features, dtype=np.float32)


def extract_features_with_module(
    module: ModuleType,
    cnn_model: torch.nn.Module,
    model_name: str,
    batch: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    signature = inspect.signature(module.extract_features)
    if "model_name" in signature.parameters:
        features = module.extract_features(cnn_model, model_name, batch, device)
    else:
        features = module.extract_features(cnn_model, batch, device)
    return np.asarray(features, dtype=np.float32)


def predict_batch(
    module: ModuleType,
    mode: str,
    cnn_model: torch.nn.Module,
    model_name: str,
    svm_model,
    extractor: torch.nn.Module | None,
    batch: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    if mode == "cnn":
        predictions, _ = module.predict_cnn(cnn_model, batch, device)
        return np.asarray(predictions, dtype=np.int64)

    if extractor is not None:
        features = extract_features_with_extractor(extractor, batch, device)
    else:
        features = extract_features_with_module(module, cnn_model, model_name, batch, device)
    return np.asarray(svm_model.predict(features), dtype=np.int64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> dict[str, float]:
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="micro",
        zero_division=0,
    )
    return {
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
    }


def evaluate_cnn_svm(
    spec: ModelSpec,
    mode: str,
    image_paths: list[Path],
    y_true: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    module = import_deploy_module(spec)
    config_path = BASE_DIR / spec.deploy_dir / "config.yaml"
    cfg = module.load_config(config_path)
    names = module.label_names(cfg)
    device_name = args.device or cfg.get("runtime", {}).get("device", "auto")
    device = module.get_device(device_name)
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE
    transform = module.build_transform(cfg)

    checkpoint_path = module.resolve_path(cfg["paths"]["cnn_checkpoint"], config_path.parent)
    svm_path = module.resolve_path(cfg["paths"]["svm_model"], config_path.parent)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint_path}")
    if mode == "svm" and not svm_path.exists():
        raise FileNotFoundError(f"SVM model not found: {svm_path}")

    print(f"\nLoading {spec.name}_{mode}")
    print(f"  CNN checkpoint: {checkpoint_path}")
    if mode == "svm":
        print(f"  SVM model: {svm_path}")

    cnn_model, model_name = load_cnn_model(module, cfg, checkpoint_path, num_classes=len(names), device=device)
    svm_model = None
    extractor = None
    if mode == "svm":
        svm_model = joblib.load(svm_path)
        extractor = build_feature_extractor(module, cnn_model, model_name, device)
    mapping = build_prediction_map(args.prediction_map, names)

    eval_predictions: list[int] = []
    inference_time = 0.0

    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    desc = f"{spec.name}_{mode}"
    for start in tqdm(range(0, len(image_paths), batch_size), total=total_batches, desc=desc, unit="batch", dynamic_ncols=True):
        batch_paths = image_paths[start : start + batch_size]
        batch = module.load_image_batch(batch_paths, transform)

        deploy_predictions, elapsed = timed_call(
            device,
            lambda: predict_batch(module, mode, cnn_model, model_name, svm_model, extractor, batch, device),
        )
        inference_time += elapsed
        mapped = apply_prediction_map(deploy_predictions, mapping)
        eval_predictions.extend(int(pred) for pred in mapped)

    labels = [0, 1]
    y_pred = np.asarray(eval_predictions, dtype=np.int64)
    metrics = compute_metrics(y_true, y_pred, labels)
    metrics.update(
        {
            "model": spec.name,
            "mode": mode,
            "status": "ok",
            "num_images": int(len(y_true)),
            "device": str(device),
            "batch_size": batch_size,
            "time_total_s": float(inference_time),
            "time_per_image_ms": float((inference_time / max(len(y_true), 1)) * 1000.0),
            "images_per_second": float(len(y_true) / inference_time) if inference_time > 0 else 0.0,
            "error": "",
        }
    )
    return metrics


def missing_row(spec: ModelSpec, mode: str, error: Exception) -> dict[str, Any]:
    return {
        "model": spec.name,
        "mode": mode,
        "status": "error",
        "error": str(error),
    }


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
        "time_total_s",
        "time_per_image_ms",
        "images_per_second",
        "error",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


def resolve_data_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()


def resolve_output_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (EVAL_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    data_dir = resolve_data_path(args.data)
    image_paths, y_true = list_eval_samples(data_dir)
    model_names = parse_model_list(args.models)

    print(f"Dataset: {data_dir}")
    print(f"Images: {len(image_paths)} | class 0={int((y_true == 0).sum())}, class 1={int((y_true == 1).sum())}")
    print("Evaluate labels: 0=khong gach, 1=co gach")
    print("Modes: cnn, svm")
    print(f"Batch size: {args.batch_size}")
    print("Timing: model inference only; image loading/preprocessing is outside the timer.")

    metric_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        spec = MODEL_SPECS[model_name]
        for mode in DEFAULT_MODES:
            try:
                metric_rows.append(evaluate_cnn_svm(spec, mode, image_paths, y_true, args))
            except Exception as exc:
                if args.fail_fast:
                    raise
                print(f"[WARN] {spec.name}_{mode} skipped: {exc}")
                metric_rows.append(missing_row(spec, mode, exc))

    metrics_frame = ordered_metrics_frame(metric_rows)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file
    metrics_frame.to_csv(output_path, index=False)

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
    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()

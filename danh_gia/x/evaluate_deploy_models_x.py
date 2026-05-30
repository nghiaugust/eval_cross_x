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
DEFAULT_DEPLOY_DIRS = ("deploy_resnet18", "deploy_resnet50", "deploy_convnext_tiny")
DEFAULT_DATA_LABEL_NAMES = ("no_x", "x_mark", "x_cancel")
DEFAULT_MODES = ("cnn", "svm")
DEFAULT_OUTPUT_DIR = "ket_qua"
DEFAULT_OUTPUT_FILE = "x_metrics.csv"
DEFAULT_BATCH_SIZE = 1
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class EvaluationTarget:
    deploy_dir: Path
    mode: str

    @property
    def model_name(self) -> str:
        return self.deploy_dir.name.removeprefix("deploy_")

    @property
    def name(self) -> str:
        return f"{self.model_name}_{self.mode}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ResNet/ConvNeXt deploy models on data/0, data/1, data/2. "
            "Each deploy folder is evaluated in both cnn and svm modes."
        )
    )
    parser.add_argument("--data", default="data", help="Dataset folder containing class folders 0, 1, 2.")
    parser.add_argument(
        "--deploy-dirs",
        nargs="+",
        default=list(DEFAULT_DEPLOY_DIRS),
        help="Deploy folders to evaluate. Default: deploy_resnet18 deploy_resnet50 deploy_convnext_tiny.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for the single metrics CSV. Relative paths are created under danh_gia/.",
    )
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="Metrics CSV filename.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ... Default: deploy config runtime.device.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used for every model. Use the same value for x and cross timing comparisons.",
    )
    parser.add_argument(
        "--data-label-names",
        nargs=3,
        default=list(DEFAULT_DATA_LABEL_NAMES),
        metavar=("LABEL_0", "LABEL_1", "LABEL_2"),
        help="Names of folders 0, 1, 2. Default: no_x x_mark x_cancel.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first model error.")
    return parser.parse_args()


def import_deploy_module(deploy_dir: Path) -> ModuleType:
    predict_path = deploy_dir / "predict.py"
    if not predict_path.exists():
        raise FileNotFoundError(f"Missing deploy predict.py: {predict_path}")

    module_name = f"deploy_predict_{deploy_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, predict_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import deploy module: {predict_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_dataset(
    data_dir: Path,
    data_label_names: Sequence[str],
    model_label_names: Sequence[str],
) -> tuple[list[Path], np.ndarray]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    model_label_to_id = {name: idx for idx, name in enumerate(model_label_names)}
    image_paths: list[Path] = []
    y_true: list[int] = []

    for folder_id, data_label_name in enumerate(data_label_names):
        class_dir = data_dir / str(folder_id)
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        if data_label_name not in model_label_to_id:
            labels = ", ".join(model_label_names)
            raise ValueError(f"Data label '{data_label_name}' from folder {folder_id} is not in deploy labels: {labels}")

        model_label_id = model_label_to_id[data_label_name]
        class_images = sorted(
            path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        image_paths.extend(class_images)
        y_true.extend([model_label_id] * len(class_images))

    if not image_paths:
        raise ValueError(f"No images found under: {data_dir}")
    return image_paths, np.asarray(y_true, dtype=np.int64)


def resolve_deploy_path(module: ModuleType, path_text: str | Path, config_dir: Path) -> Path:
    return module.resolve_path(path_text, config_dir)


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
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }


def evaluate_target(
    target: EvaluationTarget,
    data_dir: Path,
    data_label_names: Sequence[str],
    device_override: str | None,
    batch_size_override: int | None,
) -> dict[str, Any]:
    module = import_deploy_module(target.deploy_dir)
    config_path = target.deploy_dir / "config.yaml"
    cfg = module.load_config(config_path)
    config_dir = config_path.parent

    model_label_names = module.label_names(cfg)
    image_paths, y_true = load_dataset(data_dir, data_label_names, model_label_names)

    device_name = device_override or cfg.get("runtime", {}).get("device", "auto")
    device = module.get_device(device_name)
    batch_size = batch_size_override if batch_size_override is not None else DEFAULT_BATCH_SIZE

    checkpoint_path = resolve_deploy_path(module, cfg["paths"]["cnn_checkpoint"], config_dir)
    svm_path = resolve_deploy_path(module, cfg["paths"]["svm_model"], config_dir)

    print(f"\nLoading {target.name}")
    print(f"  CNN checkpoint: {checkpoint_path}")
    if target.mode == "svm":
        print(f"  SVM model: {svm_path}")

    transform = module.build_transform(cfg)
    cnn_model, model_name = load_cnn_model(module, cfg, checkpoint_path, num_classes=len(model_label_names), device=device)
    svm_model = None
    extractor = None
    if target.mode == "svm":
        if not svm_path.exists():
            raise FileNotFoundError(f"SVM model not found: {svm_path}")
        svm_model = joblib.load(svm_path)
        extractor = build_feature_extractor(module, cnn_model, model_name, device)

    predictions: list[int] = []
    total_inference_time = 0.0

    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    progress = tqdm(
        range(0, len(image_paths), batch_size),
        total=total_batches,
        desc=target.name,
        unit="batch",
        dynamic_ncols=True,
    )
    for start in progress:
        batch_paths = image_paths[start : start + batch_size]
        batch = module.load_image_batch(batch_paths, transform)

        batch_predictions, elapsed = timed_call(
            device,
            lambda: predict_batch(module, target.mode, cnn_model, model_name, svm_model, extractor, batch, device),
        )
        total_inference_time += elapsed
        predictions.extend(batch_predictions.tolist())

    labels = list(range(len(model_label_names)))
    y_pred = np.asarray(predictions, dtype=np.int64)
    metrics = compute_metrics(y_true, y_pred, labels)

    total_images = len(image_paths)
    metrics.update(
        {
            "model": target.model_name,
            "mode": target.mode,
            "status": "ok",
            "num_images": total_images,
            "device": str(device),
            "batch_size": batch_size,
            "time_total_s": float(total_inference_time),
            "time_per_image_ms": float((total_inference_time / total_images) * 1000.0),
            "images_per_second": float(total_images / total_inference_time) if total_inference_time > 0 else 0.0,
            "error": "",
        }
    )
    return metrics


def missing_row(target: EvaluationTarget, error: Exception) -> dict[str, Any]:
    return {
        "model": target.model_name,
        "mode": target.mode,
        "status": "error",
        "error": str(error),
    }


def make_targets(base_dir: Path, deploy_dirs: Sequence[str]) -> list[EvaluationTarget]:
    return [
        EvaluationTarget(deploy_dir=(base_dir / deploy_dir).resolve(), mode=mode)
        for deploy_dir in deploy_dirs
        for mode in DEFAULT_MODES
    ]


def ordered_metrics_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "model",
        "mode",
        "status",
        "num_images",
        "device",
        "batch_size",
        "precision_macro",
        "recall_macro",
        "f1_macro",
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
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {data_dir}")
    print("Modes: cnn, svm")
    print(f"Batch size: {args.batch_size}")
    print("Timing: model inference only; image loading/preprocessing is outside the timer.")

    summaries: list[dict[str, Any]] = []
    for target in make_targets(BASE_DIR, args.deploy_dirs):
        try:
            summaries.append(
                evaluate_target(
                    target=target,
                    data_dir=data_dir,
                    data_label_names=args.data_label_names,
                    device_override=args.device,
                    batch_size_override=args.batch_size,
                )
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            print(f"[WARN] {target.name} skipped: {exc}")
            summaries.append(missing_row(target, exc))

    metrics_frame = ordered_metrics_frame(summaries)
    output_path = output_dir / args.output_file
    metrics_frame.to_csv(output_path, index=False)

    display_columns = [
        "model",
        "mode",
        "status",
        "precision_macro",
        "recall_macro",
        "f1_macro",
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

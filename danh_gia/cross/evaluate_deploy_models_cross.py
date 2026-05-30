from __future__ import annotations

import argparse
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelSpec:
    name: str
    deploy_dir: Path
    kind: str


MODEL_SPECS = {
    "resnet18": ModelSpec("resnet18", Path("deploy_resnet18"), "cnn_svm"),
    "resnet50": ModelSpec("resnet50", Path("deploy_resnet50"), "cnn_svm"),
    "convnext_tiny": ModelSpec("convnext_tiny", Path("deploy_convnext_tiny"), "cnn_svm"),
    "yolov8": ModelSpec("yolov8", Path("deploy_yolov8"), "yolo"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all deploy models on evaluate/0 and evaluate/1 folders.")
    parser.add_argument("--data", default="evaluate", help="Folder with class subfolders 0 and 1.")
    parser.add_argument("--output", default="runs/deploy_evaluation/metrics.csv", help="CSV metrics output.")
    parser.add_argument("--predictions-output", default=None, help="Optional per-image predictions CSV output.")
    parser.add_argument(
        "--models",
        default="resnet18,resnet50,convnext_tiny,yolov8",
        help="Comma-separated model names to evaluate.",
    )
    parser.add_argument(
        "--cnn-mode",
        choices=["cnn", "svm", "both"],
        default="both",
        help="Mode for ResNet/ConvNeXt deploy folders. Default evaluates both cnn and svm.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=None, help="Override deploy batch size.")
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
    import_spec.loader.exec_module(module)
    return module


def parse_model_list(raw: str) -> list[str]:
    models = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in models if name not in MODEL_SPECS]
    if unknown:
        supported = ", ".join(MODEL_SPECS)
        raise ValueError(f"Unsupported model(s): {', '.join(unknown)}. Supported: {supported}")
    return models


def is_cuda_device(device: torch.device) -> bool:
    return device.type == "cuda"


def sync_if_needed(device: torch.device) -> None:
    if is_cuda_device(device):
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, fn):
    sync_if_needed(device)
    start = time.perf_counter()
    result = fn()
    sync_if_needed(device)
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


def build_prediction_map(raw: str, names: list[str]) -> dict[int, int]:
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision[1]),
        "recall": float(recall[1]),
        "f1": float(f1[1]),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "support_0": int(support[0]),
        "support_1": int(support[1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def missing_row(spec: ModelSpec, mode: str, error: Exception) -> dict[str, Any]:
    return {
        "model": spec.name,
        "mode": mode,
        "status": "error",
        "error": str(error),
    }


def resolve_cnn_paths(module: ModuleType, cfg: dict, config_dir: Path, mode: str) -> tuple[Path, Path | None]:
    checkpoint_path = module.resolve_path(cfg["paths"]["cnn_checkpoint"], config_dir)
    svm_path = None
    if mode == "svm":
        svm_path = module.resolve_path(cfg["paths"]["svm_model"], config_dir)
    return checkpoint_path, svm_path


def build_feature_extractor(module: ModuleType, spec: ModelSpec, cnn_model, model_name: str, device: torch.device):
    if spec.name == "resnet18":
        extractor = module.ResNet18FeatureExtractor(cnn_model)
    else:
        extractor = module.CNNFeatureExtractor(cnn_model, model_name)
    extractor.to(device)
    extractor.eval()
    return extractor


@torch.no_grad()
def extract_features(extractor, batch: torch.Tensor, device: torch.device) -> np.ndarray:
    features = extractor(batch.to(device))
    return features.detach().cpu().numpy().astype("float32")


def evaluate_cnn_svm(
    spec: ModelSpec,
    mode: str,
    image_paths: list[Path],
    y_true: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    module = import_deploy_module(spec)
    config_path = BASE_DIR / spec.deploy_dir / "config.yaml"
    cfg = module.load_config(config_path)
    names = module.label_names(cfg)
    device = module.get_device(args.device or cfg.get("runtime", {}).get("device", "auto"))
    batch_size = args.batch_size or int(cfg.get("runtime", {}).get("batch_size", 32))
    transform = module.build_transform(cfg)
    checkpoint_path, svm_path = resolve_cnn_paths(module, cfg, config_path.parent, mode)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint_path}")
    if mode == "svm" and svm_path is not None and not svm_path.exists():
        raise FileNotFoundError(f"SVM model not found: {svm_path}")

    if spec.name == "resnet18":
        cnn_model = module.load_cnn(checkpoint_path, num_classes=len(names), device=device)
        model_name = "resnet18"
    else:
        cnn_model, model_name = module.load_cnn(checkpoint_path, cfg, num_classes=len(names), device=device)

    svm_model = joblib.load(svm_path) if mode == "svm" and svm_path is not None else None
    extractor = build_feature_extractor(module, spec, cnn_model, model_name, device) if svm_model is not None else None
    mapping = build_prediction_map(args.prediction_map, names)

    deploy_predictions: list[int] = []
    eval_predictions: list[int] = []
    rows: list[dict[str, Any]] = []
    inference_time = 0.0

    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    desc = f"{spec.name}:{mode}"
    for start in tqdm(range(0, len(image_paths), batch_size), total=total_batches, desc=desc, unit="batch"):
        batch_paths = image_paths[start : start + batch_size]
        batch = module.load_image_batch(batch_paths, transform)

        if mode == "cnn":
            (preds, _probabilities), elapsed = timed_call(device, lambda: module.predict_cnn(cnn_model, batch, device))
        else:
            def predict_svm():
                features = extract_features(extractor, batch, device)
                return svm_model.predict(features)
            preds, elapsed = timed_call(device, predict_svm)

        inference_time += elapsed
        mapped = apply_prediction_map(np.asarray(preds), mapping)
        deploy_predictions.extend(int(pred) for pred in preds)
        eval_predictions.extend(int(pred) for pred in mapped)
        for offset, (path, deploy_pred, eval_pred) in enumerate(zip(batch_paths, preds, mapped)):
            rows.append(
                {
                    "model": spec.name,
                    "mode": mode,
                    "path": str(path),
                    "true_label": int(y_true[start + offset]),
                    "deploy_pred_label": int(deploy_pred),
                    "deploy_pred_name": names[int(deploy_pred)],
                    "pred_label": int(eval_pred),
                }
            )

    y_pred = np.asarray(eval_predictions, dtype=np.int64)
    metrics = compute_metrics(y_true, y_pred)
    metrics.update(
        {
            "model": spec.name,
            "mode": mode,
            "status": "ok",
            "num_images": int(len(y_true)),
            "time_total_s": float(inference_time),
            "time_per_image_ms": float((inference_time / max(len(y_true), 1)) * 1000.0),
            "images_per_second": float(len(y_true) / inference_time) if inference_time > 0 else 0.0,
            "prediction_map": format_prediction_map(mapping),
            "error": "",
        }
    )
    return metrics, pd.DataFrame(rows)


def evaluate_yolo(
    spec: ModelSpec,
    image_paths: list[Path],
    y_true: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    module = import_deploy_module(spec)
    config_path = BASE_DIR / spec.deploy_dir / "config.yaml"
    cfg = module.load_config(config_path)
    checkpoint_path = module.resolve_path(cfg["paths"]["checkpoint"], config_path.parent)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"YOLOv8 checkpoint not found: {checkpoint_path}")

    device = module.get_device(args.device or cfg.get("runtime", {}).get("device", "auto"))
    batch_size = args.batch_size or int(cfg.get("runtime", {}).get("batch_size", 32))
    yolo = module.YOLO(str(checkpoint_path))
    names = module.model_label_names(yolo, cfg)
    model = yolo.model.to(device)
    model.eval()
    transform = module.build_transform(cfg)
    mapping = build_prediction_map(args.prediction_map, names)

    eval_predictions: list[int] = []
    rows: list[dict[str, Any]] = []
    inference_time = 0.0

    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(image_paths), batch_size), total=total_batches, desc=spec.name, unit="batch"):
        batch_paths = image_paths[start : start + batch_size]
        batch = module.load_image_batch(batch_paths, transform)
        probabilities, elapsed = timed_call(device, lambda: module.predict_batch(model, batch, device))
        inference_time += elapsed

        deploy_preds = probabilities.argmax(axis=1)
        mapped = apply_prediction_map(np.asarray(deploy_preds), mapping)
        eval_predictions.extend(int(pred) for pred in mapped)
        for offset, (path, deploy_pred, eval_pred) in enumerate(zip(batch_paths, deploy_preds, mapped)):
            rows.append(
                {
                    "model": spec.name,
                    "mode": "classify",
                    "path": str(path),
                    "true_label": int(y_true[start + offset]),
                    "deploy_pred_label": int(deploy_pred),
                    "deploy_pred_name": names[int(deploy_pred)],
                    "pred_label": int(eval_pred),
                }
            )

    y_pred = np.asarray(eval_predictions, dtype=np.int64)
    metrics = compute_metrics(y_true, y_pred)
    metrics.update(
        {
            "model": spec.name,
            "mode": "classify",
            "status": "ok",
            "num_images": int(len(y_true)),
            "time_total_s": float(inference_time),
            "time_per_image_ms": float((inference_time / max(len(y_true), 1)) * 1000.0),
            "images_per_second": float(len(y_true) / inference_time) if inference_time > 0 else 0.0,
            "prediction_map": format_prediction_map(mapping),
            "error": "",
        }
    )
    return metrics, pd.DataFrame(rows)


def evaluate_model(
    spec: ModelSpec,
    image_paths: list[Path],
    y_true: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if spec.kind == "yolo":
        return evaluate_yolo(spec, image_paths, y_true, args)
    if args.cnn_mode == "both":
        raise ValueError("evaluate_model does not handle --cnn-mode both for CNN/SVM models.")
    return evaluate_cnn_svm(spec, args.cnn_mode, image_paths, y_true, args)


def cnn_modes(args: argparse.Namespace) -> list[str]:
    if args.cnn_mode == "both":
        return ["cnn", "svm"]
    return [args.cnn_mode]


def ordered_metrics_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "model",
        "mode",
        "status",
        "num_images",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "time_total_s",
        "time_per_image_ms",
        "images_per_second",
        "tp",
        "fp",
        "tn",
        "fn",
        "support_0",
        "support_1",
        "prediction_map",
        "error",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir
    image_paths, y_true = list_eval_samples(data_dir)
    model_names = parse_model_list(args.models)

    print(f"Dataset: {data_dir.resolve()}")
    print(f"Images: {len(image_paths)} | class 0={int((y_true == 0).sum())}, class 1={int((y_true == 1).sum())}")
    print("Evaluate labels: 0=khong gach, 1=co gach")

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name in model_names:
        spec = MODEL_SPECS[model_name]
        if spec.kind == "yolo":
            try:
                metrics, predictions = evaluate_yolo(spec, image_paths, y_true, args)
                metric_rows.append(metrics)
                prediction_frames.append(predictions)
            except Exception as exc:
                if args.fail_fast:
                    raise
                print(f"[WARN] {spec.name}:classify skipped: {exc}")
                metric_rows.append(missing_row(spec, "classify", exc))
            continue

        for mode in cnn_modes(args):
            try:
                metrics, predictions = evaluate_cnn_svm(spec, mode, image_paths, y_true, args)
                metric_rows.append(metrics)
                prediction_frames.append(predictions)
            except Exception as exc:
                if args.fail_fast:
                    raise
                print(f"[WARN] {spec.name}:{mode} skipped: {exc}")
                metric_rows.append(missing_row(spec, mode, exc))

    metrics_frame = ordered_metrics_frame(metric_rows)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(output_path, index=False)

    if args.predictions_output and prediction_frames:
        predictions_path = Path(args.predictions_output)
        if not predictions_path.is_absolute():
            predictions_path = BASE_DIR / predictions_path
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)
        print(f"Saved predictions to: {predictions_path}")

    display_columns = [
        "model",
        "mode",
        "status",
        "precision",
        "recall",
        "f1",
        "time_total_s",
        "time_per_image_ms",
        "images_per_second",
        "error",
    ]
    print(metrics_frame[display_columns].to_string(index=False))
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()

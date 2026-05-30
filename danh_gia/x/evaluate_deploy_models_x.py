from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tqdm.auto import tqdm


DEFAULT_DEPLOY_DIRS = ("deploy_resnet18", "deploy_resnet50", "deploy_convnext_tiny")
DEFAULT_DATA_LABEL_NAMES = ("no_x", "x_mark", "x_cancel")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class EvaluationTarget:
    deploy_dir: Path
    mode: str

    @property
    def name(self) -> str:
        return f"{self.deploy_dir.name}_{self.mode}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deploy models on a folder dataset. "
            "Default data layout: data/0, data/1, data/2 next to deploy_* folders."
        )
    )
    parser.add_argument("--data", default="data", help="Dataset folder containing class folders 0, 1, 2.")
    parser.add_argument(
        "--deploy-dirs",
        nargs="+",
        default=list(DEFAULT_DEPLOY_DIRS),
        help="Deploy folders to evaluate.",
    )
    parser.add_argument(
        "--mode",
        choices=["cnn", "svm", "both"],
        default="both",
        help="Evaluate CNN, SVM, or both modes for each deploy folder.",
    )
    parser.add_argument("--output-dir", default="deploy_eval_results", help="Folder for CSV outputs.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ... Default: deploy config runtime.device.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override deploy config batch size.")
    parser.add_argument(
        "--data-label-names",
        nargs=3,
        default=list(DEFAULT_DATA_LABEL_NAMES),
        metavar=("LABEL_0", "LABEL_1", "LABEL_2"),
        help="Names of folders 0, 1, 2. Default: no_x x_mark x_cancel.",
    )
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


def load_dataset(data_dir: Path, data_label_names: Sequence[str], model_label_names: Sequence[str]) -> tuple[list[Path], list[int], list[str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    model_label_to_id = {name: idx for idx, name in enumerate(model_label_names)}
    image_paths: list[Path] = []
    y_true: list[int] = []
    true_names: list[str] = []

    for folder_id, data_label_name in enumerate(data_label_names):
        class_dir = data_dir / str(folder_id)
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        if data_label_name not in model_label_to_id:
            labels = ", ".join(model_label_names)
            raise ValueError(
                f"Data label '{data_label_name}' from folder {folder_id} is not in deploy labels: {labels}"
            )

        model_label_id = model_label_to_id[data_label_name]
        class_images = sorted(
            path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        image_paths.extend(class_images)
        y_true.extend([model_label_id] * len(class_images))
        true_names.extend([data_label_name] * len(class_images))

    if not image_paths:
        raise ValueError(f"No images found under: {data_dir}")
    return image_paths, y_true, true_names


def resolve_deploy_path(module: ModuleType, path_text: str | Path, config_dir: Path) -> Path:
    return module.resolve_path(path_text, config_dir)


def cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_batch(
    module: ModuleType,
    mode: str,
    cnn_model,
    svm_model,
    batch: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    if mode == "cnn":
        predictions, _ = module.predict_cnn(cnn_model, batch, device)
        return np.asarray(predictions, dtype=np.int64)

    features = module.extract_features(cnn_model, batch, device)
    return np.asarray(svm_model.predict(features), dtype=np.int64)


def evaluate_target(
    target: EvaluationTarget,
    data_dir: Path,
    data_label_names: Sequence[str],
    output_dir: Path,
    device_override: str | None,
    batch_size_override: int | None,
) -> dict:
    module = import_deploy_module(target.deploy_dir)
    config_path = target.deploy_dir / "config.yaml"
    cfg = module.load_config(config_path)
    config_dir = config_path.parent

    model_label_names = module.label_names(cfg)
    image_paths, y_true, true_names = load_dataset(data_dir, data_label_names, model_label_names)

    device_name = device_override or cfg.get("runtime", {}).get("device", "auto")
    device = module.get_device(device_name)
    batch_size = batch_size_override or int(cfg.get("runtime", {}).get("batch_size", 8))

    checkpoint_path = resolve_deploy_path(module, cfg["paths"]["cnn_checkpoint"], config_dir)
    svm_path = resolve_deploy_path(module, cfg["paths"]["svm_model"], config_dir)

    print(f"\nLoading {target.name}")
    print(f"  CNN checkpoint: {checkpoint_path}")
    if target.mode == "svm":
        print(f"  SVM model: {svm_path}")

    transform = module.build_transform(cfg)
    cnn_model = module.load_cnn(checkpoint_path, num_classes=len(model_label_names), device=device)
    svm_model = None
    if target.mode == "svm":
        if not svm_path.exists():
            raise FileNotFoundError(f"SVM model not found: {svm_path}")
        svm_model = joblib.load(svm_path)

    predictions: list[int] = []
    total_inference_time = 0.0

    progress = tqdm(
        range(0, len(image_paths), batch_size),
        desc=target.name,
        unit="batch",
        dynamic_ncols=True,
    )
    for start in progress:
        batch_paths = image_paths[start : start + batch_size]
        batch = module.load_image_batch(batch_paths, transform)

        cuda_synchronize(device)
        start_time = time.perf_counter()
        batch_predictions = predict_batch(module, target.mode, cnn_model, svm_model, batch, device)
        cuda_synchronize(device)
        total_inference_time += time.perf_counter() - start_time

        predictions.extend(batch_predictions.tolist())

    labels = list(range(len(model_label_names)))
    report = classification_report(
        y_true,
        predictions,
        labels=labels,
        target_names=model_label_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / f"{target.name}_classification_report.csv", index=True)

    predictions_df = pd.DataFrame(
        {
            "path": [str(path) for path in image_paths],
            "true_label": y_true,
            "true_name": true_names,
            "pred_label": predictions,
            "pred_name": [model_label_names[pred] for pred in predictions],
            "correct": [true == pred for true, pred in zip(y_true, predictions)],
        }
    )
    predictions_df.to_csv(output_dir / f"{target.name}_predictions.csv", index=False)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    total_images = len(image_paths)
    return {
        "model": target.deploy_dir.name,
        "mode": target.mode,
        "num_images": total_images,
        "device": str(device),
        "batch_size": batch_size,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "time_total_s": float(total_inference_time),
        "time_ms_per_image": float((total_inference_time / total_images) * 1000),
        "images_per_second": float(total_images / total_inference_time) if total_inference_time > 0 else float("inf"),
    }


def make_targets(base_dir: Path, deploy_dirs: Sequence[str], mode: str) -> list[EvaluationTarget]:
    modes = ["cnn", "svm"] if mode == "both" else [mode]
    return [
        EvaluationTarget(deploy_dir=(base_dir / deploy_dir).resolve(), mode=model_mode)
        for deploy_dir in deploy_dirs
        for model_mode in modes
    ]


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    output_dir = (base_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = make_targets(base_dir, args.deploy_dirs, args.mode)
    summaries = [
        evaluate_target(
            target=target,
            data_dir=data_dir,
            data_label_names=args.data_label_names,
            output_dir=output_dir,
            device_override=args.device,
            batch_size_override=args.batch_size,
        )
        for target in targets
    ]

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSummary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()


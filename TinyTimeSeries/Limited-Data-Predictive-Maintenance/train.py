"""Failure-recall-oriented training for Azure predictive maintenance.

The training strategy deliberately avoids stacking aggressive oversampling,
large inverse-frequency weights, and high-gamma focal loss at the same time.
It uses:

* event-aware data sampling from dataset.py;
* mild effective-number class weighting;
* focal gamma 1.0 instead of 2.0;
* warm-up plus cosine learning-rate decay;
* validation-calibrated none/failure decision threshold;
* model selection by validation macro recall;
* one final test with the frozen validation threshold;
* lead-time and event-level diagnostics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from dataset import ID_TO_LABEL, LABEL_TO_ID, create_experiment_dataloaders
from models import (
    build_model,
    count_parameters,
    get_model_config,
    get_training_kwargs,
    normalize_model_name,
)
from utils import (
    format_percentage,
    format_recall_summary,
    save_classification_record,
    set_seed,
)


MODEL_NAME = "transformer"
# Available examples:
# lstm / transformer / linear_transformer / retnet / mamba /
# modern_tcn / patch_tst / timesnet / tslanet / lite

LOSS_NAME = "balanced_focal"
# LOSS_NAME = "balanced_ce"

FOCAL_GAMMA = 1.0
EFFECTIVE_NUMBER_BETA = 0.999
MAXIMUM_CLASS_WEIGHT = 5.0
LABEL_SMOOTHING = 0.02

EPOCHS = 30
WARMUP_EPOCHS = 2
EARLY_STOPPING_PATIENCE = 8
GRADIENT_CLIP_NORM = 1.0
RANDOM_SEED = 42

# Threshold calibration is performed only on validation. A minimum normal-class
# recall prevents an apparently high macro recall produced by excessive alarms.
THRESHOLD_MIN = 0.25
THRESHOLD_MAX = 0.75
THRESHOLD_STEPS = 101
MIN_VALIDATION_NONE_RECALL = 0.75

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RECORD_DIR = PROJECT_DIR / "Record"


class ClassBalancedLoss(nn.Module):
    """Mild class-balanced CE or focal loss with per-class weights."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        loss_name: str,
        focal_gamma: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        normalized = loss_name.strip().lower()
        if normalized not in {"balanced_focal", "balanced_ce"}:
            raise ValueError(
                "loss_name must be 'balanced_focal' or 'balanced_ce'."
            )

        self.loss_name = normalized
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "balanced_ce":
            return F.cross_entropy(
                logits,
                targets,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )

        log_probabilities = F.log_softmax(logits, dim=-1)
        target_log_probabilities = log_probabilities.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        target_probabilities = target_log_probabilities.exp()
        sample_weights = self.class_weights[targets]

        focal_factor = (1.0 - target_probabilities).pow(self.focal_gamma)
        losses = -sample_weights * focal_factor * target_log_probabilities
        return losses.mean()


def calculate_effective_number_weights(
    class_counts: torch.Tensor,
    beta: float,
    maximum_weight: float,
) -> torch.Tensor:
    """Compute effective-number weights and keep them deliberately mild."""
    counts = class_counts.float().clamp_min(1.0)
    beta_tensor = torch.tensor(beta, dtype=torch.float32)
    effective_number = 1.0 - torch.pow(beta_tensor, counts)
    weights = (1.0 - beta_tensor) / effective_number.clamp_min(1e-12)
    weights = weights / weights.mean().clamp_min(1e-12)
    return weights.clamp(max=maximum_weight)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> LambdaLR:
    total_steps = max(1, EPOCHS * steps_per_epoch)
    warmup_steps = max(1, WARMUP_EPOCHS * steps_per_epoch)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, float(step + 1) / float(warmup_steps))

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=multiplier)


def class_recalls(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> list[float]:
    recalls: list[float] = []
    for class_id in range(num_classes):
        mask = targets == class_id
        denominator = int(mask.sum())
        recalls.append(
            float((predictions[mask] == class_id).mean())
            if denominator > 0
            else 0.0
        )
    return recalls


def predict_with_failure_threshold(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Separate none-vs-failure confidence from component selection.

    The component is selected from comp1--comp4. The calibrated score compares
    that best component probability directly with none, avoiding the dilution
    that occurs when all failure probabilities are summed.
    """
    none_probability = probabilities[:, 0]
    failure_probabilities = probabilities[:, 1:]
    failure_index = failure_probabilities.argmax(axis=1)
    best_failure_probability = failure_probabilities[
        np.arange(len(failure_probabilities)), failure_index
    ]

    failure_confidence = best_failure_probability / (
        none_probability + best_failure_probability + 1e-12
    )
    predictions = failure_index.astype(np.int64) + 1
    predictions[failure_confidence < threshold] = LABEL_TO_ID["none"]
    return predictions


def calibrate_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> tuple[float, list[float], np.ndarray]:
    """Choose a validation-only threshold maximizing constrained macro recall."""
    best_threshold = 0.5
    best_recalls: list[float] = [0.0] * num_classes
    best_predictions = predict_with_failure_threshold(probabilities, 0.5)
    best_key = (-1.0, -1.0, -1.0)

    for threshold in np.linspace(
        THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS
    ):
        predictions = predict_with_failure_threshold(probabilities, float(threshold))
        recalls = class_recalls(targets, predictions, num_classes)
        none_recall = recalls[0]
        failure_macro = float(np.mean(recalls[1:]))
        macro_recall = float(np.mean(recalls))

        if none_recall < MIN_VALIDATION_NONE_RECALL:
            continue

        # Macro recall is primary. Failure recall breaks ties, followed by a
        # threshold nearer 0.5 for better calibration stability.
        key = (
            macro_recall,
            failure_macro,
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_recalls = recalls
            best_predictions = predictions

    # If every threshold violates the none-recall constraint, fall back to the
    # ordinary argmax-equivalent threshold rather than tuning on test data.
    if best_key[0] < 0.0:
        best_threshold = 0.5
        best_predictions = predict_with_failure_threshold(probabilities, 0.5)
        best_recalls = class_recalls(targets, best_predictions, num_classes)

    return best_threshold, best_recalls, best_predictions


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0
    use_amp = device.type == "cuda"

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:02d}/{EPOCHS:02d} [Train]",
        dynamic_ncols=True,
    )

    for features, labels in progress:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(features)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        predictions = logits.detach().argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        correct_count += int((predictions == labels).sum().item())
        sample_count += batch_size

        progress.set_postfix(
            loss=f"{total_loss / max(sample_count, 1):.4f}",
            acc=format_percentage(correct_count / max(sample_count, 1)),
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return total_loss / sample_count, correct_count / sample_count


@torch.no_grad()
def evaluate_probabilities(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    description: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    progress = tqdm(loader, desc=description, leave=False, dynamic_ncols=True)
    for features, labels in progress:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(features)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        targets.append(labels.cpu().numpy())

    return (
        total_loss / sample_count,
        np.concatenate(probabilities, axis=0),
        np.concatenate(targets, axis=0).astype(np.int64),
    )


def lead_time_metrics(
    dataset: object,
    predictions: np.ndarray,
) -> dict[str, tuple[float, float, int]]:
    """Return component-correct and any-failure recall by lead-time bin."""
    windows = getattr(dataset, "windows")
    result: dict[str, tuple[float, float, int]] = {}

    for lower, upper in ((0, 24), (24, 48), (48, 72)):
        indices = [
            index
            for index, window in enumerate(windows)
            if window.label_id != 0
            and lower < window.hours_to_failure <= upper
        ]
        if not indices:
            result[f"{lower}-{upper}h"] = (0.0, 0.0, 0)
            continue

        true_labels = np.array([windows[index].label_id for index in indices])
        predicted = predictions[np.asarray(indices)]
        component_recall = float((predicted == true_labels).mean())
        failure_detection_recall = float((predicted != 0).mean())
        result[f"{lower}-{upper}h"] = (
            component_recall,
            failure_detection_recall,
            len(indices),
        )

    return result


def event_level_recall(dataset: object, predictions: np.ndarray) -> tuple[float, int]:
    """Count an event as detected when any window predicts its component."""
    windows = getattr(dataset, "windows")
    grouped: dict[tuple[int, int, int], list[int]] = {}

    for index, window in enumerate(windows):
        if window.label_id == 0 or window.failure_event_id < 0:
            continue
        key = (window.machine_id, window.failure_event_id, window.label_id)
        grouped.setdefault(key, []).append(index)

    if not grouped:
        return 0.0, 0

    detected = 0
    for (_, _, label_id), indices in grouped.items():
        if np.any(predictions[np.asarray(indices)] == label_id):
            detected += 1
    return detected / len(grouped), len(grouped)


def _class_count_tensor(loader: torch.utils.data.DataLoader) -> torch.Tensor:
    counts_dict = loader.dataset.class_counts()
    return torch.tensor(
        [counts_dict[ID_TO_LABEL[index]] for index in range(len(ID_TO_LABEL))],
        dtype=torch.float32,
    )


def main() -> None:
    set_seed(RANDOM_SEED)
    model_name = normalize_model_name(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = create_experiment_dataloaders()
    train_loader = loaders.train
    validation_loader = loaders.validation
    test_loader = loaders.test

    sample_features, _ = next(iter(train_loader))
    class_names = [ID_TO_LABEL[index] for index in range(len(ID_TO_LABEL))]
    class_counts = _class_count_tensor(train_loader)
    class_weights = calculate_effective_number_weights(
        class_counts,
        beta=EFFECTIVE_NUMBER_BETA,
        maximum_weight=MAXIMUM_CLASS_WEIGHT,
    ).to(device)

    model = build_model(
        model_name=model_name,
        input_size=sample_features.shape[-1],
        num_classes=len(class_names),
        sequence_length=sample_features.shape[1],
    ).to(device)

    training_kwargs = get_training_kwargs(model_name)
    model_config = get_model_config(model_name)
    criterion = ClassBalancedLoss(
        class_weights=class_weights,
        loss_name=LOSS_NAME,
        focal_gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=training_kwargs["learning_rate"],
        weight_decay=training_kwargs["weight_decay"],
    )
    scheduler = create_scheduler(optimizer, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}.pt"
    record_path = RECORD_DIR / f"{model_name}.txt"

    print(f"{model_name.upper()} TRAINING")
    print("=" * 92)
    print(f"Device: {device}")
    print(f"Input shape: {tuple(sample_features.shape[1:])}")
    print(f"Feature count: {sample_features.shape[-1]}")
    print(f"Loss: {LOSS_NAME}, focal gamma: {FOCAL_GAMMA}")
    print(f"Class counts: {class_counts.int().tolist()}")
    print(f"Class weights: {[round(v, 4) for v in class_weights.cpu().tolist()]}")
    print(f"Training sequences: {len(train_loader.dataset):,}")
    print(f"Validation sequences: {len(validation_loader.dataset):,}")
    print(f"Testing sequences: {len(test_loader.dataset):,}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    best_macro_recall = -1.0
    best_failure_macro = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_history: list[str] = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
        )

        validation_loss, validation_probabilities, validation_targets = (
            evaluate_probabilities(
                model,
                validation_loader,
                criterion,
                device,
                f"Epoch {epoch:02d}/{EPOCHS:02d} [Validation]",
            )
        )
        threshold, validation_recalls, _ = calibrate_threshold(
            validation_probabilities,
            validation_targets,
            len(class_names),
        )
        validation_macro = float(np.mean(validation_recalls))
        validation_failure_macro = float(np.mean(validation_recalls[1:]))

        epoch_text = (
            f"Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {format_percentage(train_accuracy)} | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Threshold: {threshold:.3f} | "
            f"{format_recall_summary(validation_recalls, class_names)} | "
            f"Macro-Recall: {format_percentage(validation_macro)} | "
            f"Failure Macro-Recall: {format_percentage(validation_failure_macro)}"
        )
        print(epoch_text)
        print()
        epoch_history.append(epoch_text)

        improved = (
            validation_macro > best_macro_recall + 1e-8
            or (
                abs(validation_macro - best_macro_recall) <= 1e-8
                and validation_failure_macro > best_failure_macro
            )
        )

        if improved:
            best_macro_recall = validation_macro
            best_failure_macro = validation_failure_macro
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": threshold,
                    "epoch": epoch,
                    "validation_macro_recall": validation_macro,
                    "validation_failure_macro_recall": validation_failure_macro,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping after {EARLY_STOPPING_PATIENCE} epochs "
                "without validation improvement."
            )
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_threshold = float(checkpoint["threshold"])

    final_test_loss, test_probabilities, test_targets = evaluate_probabilities(
        model, test_loader, criterion, device, "Final Test"
    )

    raw_predictions = test_probabilities.argmax(axis=1)
    predictions = predict_with_failure_threshold(
        test_probabilities, selected_threshold
    )
    raw_recalls = class_recalls(test_targets, raw_predictions, len(class_names))
    final_recalls = class_recalls(test_targets, predictions, len(class_names))

    final_accuracy = float((predictions == test_targets).mean())
    final_macro_recall = float(np.mean(final_recalls))
    final_failure_macro = float(np.mean(final_recalls[1:]))
    raw_macro_recall = float(np.mean(raw_recalls))

    lead_metrics = lead_time_metrics(test_loader.dataset, predictions)
    event_recall, event_count = event_level_recall(test_loader.dataset, predictions)

    print("\nFINAL TEST RESULTS")
    print("=" * 92)
    print(f"Model: {model_name}")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Validation-selected threshold: {selected_threshold:.3f}")
    print(f"Test loss: {final_test_loss:.4f}")
    print(f"Raw argmax macro recall: {format_percentage(raw_macro_recall)}")
    print(f"Final accuracy: {format_percentage(final_accuracy)}")
    print(format_recall_summary(final_recalls, class_names))
    print(f"Test Macro-Recall: {format_percentage(final_macro_recall)}")
    print(
        "Failure-only Macro-Recall: "
        f"{format_percentage(final_failure_macro)}"
    )

    print("\nLEAD-TIME RECALL")
    for bin_name, (component_recall, failure_recall, count) in lead_metrics.items():
        print(
            f"{bin_name}: component={format_percentage(component_recall)} | "
            f"any-failure={format_percentage(failure_recall)} | n={count}"
        )
    print(
        f"Event-level component recall: {format_percentage(event_recall)} "
        f"across {event_count} failure events"
    )

    parameter_count = count_parameters(model)
    save_classification_record(
        output_path=record_path,
        model_name=model_name,
        model_config=model_config,
        loss_name=LOSS_NAME,
        focal_gamma=FOCAL_GAMMA,
        class_weight_power=0.0,
        class_counts=class_counts.int().tolist(),
        class_weights=class_weights.detach().cpu().tolist(),
        parameter_count=parameter_count,
        best_epoch=best_epoch,
        best_macro_recall=best_macro_recall,
        final_test_loss=final_test_loss,
        final_recalls=final_recalls,
        class_names=class_names,
        targets=test_targets.tolist(),
        predictions=predictions.tolist(),
        epoch_history=epoch_history,
    )

    with record_path.open("a", encoding="utf-8") as file:
        file.write("\n\nADDITIONAL FAILURE-AWARE EVALUATION\n")
        file.write("=" * 72 + "\n")
        file.write(f"Validation-selected threshold: {selected_threshold:.6f}\n")
        file.write(f"Raw argmax macro recall: {raw_macro_recall:.6f}\n")
        file.write(f"Final accuracy: {final_accuracy:.6f}\n")
        file.write(f"Failure-only macro recall: {final_failure_macro:.6f}\n")
        file.write(f"Event-level recall: {event_recall:.6f}\n")
        file.write(f"Event count: {event_count}\n")
        for name, values in lead_metrics.items():
            file.write(
                f"{name}: component_recall={values[0]:.6f}, "
                f"failure_detection_recall={values[1]:.6f}, n={values[2]}\n"
            )

    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Classification report saved to: {record_path}")


if __name__ == "__main__":
    main()

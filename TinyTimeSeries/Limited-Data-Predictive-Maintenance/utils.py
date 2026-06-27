"""Utilities for imbalanced classification training and reporting."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_percentage(value: float) -> str:
    """Format a ratio as a percentage with two decimal places."""
    return f"{value * 100.0:.2f}%"


class RecallMeter:
    """Track a confusion matrix and compute per-class recall."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.matrix = torch.zeros(
            num_classes,
            num_classes,
            dtype=torch.long,
        )

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Update the confusion matrix using one batch."""
        predictions = predictions.detach().view(-1).cpu()
        targets = targets.detach().view(-1).cpu()

        indices = targets * self.num_classes + predictions
        batch_matrix = torch.bincount(
            indices,
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

        self.matrix += batch_matrix

    def recalls(self) -> list[float]:
        """Return recall for every class."""
        true_positive = self.matrix.diag().float()
        support = self.matrix.sum(dim=1).float()

        recall = torch.where(
            support > 0,
            true_positive / support,
            torch.zeros_like(true_positive),
        )

        return recall.tolist()


class WeightedFocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights.

    Easy and correctly classified samples receive less influence through
    (1 - p_t) ** gamma. Class weights increase the cost of minority errors.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()

        if gamma < 0:
            raise ValueError("gamma must be non-negative.")

        self.gamma = float(gamma)

        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer(
                "class_weights",
                class_weights.detach().float(),
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()

        target_log_probabilities = log_probabilities.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        target_probabilities = probabilities.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        focal_factor = (
            1.0 - target_probabilities
        ).pow(self.gamma)

        losses = -focal_factor * target_log_probabilities

        if self.class_weights is not None:
            losses = losses * self.class_weights[targets]

        return losses.mean()


def count_training_labels(
    loader: torch.utils.data.DataLoader,
    num_classes: int,
) -> torch.Tensor:
    """Count labels from the training DataLoader."""
    counts = torch.zeros(
        num_classes,
        dtype=torch.long,
    )

    for _, labels in loader:
        counts += torch.bincount(
            labels.view(-1).cpu(),
            minlength=num_classes,
        )

    return counts


def calculate_class_weights(
    class_counts: torch.Tensor,
    power: float = 0.5,
    maximum_weight: float = 10.0,
) -> torch.Tensor:
    """Calculate normalized inverse-frequency class weights.

    power=0.5 applies square-root inverse frequency, which is less aggressive
    than full inverse-frequency weighting and is usually more stable.
    """
    if power < 0:
        raise ValueError("power must be non-negative.")

    counts = class_counts.float()

    if torch.any(counts <= 0):
        missing = torch.where(counts <= 0)[0].tolist()
        raise ValueError(
            f"Training data contains no samples for classes: {missing}"
        )

    total = counts.sum()
    num_classes = counts.numel()

    weights = (
        total / (num_classes * counts)
    ).pow(power)

    weights = weights / weights.mean()
    weights = weights.clamp(max=maximum_weight)

    return weights.float()


def build_criterion(
    loss_name: str,
    class_weights: torch.Tensor,
    focal_gamma: float,
) -> nn.Module:
    """Build a weighted cross-entropy or weighted focal loss."""
    normalized_name = loss_name.strip().lower()

    if normalized_name == "weighted_ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
        )

    if normalized_name == "focal":
        return WeightedFocalLoss(
            class_weights=class_weights,
            gamma=focal_gamma,
        )

    raise ValueError(
        "Unknown loss strategy. "
        "Available values: 'weighted_ce', 'focal'."
    )


def format_recall_progress(
    recalls: Sequence[float],
    class_names: Sequence[str],
) -> str:
    """Format class recalls for a tqdm progress line."""
    return " | ".join(
        f"{name} {format_percentage(recall)}"
        for name, recall in zip(class_names, recalls)
    )


def format_recall_summary(
    recalls: Sequence[float],
    class_names: Sequence[str],
) -> str:
    """Format class recalls for console and record output."""
    return " | ".join(
        f"Recall-{name}: {format_percentage(recall)}"
        for name, recall in zip(class_names, recalls)
    )


def build_percentage_classification_report(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
) -> str:
    """Build a classification report with percentage metrics."""
    labels = list(range(len(class_names)))

    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    lines = [
        (
            f"{'Class':<14}"
            f"{'Precision':>12}"
            f"{'Recall':>12}"
            f"{'F1-score':>12}"
            f"{'Support':>12}"
        ),
        "-" * 62,
    ]

    for class_name in class_names:
        metrics = report[class_name]
        lines.append(
            f"{class_name:<14}"
            f"{format_percentage(float(metrics['precision'])):>12}"
            f"{format_percentage(float(metrics['recall'])):>12}"
            f"{format_percentage(float(metrics['f1-score'])):>12}"
            f"{int(metrics['support']):>12d}"
        )

    lines.append("-" * 62)

    accuracy = float(report["accuracy"])
    total_support = len(targets)

    lines.append(
        f"{'accuracy':<14}"
        f"{'':>12}"
        f"{'':>12}"
        f"{format_percentage(accuracy):>12}"
        f"{total_support:>12d}"
    )

    for average_name in ("macro avg", "weighted avg"):
        metrics = report[average_name]
        lines.append(
            f"{average_name:<14}"
            f"{format_percentage(float(metrics['precision'])):>12}"
            f"{format_percentage(float(metrics['recall'])):>12}"
            f"{format_percentage(float(metrics['f1-score'])):>12}"
            f"{int(metrics['support']):>12d}"
        )

    return "\n".join(lines)


def save_classification_record(
    output_path: Path,
    model_name: str,
    model_config: dict[str, Any],
    loss_name: str,
    focal_gamma: float,
    class_weight_power: float,
    class_counts: Sequence[int],
    class_weights: Sequence[float],
    parameter_count: int,
    best_epoch: int,
    best_macro_recall: float,
    final_test_loss: float,
    final_recalls: Sequence[float],
    class_names: Sequence[str],
    targets: Sequence[int],
    predictions: Sequence[int],
    epoch_history: Sequence[str],
) -> None:
    """Save the experiment result to Record/modelname.txt."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = build_percentage_classification_report(
        targets=targets,
        predictions=predictions,
        class_names=class_names,
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(class_names))),
    )

    final_accuracy = (
        float(np.mean(np.asarray(targets) == np.asarray(predictions)))
        if targets
        else 0.0
    )
    final_macro_recall = (
        float(sum(final_recalls) / len(final_recalls))
        if final_recalls
        else 0.0
    )

    distribution_lines = [
        (
            f"{class_name}: count={int(count)}, "
            f"weight={float(weight):.4f}"
        )
        for class_name, count, weight in zip(
            class_names,
            class_counts,
            class_weights,
        )
    ]

    content = [
        f"Model: {model_name}",
        f"Model configuration: {model_config}",
        f"Loss strategy: {loss_name}",
        f"Focal gamma: {focal_gamma}",
        f"Class-weight power: {class_weight_power}",
        f"Trainable parameters: {parameter_count:,}",
        f"Best epoch: {best_epoch}",
        f"Best Macro-Recall: {format_percentage(best_macro_recall)}",
        f"Final test loss: {final_test_loss:.4f}",
        f"Final accuracy: {format_percentage(final_accuracy)}",
        f"Final Macro-Recall: {format_percentage(final_macro_recall)}",
        format_recall_summary(final_recalls, class_names),
        "",
        "Training class distribution and weights",
        "=" * 100,
        *distribution_lines,
        "",
        "Epoch history",
        "=" * 100,
        *epoch_history,
        "",
        "Classification report",
        "=" * 100,
        report,
        "",
        "Confusion matrix",
        "=" * 100,
        np.array2string(matrix),
        "",
    ]

    output_path.write_text(
        "\n".join(content),
        encoding="utf-8",
    )

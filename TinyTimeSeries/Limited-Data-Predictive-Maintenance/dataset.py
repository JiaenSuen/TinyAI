"""Leakage-resistant dataset for Azure predictive-maintenance experiments.

Main changes from the earlier version
-------------------------------------
1. Device-disjoint train/validation/test splitting remains mandatory.
2. All engineered features are causal and fitted with train-only statistics.
3. Optional errors, maintenance, and machine metadata are used when present.
4. Windows immediately after a failure are removed to avoid recovery-state
   samples being incorrectly treated as ordinary normal operation.
5. Positive windows are balanced by failure event and lead-time bin.
6. Near-failure hard negatives are capped instead of being retained without a
   limit, because excessive 72--240 h negatives can suppress failure recall.
7. Validation and test retain their natural window distributions.

Expected files
--------------
Required:
    data/PdM_telemetry.csv
    data/PdM_failures.csv

Optional (automatically used when available):
    data/PdM_errors.csv
    data/PdM_maint.csv
    data/PdM_machines.csv
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Data configuration
# =============================================================================

DATA_DIR = Path(__file__).resolve().parent / "data"

TELEMETRY_FILENAME = "PdM_telemetry.csv"
FAILURE_FILENAME = "PdM_failures.csv"
ERROR_FILENAME = "PdM_errors.csv"
MAINTENANCE_FILENAME = "PdM_maint.csv"
MACHINE_FILENAME = "PdM_machines.csv"

BASE_TELEMETRY_COLUMNS = ["volt", "rotate", "pressure", "vibration"]

LABEL_TO_ID = {
    "none": 0,
    "comp1": 1,
    "comp2": 2,
    "comp3": 3,
    "comp4": 4,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

SEQUENCE_LENGTH = 168
PREDICTION_HORIZON_HOURS = 72

# A denser stride is useful only for training. Event/bin caps below prevent
# dense windows from turning into pseudo-independent copies of one failure.
TRAIN_WINDOW_STRIDE = 12
EVALUATION_WINDOW_STRIDE = 24

TRAIN_DEVICE_RATIO = 0.70
VALIDATION_DEVICE_RATIO = 0.15
TEST_DEVICE_RATIO = 0.15
DEVICE_SPLIT_TRIALS = 1000

BATCH_SIZE = 32
NUM_WORKERS = 0
RANDOM_SEED = 42


# =============================================================================
# Causal feature engineering
# =============================================================================

TELEMETRY_ROLLING_WINDOWS = (6, 24, 72)
ERROR_ROLLING_WINDOWS = (24, 72)
MAINTENANCE_ROLLING_WINDOW = 720
FEATURE_CLIP_VALUE = 12.0


# =============================================================================
# Training-only sampling
# =============================================================================

# The old ratio of 5 normals per positive left the training set strongly
# dominated by none. This remains imbalanced, but is moderate rather than
# overwhelming.
NORMAL_TO_POSITIVE_RATIO = 2.5
MINIMUM_TRAIN_NORMALS = 500

HARD_NEGATIVE_HORIZON_HOURS = 240
HARD_NEGATIVE_DEVIATION_QUANTILE = 0.85
HARD_NEGATIVE_TARGET_FRACTION = 0.40

HARD_NEGATIVE_MIN_SPACING_HOURS = 24
EASY_NORMAL_MIN_SPACING_HOURS = 72

# At 12 h stride this keeps at most two examples in each 24 h lead-time bin
# for one physical failure event.
MAX_POSITIVE_WINDOWS_PER_EVENT_BIN = 2
LEAD_TIME_BIN_HOURS = 24

# Recovery/transient operation immediately after a failure is not a clean
# normal sample and often creates contradictory labels.
POST_FAILURE_EXCLUSION_HOURS = 24

CONDITION_LOW_THRESHOLD = -0.5
CONDITION_HIGH_THRESHOLD = 0.5


@dataclass(frozen=True)
class OptionalTables:
    errors: Optional[pd.DataFrame]
    maintenance: Optional[pd.DataFrame]
    machines: Optional[pd.DataFrame]


@dataclass(frozen=True)
class Window:
    """Metadata for one fixed-length telemetry sequence."""

    machine_id: int
    session_id: int
    start: int
    end: int
    label_id: int
    hours_to_failure: float
    hours_since_failure: float
    failure_event_id: int
    lead_time_bin: int
    deviation_score: float
    condition_key: Tuple[int, ...]


@dataclass(frozen=True)
class DatasetSplits:
    train: "AzurePredictiveMaintenanceDataset"
    validation: "AzurePredictiveMaintenanceDataset"
    test: "AzurePredictiveMaintenanceDataset"


@dataclass(frozen=True)
class DataLoaderSplits:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def _find_file(data_dir: Path, filename: str, required: bool = True) -> Optional[Path]:
    """Find a file without depending on filename capitalization."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory was not found: {data_dir}")

    files = {
        path.name.lower(): path
        for path in data_dir.iterdir()
        if path.is_file()
    }
    path = files.get(filename.lower())

    if path is None and required:
        raise FileNotFoundError(
            f"Required file '{filename}' was not found in '{data_dir}'."
        )
    return path


def _parse_datetime_and_machine(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    frame["machineID"] = pd.to_numeric(
        frame["machineID"], errors="raise"
    ).astype(np.int64)
    return frame


def load_tables(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load required telemetry and failure tables."""
    telemetry_path = _find_file(data_dir, TELEMETRY_FILENAME, required=True)
    failure_path = _find_file(data_dir, FAILURE_FILENAME, required=True)
    assert telemetry_path is not None and failure_path is not None

    telemetry = pd.read_csv(telemetry_path)
    failures = pd.read_csv(failure_path)

    required_telemetry = {"datetime", "machineID", *BASE_TELEMETRY_COLUMNS}
    required_failures = {"datetime", "machineID", "failure"}

    missing = required_telemetry - set(telemetry.columns)
    if missing:
        raise ValueError(f"Telemetry columns are missing: {sorted(missing)}")

    missing = required_failures - set(failures.columns)
    if missing:
        raise ValueError(f"Failure columns are missing: {sorted(missing)}")

    telemetry = _parse_datetime_and_machine(telemetry)
    failures = _parse_datetime_and_machine(failures)

    telemetry[BASE_TELEMETRY_COLUMNS] = telemetry[
        BASE_TELEMETRY_COLUMNS
    ].apply(pd.to_numeric, errors="raise")

    failures["failure"] = (
        failures["failure"].astype(str).str.strip().str.lower()
    )
    unknown = sorted(set(failures["failure"]) - set(LABEL_TO_ID))
    if unknown:
        raise ValueError(f"Unknown failure labels were found: {unknown}")

    telemetry = telemetry.sort_values(
        ["machineID", "datetime"]
    ).reset_index(drop=True)
    failures = failures.sort_values(
        ["machineID", "datetime"]
    ).reset_index(drop=True)
    return telemetry, failures


def load_optional_tables(data_dir: Path = DATA_DIR) -> OptionalTables:
    """Load optional context tables when they exist."""

    def read_optional(filename: str) -> Optional[pd.DataFrame]:
        path = _find_file(data_dir, filename, required=False)
        return None if path is None else pd.read_csv(path)

    errors = read_optional(ERROR_FILENAME)
    maintenance = read_optional(MAINTENANCE_FILENAME)
    machines = read_optional(MACHINE_FILENAME)

    if errors is not None:
        required = {"datetime", "machineID", "errorID"}
        missing = required - set(errors.columns)
        if missing:
            raise ValueError(f"Error columns are missing: {sorted(missing)}")
        errors = _parse_datetime_and_machine(errors)
        errors["errorID"] = errors["errorID"].astype(str).str.strip().str.lower()

    if maintenance is not None:
        required = {"datetime", "machineID", "comp"}
        missing = required - set(maintenance.columns)
        if missing:
            raise ValueError(
                f"Maintenance columns are missing: {sorted(missing)}"
            )
        maintenance = _parse_datetime_and_machine(maintenance)
        maintenance["comp"] = (
            maintenance["comp"].astype(str).str.strip().str.lower()
        )

    if machines is not None:
        required = {"machineID"}
        missing = required - set(machines.columns)
        if missing:
            raise ValueError(f"Machine columns are missing: {sorted(missing)}")
        machines = machines.copy()
        machines["machineID"] = pd.to_numeric(
            machines["machineID"], errors="raise"
        ).astype(np.int64)

    return OptionalTables(errors, maintenance, machines)


def _rolling_transform(
    frame: pd.DataFrame,
    column: str,
    window: int,
    operation: str,
) -> pd.Series:
    grouped = frame.groupby("machineID", sort=False)[column]
    rolling = grouped.rolling(window=window, min_periods=1)

    if operation == "mean":
        result = rolling.mean()
    elif operation == "std":
        result = rolling.std(ddof=0)
    elif operation == "sum":
        result = rolling.sum()
    else:
        raise ValueError(f"Unknown rolling operation: {operation}")

    return result.reset_index(level=0, drop=True).reindex(frame.index)


def build_causal_feature_frame(
    telemetry: pd.DataFrame,
    optional_tables: OptionalTables,
) -> Tuple[pd.DataFrame, list[str]]:
    """Construct causal sequence features without future information."""
    frame = telemetry[["datetime", "machineID", *BASE_TELEMETRY_COLUMNS]].copy()
    frame = frame.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    feature_columns: list[str] = list(BASE_TELEMETRY_COLUMNS)

    # Telemetry derivatives and rolling statistics. All rolling windows are
    # backward-looking because pandas rolling aligns to the current row.
    for column in BASE_TELEMETRY_COLUMNS:
        grouped = frame.groupby("machineID", sort=False)[column]

        delta_name = f"{column}_delta1"
        frame[delta_name] = grouped.diff().fillna(0.0)
        feature_columns.append(delta_name)

        for window in TELEMETRY_ROLLING_WINDOWS:
            mean_name = f"{column}_mean_{window}h"
            frame[mean_name] = _rolling_transform(
                frame, column, window, "mean"
            )
            feature_columns.append(mean_name)

        std_name = f"{column}_std_24h"
        frame[std_name] = _rolling_transform(frame, column, 24, "std")
        feature_columns.append(std_name)

        deviation_name = f"{column}_deviation_24h"
        frame[deviation_name] = frame[column] - frame[f"{column}_mean_24h"]
        feature_columns.append(deviation_name)

    errors = optional_tables.errors
    if errors is not None and not errors.empty:
        error_dummies = pd.get_dummies(
            errors["errorID"], prefix="error", dtype=np.float32
        )
        error_hourly = pd.concat(
            [errors[["machineID", "datetime"]], error_dummies], axis=1
        ).groupby(["machineID", "datetime"], as_index=False).sum()

        error_columns = [
            column
            for column in error_hourly.columns
            if column not in {"machineID", "datetime"}
        ]
        frame = frame.merge(
            error_hourly, on=["machineID", "datetime"], how="left"
        )
        frame[error_columns] = frame[error_columns].fillna(0.0)

        for column in error_columns:
            feature_columns.append(column)
            for window in ERROR_ROLLING_WINDOWS:
                name = f"{column}_count_{window}h"
                frame[name] = _rolling_transform(frame, column, window, "sum")
                feature_columns.append(name)

    maintenance = optional_tables.maintenance
    if maintenance is not None and not maintenance.empty:
        maint_dummies = pd.get_dummies(
            maintenance["comp"], prefix="maint", dtype=np.float32
        )
        maint_hourly = pd.concat(
            [maintenance[["machineID", "datetime"]], maint_dummies], axis=1
        ).groupby(["machineID", "datetime"], as_index=False).sum()

        maint_columns = [
            column
            for column in maint_hourly.columns
            if column not in {"machineID", "datetime"}
        ]
        frame = frame.merge(
            maint_hourly, on=["machineID", "datetime"], how="left"
        )
        frame[maint_columns] = frame[maint_columns].fillna(0.0)

        for column in maint_columns:
            feature_columns.append(column)
            count_name = f"{column}_count_{MAINTENANCE_ROLLING_WINDOW}h"
            frame[count_name] = _rolling_transform(
                frame, column, MAINTENANCE_ROLLING_WINDOW, "sum"
            )
            feature_columns.append(count_name)

    machines = optional_tables.machines
    if machines is not None and not machines.empty:
        machine_features = machines[["machineID"]].copy()

        if "age" in machines.columns:
            machine_features["machine_age"] = pd.to_numeric(
                machines["age"], errors="raise"
            ).astype(np.float32)

        if "model" in machines.columns:
            model_dummies = pd.get_dummies(
                machines["model"].astype(str).str.strip().str.lower(),
                prefix="machine_model",
                dtype=np.float32,
            )
            machine_features = pd.concat(
                [machine_features, model_dummies], axis=1
            )

        extra_columns = [
            column for column in machine_features.columns if column != "machineID"
        ]
        if extra_columns:
            frame = frame.merge(machine_features, on="machineID", how="left")
            frame[extra_columns] = frame[extra_columns].fillna(0.0)
            feature_columns.extend(extra_columns)

    frame[feature_columns] = frame[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    return frame, feature_columns


def _score_device_split(
    split_device_ids: Sequence[np.ndarray],
    failure_matrix: np.ndarray,
    machine_to_row: Dict[int, int],
    target_ratios: np.ndarray,
) -> float:
    class_totals = failure_matrix.sum(axis=0)
    score = 0.0

    for split_index, machine_ids in enumerate(split_device_ids):
        row_indices = [machine_to_row[int(machine_id)] for machine_id in machine_ids]
        split_counts = failure_matrix[row_indices].sum(axis=0)
        expected = class_totals * target_ratios[split_index]
        score += float(
            np.sum(np.abs(split_counts - expected) / np.maximum(class_totals, 1.0))
        )

        missing_mask = (
            (class_totals >= len(split_device_ids)) & (split_counts == 0)
        )
        score += float(missing_mask.sum()) * 10.0

    return score


def split_devices(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> Tuple[set[int], set[int], set[int]]:
    """Create deterministic device-disjoint splits with event stratification."""
    machine_ids = np.array(
        sorted(telemetry["machineID"].astype(int).unique().tolist()),
        dtype=np.int64,
    )
    if len(machine_ids) < 3:
        raise ValueError("At least three devices are required for three splits.")

    ratios = np.array(
        [TRAIN_DEVICE_RATIO, VALIDATION_DEVICE_RATIO, TEST_DEVICE_RATIO],
        dtype=np.float64,
    )
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Device split ratios must sum to 1.0.")

    number_of_devices = len(machine_ids)
    train_count = max(1, int(round(number_of_devices * TRAIN_DEVICE_RATIO)))
    validation_count = max(
        1, int(round(number_of_devices * VALIDATION_DEVICE_RATIO))
    )
    if train_count + validation_count >= number_of_devices:
        validation_count = max(1, number_of_devices - train_count - 1)

    component_names = ["comp1", "comp2", "comp3", "comp4"]
    event_table = pd.crosstab(
        failures["machineID"], failures["failure"]
    ).reindex(index=machine_ids, columns=component_names, fill_value=0)

    failure_matrix = event_table.to_numpy(dtype=np.float64)
    machine_to_row = {
        int(machine_id): index for index, machine_id in enumerate(machine_ids)
    }
    rng = np.random.default_rng(random_seed)

    best_score = float("inf")
    best_split: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    for _ in range(DEVICE_SPLIT_TRIALS):
        permutation = rng.permutation(machine_ids)
        candidate = (
            permutation[:train_count],
            permutation[train_count : train_count + validation_count],
            permutation[train_count + validation_count :],
        )
        score = _score_device_split(
            candidate, failure_matrix, machine_to_row, ratios
        )
        if score < best_score:
            best_score = score
            best_split = candidate

    if best_split is None:
        raise RuntimeError("Could not create a valid device split.")

    return tuple(set(part.astype(int).tolist()) for part in best_split)  # type: ignore[return-value]


def _build_failure_lookup(
    failures: pd.DataFrame,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    prepared = failures.copy()
    prepared["label_id"] = prepared["failure"].map(LABEL_TO_ID).astype(np.int64)

    lookup: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for machine_id, group in prepared.groupby("machineID", sort=False):
        lookup[int(machine_id)] = (
            group["datetime"].to_numpy(dtype="datetime64[ns]"),
            group["label_id"].to_numpy(dtype=np.int64),
        )
    return lookup


def _failure_information(
    machine_id: int,
    timestamp: np.datetime64,
    failure_lookup: Dict[int, Tuple[np.ndarray, np.ndarray]],
    prediction_horizon_hours: int,
) -> Tuple[int, float, float, int]:
    """Return label, hours to next, hours since previous, and event id."""
    if machine_id not in failure_lookup:
        return LABEL_TO_ID["none"], float("inf"), float("inf"), -1

    failure_times, failure_labels = failure_lookup[machine_id]
    next_index = int(np.searchsorted(failure_times, timestamp, side="right"))

    if next_index > 0:
        previous_time = failure_times[next_index - 1]
        hours_since = float((timestamp - previous_time) / np.timedelta64(1, "h"))
    else:
        hours_since = float("inf")

    if next_index >= len(failure_times):
        return LABEL_TO_ID["none"], float("inf"), hours_since, -1

    next_time = failure_times[next_index]
    hours_to = float((next_time - timestamp) / np.timedelta64(1, "h"))

    if 0.0 < hours_to <= prediction_horizon_hours:
        return int(failure_labels[next_index]), hours_to, hours_since, next_index

    return LABEL_TO_ID["none"], hours_to, hours_since, -1


def _continuous_session_ranges(timestamps: np.ndarray) -> list[Tuple[int, int]]:
    if len(timestamps) == 0:
        return []

    breaks = np.where(np.diff(timestamps) > np.timedelta64(1, "h"))[0] + 1
    starts = np.concatenate([np.array([0]), breaks])
    ends = np.concatenate([breaks, np.array([len(timestamps)])])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _condition_key(window_features: np.ndarray) -> Tuple[int, ...]:
    # Only the first four standardized raw telemetry dimensions define the
    # operating-condition bin. Optional sparse features should not dominate it.
    means = window_features[:, : len(BASE_TELEMETRY_COLUMNS)].mean(axis=0)
    bins = np.where(
        means < CONDITION_LOW_THRESHOLD,
        0,
        np.where(means > CONDITION_HIGH_THRESHOLD, 2, 1),
    )
    return tuple(int(value) for value in bins)


def _thin_by_time(
    windows: Sequence[Window], minimum_spacing_hours: int
) -> list[Window]:
    grouped: Dict[Tuple[int, int], list[Window]] = defaultdict(list)
    for window in windows:
        grouped[(window.machine_id, window.session_id)].append(window)

    kept: list[Window] = []
    for group in grouped.values():
        last_end: Optional[int] = None
        for window in sorted(group, key=lambda item: item.end):
            if last_end is None or window.end - last_end >= minimum_spacing_hours:
                kept.append(window)
                last_end = window.end
    return kept


def _round_robin_diverse_sample(
    windows: Sequence[Window], sample_count: int, random_seed: int
) -> list[Window]:
    if sample_count <= 0:
        return []

    rng = np.random.default_rng(random_seed)
    grouped: Dict[Tuple[int, Tuple[int, ...]], list[Window]] = defaultdict(list)
    for window in windows:
        grouped[(window.machine_id, window.condition_key)].append(window)

    queues: list[deque[Window]] = []
    for key in sorted(grouped, key=str):
        group = grouped[key]
        order = rng.permutation(len(group))
        queues.append(deque(group[index] for index in order))

    selected: list[Window] = []
    while queues and len(selected) < sample_count:
        remaining: list[deque[Window]] = []
        for queue in queues:
            if queue and len(selected) < sample_count:
                selected.append(queue.popleft())
            if queue:
                remaining.append(queue)
        queues = remaining
    return selected


def _cap_positive_windows_by_event(
    windows: Sequence[Window],
) -> list[Window]:
    """Prevent one failure event from dominating via overlapping windows."""
    grouped: Dict[Tuple[int, int, int, int], list[Window]] = defaultdict(list)
    for window in windows:
        grouped[
            (
                window.machine_id,
                window.failure_event_id,
                window.label_id,
                window.lead_time_bin,
            )
        ].append(window)

    selected: list[Window] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda item: item.hours_to_failure)
        if len(group) <= MAX_POSITIVE_WINDOWS_PER_EVENT_BIN:
            selected.extend(group)
            continue

        # Deterministic evenly spaced representatives within the lead-time bin.
        indices = np.linspace(
            0,
            len(group) - 1,
            num=MAX_POSITIVE_WINDOWS_PER_EVENT_BIN,
            dtype=int,
        )
        selected.extend(group[int(index)] for index in indices)

    return selected


def sample_training_windows(
    candidate_windows: Sequence[Window], random_seed: int = RANDOM_SEED
) -> Tuple[list[Window], Dict[str, int]]:
    """Event-aware positive sampling and moderate normal reduction."""
    raw_positives = [
        window for window in candidate_windows if window.label_id != LABEL_TO_ID["none"]
    ]
    positive_windows = _cap_positive_windows_by_event(raw_positives)

    normal_windows = [
        window for window in candidate_windows if window.label_id == LABEL_TO_ID["none"]
    ]

    if normal_windows:
        deviation_threshold = float(
            np.quantile(
                [window.deviation_score for window in normal_windows],
                HARD_NEGATIVE_DEVIATION_QUANTILE,
            )
        )
    else:
        deviation_threshold = float("inf")

    hard_negatives = [
        window
        for window in normal_windows
        if (
            window.hours_to_failure <= HARD_NEGATIVE_HORIZON_HOURS
            or window.deviation_score >= deviation_threshold
        )
    ]
    hard_keys = {
        (window.machine_id, window.session_id, window.start, window.end)
        for window in hard_negatives
    }
    easy_normals = [
        window
        for window in normal_windows
        if (window.machine_id, window.session_id, window.start, window.end)
        not in hard_keys
    ]

    hard_negatives = _thin_by_time(
        hard_negatives, HARD_NEGATIVE_MIN_SPACING_HOURS
    )
    easy_normals = _thin_by_time(easy_normals, EASY_NORMAL_MIN_SPACING_HOURS)

    normal_target = min(
        len(normal_windows),
        max(
            MINIMUM_TRAIN_NORMALS,
            int(round(len(positive_windows) * NORMAL_TO_POSITIVE_RATIO)),
        ),
    )

    hard_target = min(
        len(hard_negatives),
        int(round(normal_target * HARD_NEGATIVE_TARGET_FRACTION)),
    )
    selected_hard = _round_robin_diverse_sample(
        hard_negatives, hard_target, random_seed + 1
    )
    selected_easy = _round_robin_diverse_sample(
        easy_normals, normal_target - len(selected_hard), random_seed + 2
    )

    selected = positive_windows + selected_hard + selected_easy
    selected = sorted(
        selected,
        key=lambda item: (item.machine_id, item.session_id, item.start),
    )

    summary = {
        "candidate_total": len(candidate_windows),
        "raw_positive": len(raw_positives),
        "event_balanced_positive": len(positive_windows),
        "candidate_normal": len(normal_windows),
        "selected_hard_negative": len(selected_hard),
        "selected_easy_normal": len(selected_easy),
        "selected_total": len(selected),
    }
    return selected, summary


class AzurePredictiveMaintenanceDataset(Dataset):
    """Fixed-length causal feature sequences for component classification."""

    def __init__(
        self,
        feature_frame: pd.DataFrame,
        failures: pd.DataFrame,
        feature_columns: Sequence[str],
        split_name: str,
        sequence_length: int = SEQUENCE_LENGTH,
        prediction_horizon_hours: int = PREDICTION_HORIZON_HOURS,
        stride: int = EVALUATION_WINDOW_STRIDE,
        feature_mean: Optional[Sequence[float]] = None,
        feature_std: Optional[Sequence[float]] = None,
        sample_training_data: bool = False,
    ) -> None:
        super().__init__()

        if feature_frame.empty:
            raise ValueError("Feature frame must not be empty.")
        if sequence_length <= 0 or prediction_horizon_hours <= 0 or stride <= 0:
            raise ValueError("Sequence length, horizon, and stride must be positive.")

        self.split_name = split_name
        self.sequence_length = int(sequence_length)
        self.prediction_horizon_hours = int(prediction_horizon_hours)
        self.stride = int(stride)
        self.feature_columns = list(feature_columns)

        raw_features = feature_frame[self.feature_columns].to_numpy(dtype=np.float32)
        mean = (
            raw_features.mean(axis=0)
            if feature_mean is None
            else np.asarray(feature_mean, dtype=np.float32)
        )
        std = (
            raw_features.std(axis=0)
            if feature_std is None
            else np.asarray(feature_std, dtype=np.float32)
        )

        expected_shape = (len(self.feature_columns),)
        if mean.shape != expected_shape or std.shape != expected_shape:
            raise ValueError(
                f"feature_mean and feature_std must have shape {expected_shape}."
            )
        std = np.where(std < 1e-8, 1.0, std)

        self.feature_mean = mean.astype(np.float32)
        self.feature_std = std.astype(np.float32)
        failure_lookup = _build_failure_lookup(failures)

        self.frames: Dict[int, pd.DataFrame] = {}
        self.features: Dict[int, np.ndarray] = {}
        candidate_windows: list[Window] = []

        for machine_id, machine_frame in feature_frame.groupby(
            "machineID", sort=True
        ):
            machine_id = int(machine_id)
            machine_frame = machine_frame.sort_values("datetime").reset_index(drop=True)
            timestamps = machine_frame["datetime"].to_numpy(dtype="datetime64[ns]")

            features = machine_frame[self.feature_columns].to_numpy(dtype=np.float32)
            features = (features - self.feature_mean) / self.feature_std
            features = np.clip(
                features, -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE
            ).astype(np.float32)

            self.frames[machine_id] = machine_frame
            self.features[machine_id] = features

            for session_id, (session_start, session_end) in enumerate(
                _continuous_session_ranges(timestamps)
            ):
                if session_end - session_start < self.sequence_length:
                    continue

                last_start = session_end - self.sequence_length
                for start in range(session_start, last_start + 1, self.stride):
                    end = start + self.sequence_length
                    expected_duration = np.timedelta64(self.sequence_length - 1, "h")
                    if timestamps[end - 1] - timestamps[start] != expected_duration:
                        continue

                    (
                        label_id,
                        hours_to_failure,
                        hours_since_failure,
                        event_id,
                    ) = _failure_information(
                        machine_id,
                        timestamps[end - 1],
                        failure_lookup,
                        self.prediction_horizon_hours,
                    )

                    # Exclude recovery windows from every split. They are not
                    # clean normal operation and are not valid pre-failure data.
                    if hours_since_failure <= POST_FAILURE_EXCLUSION_HOURS:
                        continue

                    window_features = features[start:end]
                    lead_bin = (
                        int((hours_to_failure - 1e-6) // LEAD_TIME_BIN_HOURS)
                        if label_id != LABEL_TO_ID["none"]
                        else -1
                    )

                    candidate_windows.append(
                        Window(
                            machine_id=machine_id,
                            session_id=session_id,
                            start=start,
                            end=end,
                            label_id=label_id,
                            hours_to_failure=hours_to_failure,
                            hours_since_failure=hours_since_failure,
                            failure_event_id=event_id,
                            lead_time_bin=lead_bin,
                            deviation_score=float(
                                np.mean(
                                    np.abs(
                                        window_features[
                                            :, : len(BASE_TELEMETRY_COLUMNS)
                                        ]
                                    )
                                )
                            ),
                            condition_key=_condition_key(window_features),
                        )
                    )

        if not candidate_windows:
            raise ValueError(f"No valid windows were created for {split_name}.")

        if sample_training_data:
            self.windows, self.sampling_summary = sample_training_windows(
                candidate_windows
            )
        else:
            self.windows = candidate_windows
            self.sampling_summary = {
                "candidate_total": len(candidate_windows),
                "selected_total": len(candidate_windows),
            }

        if not self.windows:
            raise ValueError(f"No windows remain for {split_name}.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[index]
        feature_array = self.features[window.machine_id][window.start : window.end]
        return (
            torch.from_numpy(np.ascontiguousarray(feature_array)).float(),
            torch.tensor(window.label_id, dtype=torch.long),
        )

    def get_metadata(self, index: int) -> Dict[str, object]:
        window = self.windows[index]
        frame = self.frames[window.machine_id]
        return {
            "split": self.split_name,
            "machine_id": window.machine_id,
            "session_id": window.session_id,
            "start_time": frame.iloc[window.start]["datetime"],
            "end_time": frame.iloc[window.end - 1]["datetime"],
            "label_id": window.label_id,
            "label_name": ID_TO_LABEL[window.label_id],
            "hours_to_failure": window.hours_to_failure,
            "hours_since_failure": window.hours_since_failure,
            "failure_event_id": window.failure_event_id,
            "lead_time_bin": window.lead_time_bin,
            "condition_key": window.condition_key,
        }

    def class_counts(self) -> Dict[str, int]:
        counts = {name: 0 for name in LABEL_TO_ID}
        for window in self.windows:
            counts[ID_TO_LABEL[window.label_id]] += 1
        return counts

    def machine_ids(self) -> set[int]:
        return set(self.frames)


def create_experiment_datasets(data_dir: Path = DATA_DIR) -> DatasetSplits:
    """Create feature-rich, device-disjoint experiment datasets."""
    telemetry, failures = load_tables(data_dir)
    optional_tables = load_optional_tables(data_dir)
    feature_frame, feature_columns = build_causal_feature_frame(
        telemetry, optional_tables
    )

    train_ids, validation_ids, test_ids = split_devices(telemetry, failures)

    train_frame = feature_frame[feature_frame["machineID"].isin(train_ids)].copy()
    validation_frame = feature_frame[
        feature_frame["machineID"].isin(validation_ids)
    ].copy()
    test_frame = feature_frame[feature_frame["machineID"].isin(test_ids)].copy()

    train_dataset = AzurePredictiveMaintenanceDataset(
        feature_frame=train_frame,
        failures=failures,
        feature_columns=feature_columns,
        split_name="train",
        stride=TRAIN_WINDOW_STRIDE,
        sample_training_data=True,
    )
    validation_dataset = AzurePredictiveMaintenanceDataset(
        feature_frame=validation_frame,
        failures=failures,
        feature_columns=feature_columns,
        split_name="validation",
        stride=EVALUATION_WINDOW_STRIDE,
        feature_mean=train_dataset.feature_mean,
        feature_std=train_dataset.feature_std,
        sample_training_data=False,
    )
    test_dataset = AzurePredictiveMaintenanceDataset(
        feature_frame=test_frame,
        failures=failures,
        feature_columns=feature_columns,
        split_name="test",
        stride=EVALUATION_WINDOW_STRIDE,
        feature_mean=train_dataset.feature_mean,
        feature_std=train_dataset.feature_std,
        sample_training_data=False,
    )

    datasets = DatasetSplits(train_dataset, validation_dataset, test_dataset)
    _validate_split_isolation(datasets)
    return datasets


def _validate_split_isolation(datasets: DatasetSplits) -> None:
    train_ids = datasets.train.machine_ids()
    validation_ids = datasets.validation.machine_ids()
    test_ids = datasets.test.machine_ids()

    if train_ids & validation_ids:
        raise RuntimeError("Device leakage between train and validation.")
    if train_ids & test_ids:
        raise RuntimeError("Device leakage between train and test.")
    if validation_ids & test_ids:
        raise RuntimeError("Device leakage between validation and test.")


def create_experiment_dataloaders(
    data_dir: Path = DATA_DIR,
) -> DataLoaderSplits:
    datasets = create_experiment_datasets(data_dir)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": NUM_WORKERS > 0,
    }

    return DataLoaderSplits(
        train=DataLoader(
            datasets.train,
            shuffle=True,
            generator=generator,
            drop_last=False,
            **common,
        ),
        validation=DataLoader(
            datasets.validation, shuffle=False, drop_last=False, **common
        ),
        test=DataLoader(datasets.test, shuffle=False, drop_last=False, **common),
    )


# =============================================================================
# Backward-compatible two-way API
# =============================================================================


def create_datasets(
    data_dir: Path = DATA_DIR,
) -> Tuple[AzurePredictiveMaintenanceDataset, AzurePredictiveMaintenanceDataset]:
    datasets = create_experiment_datasets(data_dir)
    return datasets.train, datasets.test


def create_dataloaders(
    data_dir: Path = DATA_DIR,
) -> Tuple[DataLoader, DataLoader]:
    loaders = create_experiment_dataloaders(data_dir)
    return loaders.train, loaders.test


def _print_split(dataset: AzurePredictiveMaintenanceDataset) -> None:
    features, label = dataset[0]
    print(f"\n{dataset.split_name.upper()}")
    print("=" * 78)
    print(f"Devices: {len(dataset.machine_ids())}")
    print(f"Sequences: {len(dataset):,}")
    print(f"Features: {len(dataset.feature_columns)}")
    print(f"Class counts: {dataset.class_counts()}")
    print(f"Sampling: {dataset.sampling_summary}")
    print(f"Sample shape: {tuple(features.shape)}")
    print(f"Sample label: {label.item()} ({ID_TO_LABEL[label.item()]})")


def main() -> None:
    datasets = create_experiment_datasets()
    print("DEVICE-DISJOINT, CAUSAL FEATURE DATASET")
    print("=" * 78)
    _print_split(datasets.train)
    _print_split(datasets.validation)
    _print_split(datasets.test)

    loaders = create_experiment_dataloaders()
    for name, loader in (
        ("Train", loaders.train),
        ("Validation", loaders.validation),
        ("Test", loaders.test),
    ):
        features, labels = next(iter(loader))
        print(f"{name}: {tuple(features.shape)}, {tuple(labels.shape)}")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TempDataset(Dataset):
    """
    Time-series regression dataset.

    Each sample:
    - inputs: shape [lookback // step, feature_dim]
    - label: scalar
    - time: timestamp of the prediction target
    """

    def __init__(
        self,
        data: np.ndarray,
        timestamps,
        lookback: int,
        delay: int,
        step: int,
        target_idx: int,
        start_anchor: int,
        end_anchor: int,
    ):
        """
        data:         numpy array, shape [N, feature_dim]
        timestamps:   list / array, length N
        lookback:     number of past steps used as context
        delay:        forecasting horizon in steps
        step:         sampling interval inside the lookback window
        target_idx:    target column index in data
        start_anchor:  inclusive start index of the anchor i
        end_anchor:    exclusive end index of the anchor i
        """
        self.data = data
        self.timestamps = timestamps
        self.lookback = lookback
        self.delay = delay
        self.step = step
        self.target_idx = target_idx

        self.indices = []
        lower = max(lookback, start_anchor)
        upper = min(end_anchor, len(data) - delay)

        for i in range(lower, upper):
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        x = self.data[i - self.lookback : i : self.step]
        y = self.data[i + self.delay, self.target_idx]
        t = self.timestamps[i + self.delay]

        return {
            "inputs": torch.tensor(x, dtype=torch.float32),
            "label": torch.tensor(y, dtype=torch.float32),
            "time": str(t),
        }


def load_temperature_dataframe(csv_path: str):
    df = pd.read_csv(csv_path)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.sort_values("Date Time").reset_index(drop=True)
    return df


def split_and_standardize(
    df: pd.DataFrame,
    target_col: str = "T (degC)",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    """
    Split by time order and standardize using training statistics only.
    """
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError(
            "train_ratio and val_ratio must satisfy: train_ratio > 0, "
            "val_ratio > 0, train_ratio + val_ratio < 1.0"
        )

    feature_cols = [c for c in df.columns if c != "Date Time"]

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError(
            f"Invalid split points: train_end={train_end}, val_end={val_end}, total_rows={n}. "
            "Please adjust train_ratio and val_ratio."
        )

    train_df = df.iloc[:train_end].copy()

    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1.0)

    scaled_df = df.copy()
    scaled_df[feature_cols] = (df[feature_cols] - mean) / std

    target_idx = feature_cols.index(target_col)

    return scaled_df, feature_cols, target_idx, train_end, val_end, mean, std


def make_datasets(
    csv_path: str,
    lookback: int = 720,
    delay: int = 144,
    step: int = 3,
    target_col: str = "T (degC)",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    df = load_temperature_dataframe(csv_path)
    scaled_df, feature_cols, target_idx, train_end, val_end, mean, std = split_and_standardize(
        df,
        target_col=target_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    data = scaled_df[feature_cols].to_numpy(dtype=np.float32)
    timestamps = scaled_df["Date Time"].tolist()

    train_dataset = TempDataset(
        data=data,
        timestamps=timestamps,
        lookback=lookback,
        delay=delay,
        step=step,
        target_idx=target_idx,
        start_anchor=lookback,
        end_anchor=train_end - delay,
    )

    val_dataset = TempDataset(
        data=data,
        timestamps=timestamps,
        lookback=lookback,
        delay=delay,
        step=step,
        target_idx=target_idx,
        start_anchor=train_end,
        end_anchor=val_end - delay,
    )

    test_dataset = TempDataset(
        data=data,
        timestamps=timestamps,
        lookback=lookback,
        delay=delay,
        step=step,
        target_idx=target_idx,
        start_anchor=val_end,
        end_anchor=len(data) - delay,
    )

    target_mean = float(mean[target_col])
    target_std = float(std[target_col])

    return {
        "df": df,
        "scaled_df": scaled_df,
        "timestamps": timestamps,
        "feature_cols": feature_cols,
        "target_idx": target_idx,
        "train_end": train_end,
        "val_end": val_end,
        "mean": mean,
        "std": std,
        "target_mean": target_mean,
        "target_std": target_std,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "effective_seq_len": lookback // step,
    }
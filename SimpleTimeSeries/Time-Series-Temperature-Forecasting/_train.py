import os
import math
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import make_datasets
from models._model_factory import build_model


MODEL_NAME = "lstm"  # lstm / gru / convlstm / cnn_gru / nbeats / nhits / dlinear / patchtst / itransformer / timesnet / tcn / xlstm /  rwkv / transformer / selfattn_gru / multiattn_gru / luong_gru / cnn_gru_lstm / stacked_gru_lstm / gru_transformer / deep_fusion_gru / ConvRNN
CSV_PATH = "jena_climate_2009_2016.csv"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-4
LOOKBACK = 720
DELAY = 144
STEP = 3
SEED = 42
PATIENCE = 10
MIN_DELTA = 1e-6
GRAD_CLIP_NORM = 1.0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def denormalize(x, mean, std):
    return x * std + mean


def get_num_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def forward_model(model, inputs, model_name):
    """
    Forward pass wrapper.

    Most models in this project expect [B, T, F].
    TCNRegressor already converts [B, T, F] -> [B, F, T] internally,
    so we must NOT transpose again here.
    """
    if model_name == "mlp":
        return model(inputs.view(inputs.size(0), -1))

    return model(inputs)


def evaluate(model, loader, criterion, device, target_mean, target_std, model_name):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0

    preds_list = []
    labels_list = []
    times_list = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            times = batch["time"]

            outputs = forward_model(model, inputs, model_name)
            loss = criterion(outputs, labels)

            pred_c = denormalize(outputs, target_mean, target_std)
            label_c = denormalize(labels, target_mean, target_std)

            abs_err = torch.abs(pred_c - label_c)
            sq_err = (pred_c - label_c) ** 2

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_mae += abs_err.sum().item()
            total_mse += sq_err.sum().item()
            total_samples += bs

            preds_list.append(pred_c.detach().cpu().numpy())
            labels_list.append(label_c.detach().cpu().numpy())
            times_list.extend(times)

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples
    avg_rmse = math.sqrt(avg_mse)

    preds_all = np.concatenate(preds_list, axis=0) if preds_list else np.array([])
    labels_all = np.concatenate(labels_list, axis=0) if labels_list else np.array([])

    return {
        "loss": avg_loss,   # normalized MSE
        "mae": avg_mae,     # original °C
        "mse": avg_mse,     # original °C^2
        "rmse": avg_rmse,   # original °C
        "preds": preds_all,
        "labels": labels_all,
        "times": times_list,
    }


def train_one_epoch(model, loader, criterion, optimizer, device, target_mean, target_std, model_name, grad_clip_norm=1.0):
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch in progress_bar:
        inputs = batch["inputs"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = forward_model(model, inputs, model_name)
        loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        pred_c = denormalize(outputs, target_mean, target_std)
        label_c = denormalize(labels, target_mean, target_std)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_mae += torch.abs(pred_c - label_c).sum().item()
        total_samples += bs

        progress_bar.set_postfix({
            "Train MSE": f"{total_loss / total_samples:.4f}",
            "Train MAE(°C)": f"{total_mae / total_samples:.4f}",
        })

    return {
        "loss": total_loss / total_samples,
        "mae": total_mae / total_samples,
    }


def get_label_time_range(dataset, timestamps):
    if len(dataset) == 0:
        return ("N/A", "N/A")

    first_anchor = dataset.indices[0]
    last_anchor = dataset.indices[-1]

    first_label_time = timestamps[first_anchor + dataset.delay]
    last_label_time = timestamps[last_anchor + dataset.delay]

    return str(first_label_time), str(last_label_time)


def save_predictions_csv(result, path):
    pred_df = pd.DataFrame({
        "Date Time": result["times"],
        "Actual_T(degC)": result["labels"],
        "Pred_T(degC)": result["preds"],
        "Abs_Error(degC)": np.abs(result["preds"] - result["labels"]),
    })
    pred_df.to_csv(path, index=False)


def train_model(
    model_selection="tcn",
    csv_path="jena_climate_2009_2016.csv",
    epochs=10,
    batch_size=128,
    lr=1e-3,
    lookback=720,
    delay=144,
    step=3,
    train_ratio=0.7,
    val_ratio=0.1,
    seed=42,
    patience=10,
    min_delta=1e-6,
    grad_clip_norm=1.0,
):
    set_seed(seed)

    os.makedirs("record", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    data_bundle = make_datasets(
        csv_path=csv_path,
        lookback=lookback,
        delay=delay,
        step=step,
        target_col="T (degC)",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    train_dataset = data_bundle["train_dataset"]
    val_dataset = data_bundle["val_dataset"]
    test_dataset = data_bundle["test_dataset"]

    target_mean = data_bundle["target_mean"]
    target_std = data_bundle["target_std"]
    feature_cols = data_bundle["feature_cols"]
    timestamps = data_bundle["timestamps"]

    seq_len = data_bundle["effective_seq_len"]
    num_features = len(feature_cols)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_selection == "mlp":
        input_dim = seq_len * num_features
    else:
        input_dim = num_features

    model = build_model(model_selection, input_dim=input_dim, seq_len=seq_len).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_params, trainable_params = get_num_parameters(model)
    best_path = f"checkpoints/{model_selection}_best.pt"
    history_path = f"TrainRecord/{model_selection}_record.csv"
    summary_path = f"record/{model_selection}_summary.json"
    val_pred_path = f"record/{model_selection}_val_predictions.csv"
    test_pred_path = f"record/{model_selection}_test_predictions.csv"

    train_start_time, train_end_time = get_label_time_range(train_dataset, timestamps)
    val_start_time, val_end_time = get_label_time_range(val_dataset, timestamps)
    test_start_time, test_end_time = get_label_time_range(test_dataset, timestamps)

    print("=" * 88)
    print("Dataset summary")
    print(f"Rows: {data_bundle['n_rows']}")
    print(f"Features: {data_bundle['n_features']}")
    print(f"Target feature: T (degC)")
    print(f"Lookback: {lookback} steps")
    print(f"Delay: {delay} steps")
    print(f"Step inside window: {step}")
    print(f"Effective input sequence length: {seq_len}")
    print(f"Train/Val/Test samples: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}")
    print(f"Train time range: {train_start_time} -> {train_end_time}")
    print(f"Val time range:   {val_start_time} -> {val_end_time}")
    print(f"Test time range:  {test_start_time} -> {test_end_time}")
    print(f"Device: {device}")
    print(f"Model: {model_selection}")
    print("=" * 88)
    print(f"Parameter count: total={total_params:,}, trainable={trainable_params:,}")
    print(f"Model checkpoint path: {best_path}")
    print(f"Patience: {patience}")
    print("=" * 88)
    print(model)
    print("=" * 88)

    # Sanity check for input shape
    sample_batch = next(iter(train_loader))
    sample_inputs = sample_batch["inputs"]
    print(f"Sample batch input shape: {tuple(sample_inputs.shape)}")
    print("Expected model input shape for current setup: [B, T, F]")
    print("TCNRegressor will transpose internally to [B, F, T].")
    print("=" * 88)

    history = []
    best_val_rmse = float("inf")
    best_epoch = 0
    best_val_metrics = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            model_name=model_selection,
            grad_clip_norm=grad_clip_norm,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            model_name=model_selection,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_metrics['loss']:.6f} | Train MAE: {train_metrics['mae']:.4f}°C | "
            f"Val MSE: {val_metrics['loss']:.6f} | Val MAE: {val_metrics['mae']:.4f}°C | "
            f"Val RMSE: {val_metrics['rmse']:.4f}°C | LR: {current_lr:.6e}"
        )

        is_best = val_metrics["rmse"] < (best_val_rmse - min_delta)
        if is_best:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_selection": model_selection,
                    "input_dim": input_dim,
                    "seq_len": seq_len,
                    "lookback": lookback,
                    "delay": delay,
                    "step": step,
                    "train_ratio": train_ratio,
                    "val_ratio": val_ratio,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "feature_cols": feature_cols,
                    "best_val_rmse": best_val_rmse,
                    "best_val_mae": val_metrics["mae"],
                    "best_val_mse": val_metrics["loss"],
                },
                best_path,
            )
            print(f"  -> New best checkpoint saved at epoch {epoch}")
        else:
            patience_counter += 1

        history.append(
            {
                "epoch": epoch,
                "model": model_selection,
                "train_mse_norm": train_metrics["loss"],
                "train_mae_c": train_metrics["mae"],
                "val_mse_norm": val_metrics["loss"],
                "val_mae_c": val_metrics["mae"],
                "val_rmse_c": val_metrics["rmse"],
                "lr": current_lr,
                "best_so_far": is_best,
                "lookback": lookback,
                "delay": delay,
                "step": step,
                "batch_size": batch_size,
                "lr_config": lr,
                "target_mean": target_mean,
                "target_std": target_std,
            }
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

    record_df = pd.DataFrame(history)
    record_df.to_csv(history_path, index=False)

    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Best checkpoint was not saved: {best_path}")

    ckpt = torch.load(best_path, map_location=device,weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_result = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        model_name=model_selection,
    )

    test_result = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        model_name=model_selection,
    )

    save_predictions_csv(val_result, val_pred_path)
    save_predictions_csv(test_result, test_pred_path)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_selection,
        "csv_path": csv_path,
        "seed": seed,
        "device": str(device),
        "rows": data_bundle["n_rows"],
        "features": data_bundle["n_features"],
        "target_feature": "T (degC)",
        "lookback": lookback,
        "delay": delay,
        "step": step,
        "effective_seq_len": seq_len,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_time_range": [train_start_time, train_end_time],
        "val_time_range": [val_start_time, val_end_time],
        "test_time_range": [test_start_time, test_end_time],
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "checkpoint_path": best_path,
        "best_epoch": int(best_epoch),
        "best_val_rmse_c": float(best_val_metrics["rmse"]) if best_val_metrics else None,
        "best_val_mae_c": float(best_val_metrics["mae"]) if best_val_metrics else None,
        "best_val_mse_norm": float(best_val_metrics["loss"]) if best_val_metrics else None,
        "final_val_rmse_c": float(val_result["rmse"]),
        "final_val_mae_c": float(val_result["mae"]),
        "final_val_mse_norm": float(val_result["loss"]),
        "final_test_rmse_c": float(test_result["rmse"]),
        "final_test_mae_c": float(test_result["mae"]),
        "final_test_mse_norm": float(test_result["loss"]),
        "history_csv": history_path,
        "val_predictions_csv": val_pred_path,
        "test_predictions_csv": test_pred_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\nFinal Report")
    print(f"# {MODEL_NAME}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val RMSE: {best_val_metrics['rmse']:.4f} °C" if best_val_metrics else "Best val RMSE: N/A")
    print(f"Best val MAE : {best_val_metrics['mae']:.4f} °C" if best_val_metrics else "Best val MAE : N/A")
    print(f"Validation RMSE (best checkpoint): {val_result['rmse']:.4f} °C")
    print(f"Validation MAE  (best checkpoint): {val_result['mae']:.4f} °C")
    print(f"Test MSE  : {test_result['loss']:.6f}   (normalized)")
    print(f"Test MAE  : {test_result['mae']:.4f} °C")
    print(f"Test RMSE : {test_result['rmse']:.4f} °C")
    print(f"Best model saved to: {best_path}")
    print(f"Epoch history saved to: {history_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Validation predictions saved to: {val_pred_path}")
    print(f"Test predictions saved to: {test_pred_path}")


if __name__ == "__main__":
    train_model(
        model_selection=MODEL_NAME,
        csv_path=CSV_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        lookback=LOOKBACK,
        delay=DELAY,
        step=STEP,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        grad_clip_norm=GRAD_CLIP_NORM,
    )
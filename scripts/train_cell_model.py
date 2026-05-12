# -*- coding: utf-8 -*-

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.utils.io import load_yaml, load_pickle, save_pickle, ensure_dir, save_npz
from src.data.sampling import (
    extract_soc_samples,
    extract_soh_samples,
    data_load_soc,
    data_load_soh,
)
from src.data.dataset import build_train_test_pairs, create_dataloaders
from src.models.dual_cnn_lstm import DualCNNLSTMModel, init_weights
from src.training.trainer import train_model
from src.training.evaluate import evaluate_model


def build_voltage_range(config):
    v_cfg = config["sampling"]["voltage"]

    voltage_range = np.arange(
        v_cfg["min_v"],
        v_cfg["max_v"] + v_cfg["step_v"],
        v_cfg["step_v"],
    )

    return np.round(voltage_range, v_cfg.get("round_digits", 4))


def get_soh_num_samples(config, voltage_range):
    soh_cfg = config["sampling"]["soh"]
    window_size = soh_cfg["window_size"]
    mode = soh_cfg.get("num_samples_mode", "auto_from_voltage_range")

    if "num_samples" in soh_cfg:
        return int(soh_cfg["num_samples"])

    if mode == "auto_from_voltage_range":
        return len(voltage_range) - window_size

    if mode == "auto_from_voltage_range_plus_one":
        return len(voltage_range) - window_size + 1

    raise ValueError(f"Unsupported SOH num_samples_mode: {mode}")


def load_cell_cycles(config):
    loaded_data = []

    print("Cell data loading order:")
    for rel_path in config["dataset"]["cell_data_paths"]:
        file_path = ROOT / rel_path
        print("  ", file_path)
        data = load_pickle(file_path)
        loaded_data.extend(data)

    return loaded_data


def prepare_cell_timeseries(config, cycle_timeseries):
    cell_ts_cfg = config["sampling"].get("cell_timeseries", {})

    downsample_if_longer_than = cell_ts_cfg.get("downsample_if_longer_than", None)
    downsample_stride = cell_ts_cfg.get("downsample_stride", 1)

    if (
        downsample_if_longer_than is not None
        and len(cycle_timeseries) > downsample_if_longer_than
    ):
        cycle_timeseries = cycle_timeseries[::downsample_stride]

    return cycle_timeseries


def prepare_cycle_metadata(config, cycle_metadata):
    """
    Prepare cycle-level metadata before sample extraction.

    Default: keep metadata unchanged.
    ESSH1: recompute DR as:
        DR = (initial_capacity - SOH * rate_capacity) / cycle
    """
    metadata = cycle_metadata.copy()

    metadata_cfg = config["sampling"].get("metadata", {})
    dr_mode = metadata_cfg.get("dr_mode", "as_is")

    if dr_mode in [None, "as_is", "metadata"]:
        return metadata

    if dr_mode == "essh1_recompute":
        required_cols = [
            "rate_capacity (Ah)",
            "cycle",
            "initial_capacity (Ah)",
            "SOH",
        ]

        missing_cols = [
            col for col in required_cols
            if col not in metadata.columns
        ]

        if missing_cols:
            raise KeyError(
                f"Cannot recompute DR. Missing metadata columns: {missing_cols}"
            )

        rate_capacity = metadata["rate_capacity (Ah)"].iloc[0]
        cycle_number = metadata["cycle"].iloc[0]
        initial_capacity = metadata["initial_capacity (Ah)"].iloc[0]
        soh = metadata["SOH"].iloc[0]

        if cycle_number == 0:
            metadata["DR"] = np.nan
        else:
            metadata["DR"] = (
                initial_capacity - soh * rate_capacity
            ) / cycle_number

        return metadata

    raise ValueError(f"Unsupported metadata dr_mode: {dr_mode}")


def has_invalid_array(x):
    return np.any(np.isnan(x)) or np.any(np.isinf(x))


def build_processed_pairs(config, loaded_data):
    columns = config["columns"]
    sampling = config["sampling"]

    voltage_range = build_voltage_range(config)
    step_size = sampling["voltage"]["step_v"]

    min_length = sampling["min_length"]

    soc_cfg = sampling["soc"]
    soh_cfg = sampling["soh"]

    soc_window_size = soc_cfg["window_size"]
    soc_num_samples = soc_cfg["num_samples"]

    soh_window_size = soh_cfg["window_size"]
    soh_num_samples = get_soh_num_samples(config, voltage_range)

    soc_pad_short_windows = soc_cfg.get("pad_short_windows", False)
    soc_unique_start_indices = soc_cfg.get("unique_start_indices", False)

    soh_pad_short_windows = soh_cfg.get("pad_short_windows", True)
    soh_unique_start_indices = soh_cfg.get("unique_start_indices", False)
    soh_time_transform = soh_cfg.get("time_transform", "log10")

    filter_invalid_all = sampling.get(
        "filter_invalid_all",
        config["dataset"]["name"].upper() == "PBSL1",
    )

    processed_data_pairs = []

    for cycle_idx, cycle_all in enumerate(loaded_data):
        cycle_metadata = prepare_cycle_metadata(
            config=config,
            cycle_metadata=cycle_all["constant_metadata"],
        )

        cycle_timeseries = cycle_all["timeseries_data"]
        cycle_timeseries = prepare_cell_timeseries(config, cycle_timeseries)

        if cycle_timeseries.shape[0] < min_length:
            continue

        delta_q = (
            cycle_timeseries[columns["charge_capacity"]].iloc[-1]
            - cycle_timeseries[columns["charge_capacity"]].iloc[soc_window_size - 1]
        ) / cycle_metadata["rate_capacity (Ah)"].iloc[0]

        soc_samples = extract_soc_samples(
            cycle_timeseries=cycle_timeseries,
            cycle_metadata=cycle_metadata,
            window_size=soc_window_size,
            num_samples=soc_num_samples,
            voltage_name=columns["cell_voltage"],
            temperature_name=columns["cell_temperature"],
            pad_short_windows=soc_pad_short_windows,
            unique_start_indices=soc_unique_start_indices,
        )

        soh_samples = extract_soh_samples(
            cycle_timeseries=cycle_timeseries,
            cycle_metadata=cycle_metadata,
            delta_q=delta_q,
            voltage_range=voltage_range,
            window_size=soh_window_size,
            num_samples=soh_num_samples,
            voltage_name=columns["cell_voltage"],
            temperature_name=columns["cell_temperature"],
            step_size=step_size,
            pad_short_windows=soh_pad_short_windows,
            unique_start_indices=soh_unique_start_indices,
            time_transform=soh_time_transform,
        )

        if len(soc_samples) == 0 or len(soh_samples) == 0:
            continue

        soh_all_nan = all(np.all(np.isnan(sample)) for sample in soh_samples)
        soc_all_nan = all(np.all(np.isnan(sample)) for sample in soc_samples)

        if soh_all_nan or soc_all_nan:
            continue

        data_soh, label_soh = data_load_soh(soh_samples)
        data_soc, label_soc = data_load_soc(soc_samples)

        if has_invalid_array(label_soc):
            continue

        if filter_invalid_all:
            if (
                has_invalid_array(label_soh)
                or has_invalid_array(data_soh)
                or has_invalid_array(data_soc)
            ):
                continue

        processed_data_pairs.append(
            (
                data_soh,
                label_soh,
                data_soc,
                label_soc,
            )
        )

    return processed_data_pairs


def get_device(config):
    if config["training"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["training"]["device"])

    if device.type == "cuda":
        torch.cuda.set_device(config["training"].get("cuda_device", 0))

    return device


def make_scheduler(config, optimizer):
    sch_cfg = config["training"]["scheduler"]

    if sch_cfg["name"] == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sch_cfg["mode"],
            factor=sch_cfg["factor"],
            patience=sch_cfg["patience"],
            min_lr=sch_cfg["min_lr"],
            eps=sch_cfg["eps"],
        )

    return None


def build_model(config, device):
    model_cfg = config["model"]

    model_kwargs = {
        "soh_input_channels": model_cfg["soh"]["input_channels"],
        "soc_input_channels": model_cfg["soc"]["input_channels"],
        "soh_seq_length": model_cfg["soh"]["seq_length"],
        "soc_seq_length": model_cfg["soc"]["seq_length"],
        "hidden_dim": model_cfg["hidden_dim"],
        "aux_dim": model_cfg["aux_dim"],
        "conv_dropout": model_cfg.get("conv_dropout", 0.0),
        "soh_aux_start": model_cfg.get("soh_aux_start", 6),
        "soc_aux_start": model_cfg.get("soc_aux_start", 2),
    }

    accepted = inspect.signature(DualCNNLSTMModel.__init__).parameters
    model_kwargs = {
        key: value
        for key, value in model_kwargs.items()
        if key in accepted
    }

    model = DualCNNLSTMModel(**model_kwargs).to(device)

    seed = config["training"].get("seed", 42)
    try:
        model.apply(lambda m: init_weights(m, seed=seed))
    except TypeError:
        model.apply(init_weights)

    return model


def flatten_valid(label, pred):
    label = np.asarray(label).reshape(-1)
    pred = np.asarray(pred).reshape(-1)

    mask = np.isfinite(label) & np.isfinite(pred)
    return label[mask], pred[mask]


def plot_training_history(history, save_path, show_figures=False):
    epochs = np.arange(1, len(history["total"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.plot(epochs, history["total"], label="Total Loss")
    ax.plot(epochs, history["soh"], label="SOH Loss")
    ax.plot(epochs, history["soc"], label="SOC Loss")
    ax.plot(epochs, history["soh_soc"], label="SOH-SOC Loss")
    ax.plot(epochs, history["dr"], label="DR Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


    plt.show()



def plot_parity(ax, label, pred, title, metrics_text):
    label, pred = flatten_valid(label, pred)

    if len(label) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        return

    data_min = min(np.min(label), np.min(pred))
    data_max = max(np.max(label), np.max(pred))

    ax.scatter(label, pred, s=8, alpha=0.5)
    ax.plot([data_min, data_max], [data_min, data_max], "--", linewidth=1)

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax.text(
        0.03,
        0.97,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.7"),
    )


def plot_train_test_results(train_results, test_results, save_path, show_figures=False):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)

    train_metrics = train_results["metrics"]
    test_metrics = test_results["metrics"]

    plot_parity(
        axes[0, 0],
        train_results["labels"]["soh"],
        train_results["predictions"]["soh"],
        "Train SOH",
        (
            f"MAE={train_metrics['soh_mae']:.4f}\n"
            f"RMSE={train_metrics['soh_rmse']:.4f}\n"
            f"MAPE={train_metrics['soh_mape']:.2f}%"
        ),
    )

    plot_parity(
        axes[0, 1],
        test_results["labels"]["soh"],
        test_results["predictions"]["soh"],
        "Test SOH",
        (
            f"MAE={test_metrics['soh_mae']:.4f}\n"
            f"RMSE={test_metrics['soh_rmse']:.4f}\n"
            f"MAPE={test_metrics['soh_mape']:.2f}%"
        ),
    )

    plot_parity(
        axes[1, 0],
        train_results["labels"]["soc"],
        train_results["predictions"]["soc"],
        "Train SOC",
        (
            f"MAE={train_metrics['soc_mae']:.4f}\n"
            f"RMSE={train_metrics['soc_rmse']:.4f}\n"
            f"MAPE={train_metrics['soc_mape']:.2f}%"
        ),
    )

    plot_parity(
        axes[1, 1],
        test_results["labels"]["soc"],
        test_results["predictions"]["soc"],
        "Test SOC",
        (
            f"MAE={test_metrics['soc_mae']:.4f}\n"
            f"RMSE={test_metrics['soc_rmse']:.4f}\n"
            f"MAPE={test_metrics['soc_mape']:.2f}%"
        ),
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


    plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "essl1.yaml"),
    )
    args = parser.parse_args()

    config = load_yaml(args.config)

    print(f"Config file: {args.config}")
    print(f"Dataset name: {config['dataset']['name']}")
    print(f"Checkpoint name: {config['output']['best_model_name']}")

    seed = config["training"]["seed"]
    set_seed(seed)

    checkpoint_dir = ensure_dir(ROOT / config["output"]["checkpoint_dir"])
    prediction_dir = ensure_dir(ROOT / config["output"]["prediction_dir"])
    figure_dir = ensure_dir(ROOT / config["output"]["figure_dir"])
    ensure_dir(ROOT / config["output"]["table_dir"])

    device = get_device(config)
    print(f"Using device: {device}")

    loaded_data = load_cell_cycles(config)
    print(f"Loaded cell cycles: {len(loaded_data)}")

    processed_data_pairs = build_processed_pairs(config, loaded_data)
    print(f"Valid processed cycles: {len(processed_data_pairs)}")

    if len(processed_data_pairs) == 0:
        raise RuntimeError(
            "No valid processed cycles were found. "
            "Please check data paths, voltage range, and column names."
        )

    train_pairs, test_pairs, norm_params = build_train_test_pairs(
        processed_data_pairs=processed_data_pairs,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    print(f"Train cycles: {len(train_pairs)}")
    print(f"Test cycles : {len(test_pairs)}")

    train_loader, test_loader = create_dataloaders(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        batch_size=config["dataloader"]["batch_size"],
        seed=seed,
        num_workers=config["dataloader"]["num_workers"],
    )

    model = build_model(config, device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = make_scheduler(config, optimizer)

    checkpoint_path = checkpoint_dir / config["output"]["best_model_name"]

    extra_state = {
        "config": config,
        "norm_params": norm_params,
    }

    early_cfg = config["training"].get("early_stopping", {})
    if early_cfg.get("enabled", True):
        early_patience = early_cfg.get("patience", 20)
    else:
        early_patience = None

    loss_cfg = config.get("loss", {})
    dr_mode = loss_cfg.get("dr_mode", "capacity_difference")
    dr_target_scale = loss_cfg.get("dr_target_scale", 1.0)

    history, best_loss = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_weights=config["loss"]["weights"],
        num_epochs=config["training"]["num_epochs"],
        checkpoint_path=str(checkpoint_path),
        early_stopping_patience=early_patience,
        extra_state=extra_state,
        seed=seed,
        dr_mode=dr_mode,
        dr_target_scale=dr_target_scale,
    )

    print(f"Best training loss: {best_loss:.6f}")

    with open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=device)

    print("Checkpoint keys:", checkpoint.keys())

    if "norm_params" not in checkpoint:
        raise KeyError("norm_params was not saved in the checkpoint.")

    model.load_state_dict(checkpoint["model_state_dict"])

    train_results = evaluate_model(
        model=model,
        data_loader=train_loader,
        device=device,
        loss_weights=config["loss"]["weights"],
        dataset_name="Training Set",
        dr_mode=dr_mode,
        dr_target_scale=dr_target_scale,
    )

    test_results = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        loss_weights=config["loss"]["weights"],
        dataset_name="Test Set",
        dr_mode=dr_mode,
        dr_target_scale=dr_target_scale,
    )

    summary_path = prediction_dir / f"{config['dataset']['name'].lower()}_training_summary.pkl"

    save_pickle(
        {
            "history": history,
            "best_loss": best_loss,
            "norm_params": norm_params,
            "train_metrics": train_results["metrics"],
            "test_metrics": test_results["metrics"],
            "checkpoint_path": str(checkpoint_path),
            "dr_mode": dr_mode,
            "dr_target_scale": dr_target_scale,
        },
        summary_path,
    )

    cell_prediction_file = config["output"].get(
        "cell_prediction_file",
        f"outputs/predictions/{config['dataset']['name'].lower()}_cell_predictions.npz",
    )
    cell_prediction_path = ROOT / cell_prediction_file

    save_npz(
        cell_prediction_path,
        train_soh_pred=train_results["predictions"]["soh"],
        train_soh_label=train_results["labels"]["soh"],
        train_soc_pred=train_results["predictions"]["soc"],
        train_soc_label=train_results["labels"]["soc"],
        train_dr_pred=train_results["predictions"]["dr"],
        train_dr_label=train_results["labels"]["dr"],
        test_soh_pred=test_results["predictions"]["soh"],
        test_soh_label=test_results["labels"]["soh"],
        test_soc_pred=test_results["predictions"]["soc"],
        test_soc_label=test_results["labels"]["soc"],
        test_dr_pred=test_results["predictions"]["dr"],
        test_dr_label=test_results["labels"]["dr"],
    )

    dataset_name_lower = config["dataset"]["name"].lower()
    show_figures = config.get("plot", {}).get("show_figures", False)

    loss_fig_path = figure_dir / f"{dataset_name_lower}_training_loss.png"
    result_fig_path = figure_dir / f"{dataset_name_lower}_train_test_results.png"

    plot_training_history(
        history=history,
        save_path=loss_fig_path,
        show_figures=show_figures,
    )

    plot_train_test_results(
        train_results=train_results,
        test_results=test_results,
        save_path=result_fig_path,
        show_figures=show_figures,
    )

    print("Training finished.")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Training summary saved to: {summary_path}")
    print(f"Predictions saved to: {cell_prediction_path}")
    print(f"Loss figure saved to: {loss_fig_path}")
    print(f"Train/Test result figure saved to: {result_fig_path}")


if __name__ == "__main__":
    main()
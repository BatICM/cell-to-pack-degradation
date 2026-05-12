# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml, ensure_dir
from src.pack.capacity import (
    estimate_pack_soh_from_cell_states,
    compute_error_metrics,
    build_pack_estimation_table,
)


def plot_pack_estimation(
    table,
    metrics_proposed,
    metrics_min_cell,
    save_path,
    dataset_name,
    show_figures=False,
):
    x = table["Index"].values
    real = table["Real"].values
    proposed = table["Pack_SOH"].values
    min_cell = table["Min_cells"].values

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    ax1 = axes[0]
    ax2 = axes[1]

    ax1.plot(x, real, label="Real", linewidth=1.8)
    ax1.plot(x, proposed, label="Proposed", linewidth=1.5)
    ax1.plot(x, min_cell, label="Min-cell", linewidth=1.5)

    ax1.set_ylabel("SOH (%)")
    ax1.set_title(f"{dataset_name} Pack SOH Estimation")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False)

    metrics_text = (
        "Proposed:\n"
        f"MAE = {metrics_proposed['mae_percent']:.4f}%\n"
        f"RMSE = {metrics_proposed['rmse_percent']:.4f}%\n"
        "\n"
        "Min-cell:\n"
        f"MAE = {metrics_min_cell['mae_percent']:.4f}%\n"
        f"RMSE = {metrics_min_cell['rmse_percent']:.4f}%"
    )

    ax1.text(
        0.02,
        0.02,
        metrics_text,
        transform=ax1.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    proposed_error = proposed - real
    min_cell_error = min_cell - real

    ax2.plot(x, proposed_error, label="Proposed error", linewidth=1.3)
    ax2.plot(x, min_cell_error, label="Min-cell error", linewidth=1.3)
    ax2.axhline(0, linestyle="--", linewidth=1.0)

    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Error (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


    plt.show()



def get_capacity_estimation_config(config):
    capacity_cfg = (
        config
        .get("pack_inference", {})
        .get("capacity_estimation", {})
    )

    reduce_soh_windows = capacity_cfg.get("reduce_soh_windows", None)
    iqr_multiplier = capacity_cfg.get("iqr_multiplier", 1.5)

    if reduce_soh_windows in ["", "null", "none", "None"]:
        reduce_soh_windows = None

    soc_cleaning_cfg = capacity_cfg.get("soc_cleaning", {})

    soc_cleaning_enabled = soc_cleaning_cfg.get("enabled", False)
    soc_cleaning_method = soc_cleaning_cfg.get("method", "median_std")
    soc_cleaning_k = soc_cleaning_cfg.get("k", 1.0)

    return {
        "reduce_soh_windows": reduce_soh_windows,
        "iqr_multiplier": iqr_multiplier,
        "soc_cleaning_enabled": soc_cleaning_enabled,
        "soc_cleaning_method": soc_cleaning_method,
        "soc_cleaning_k": soc_cleaning_k,
    }


def load_external_pack_soh(config):
    labels_cfg = config.get("labels", {})

    pack_capacity_path = ROOT / labels_cfg["pack_capacity_path"]
    column_start = int(labels_cfg.get("pack_capacity_column_start", 1))
    divisor = float(labels_cfg.get("capacity_to_soh_divisor", 1.0))

    if not pack_capacity_path.exists():
        raise FileNotFoundError(
            f"External pack capacity file was not found: {pack_capacity_path}"
        )

    pack_capacity = pd.read_csv(pack_capacity_path)
    pack_capacity_arr = np.asarray(pack_capacity)

    if pack_capacity_arr.shape[1] <= column_start:
        raise ValueError(
            f"pack_capacity file has shape {pack_capacity_arr.shape}, "
            f"but pack_capacity_column_start={column_start}."
        )

    real_pack_soh = pack_capacity_arr[:, column_start:].reshape(-1) / divisor

    return real_pack_soh.astype(float)


def get_real_pack_soh(config, npz_real_pack_soh, expected_length):
    labels_cfg = config.get("labels", {})
    source = labels_cfg.get("real_pack_soh_source", "metadata")

    if source in [None, "metadata", "npz", "pack_metadata"]:
        real_pack_soh = np.asarray(npz_real_pack_soh, dtype=float)

    elif source == "external_pack_capacity":
        real_pack_soh = load_external_pack_soh(config)

    else:
        raise ValueError(f"Unsupported real_pack_soh_source: {source}")

    if len(real_pack_soh) != expected_length:
        raise ValueError(
            f"Length mismatch between predictions and real pack SOH: "
            f"prediction length={expected_length}, real_pack_soh length={len(real_pack_soh)}. "
            f"Please check label alignment."
        )

    return real_pack_soh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "essl1.yaml"),
    )
    args = parser.parse_args()

    config = load_yaml(args.config)

    dataset_name = config["dataset"]["name"]
    dataset_name_lower = dataset_name.lower()

    print(f"Config file: {args.config}")
    print(f"Dataset name: {dataset_name}")

    prediction_file = ROOT / config["output"]["pack_prediction_file"]
    print(f"Loading pack predictions from: {prediction_file}")

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Pack prediction file was not found: {prediction_file}\n"
            f"Please run infer_pack_states.py first."
        )

    data = np.load(prediction_file, allow_pickle=True)

    soh_pred = data["soh_pred"]
    soc_pred = data["soc_pred"]
    npz_real_pack_soh = data["real_pack_soh"]

    print("soh_pred shape:", soh_pred.shape)
    print("soc_pred shape:", soc_pred.shape)
    print("npz real_pack_soh shape:", npz_real_pack_soh.shape)

    capacity_cfg = get_capacity_estimation_config(config)

    print("capacity reduce_soh_windows:", capacity_cfg["reduce_soh_windows"])
    print("capacity iqr_multiplier:", capacity_cfg["iqr_multiplier"])
    print("soc cleaning enabled:", capacity_cfg["soc_cleaning_enabled"])
    print("soc cleaning method:", capacity_cfg["soc_cleaning_method"])
    print("soc cleaning k:", capacity_cfg["soc_cleaning_k"])

    estimated = estimate_pack_soh_from_cell_states(
        soh_pred=soh_pred,
        soc_pred=soc_pred,
        reduce_soh_windows=capacity_cfg["reduce_soh_windows"],
        iqr_multiplier=capacity_cfg["iqr_multiplier"],
        soc_cleaning_enabled=capacity_cfg["soc_cleaning_enabled"],
        soc_cleaning_method=capacity_cfg["soc_cleaning_method"],
        soc_cleaning_k=capacity_cfg["soc_cleaning_k"],
    )

    real_pack_soh = get_real_pack_soh(
        config=config,
        npz_real_pack_soh=npz_real_pack_soh,
        expected_length=len(estimated["pack_soh"]),
    )

    print("real_pack_soh shape:", real_pack_soh.shape)
    print(
        "real_pack_soh min/max:",
        np.nanmin(real_pack_soh),
        np.nanmax(real_pack_soh),
    )

    metrics_proposed = compute_error_metrics(
        pred=estimated["pack_soh"],
        target=real_pack_soh,
    )

    metrics_min_cell = compute_error_metrics(
        pred=estimated["min_cell_soh"],
        target=real_pack_soh,
    )

    table = build_pack_estimation_table(
        real_pack_soh=real_pack_soh,
        estimated=estimated,
    )

    ensure_dir(ROOT / config["output"]["table_dir"])
    prediction_dir = ensure_dir(ROOT / config["output"]["prediction_dir"])
    figure_dir = ensure_dir(ROOT / config["output"]["figure_dir"])

    table_path = ROOT / config["output"]["pack_capacity_table"]
    table_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path = prediction_dir / f"{dataset_name_lower}_pack_metrics.json"
    figure_path = figure_dir / f"{dataset_name_lower}_pack_estimation.png"

    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    metrics = {
        "dataset": dataset_name,
        "real_pack_soh_source": config.get("labels", {}).get(
            "real_pack_soh_source",
            "metadata",
        ),
        "capacity_estimation": capacity_cfg,
        "proposed": metrics_proposed,
        "min_cell_baseline": metrics_min_cell,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    show_figures = config.get("plot", {}).get("show_figures", False)

    plot_pack_estimation(
        table=table,
        metrics_proposed=metrics_proposed,
        metrics_min_cell=metrics_min_cell,
        save_path=figure_path,
        dataset_name=dataset_name,
        show_figures=show_figures,
    )

    print("\nPack SOH estimation metrics")
    print("-" * 60)
    print("Proposed:")
    print(f"  MAE  = {metrics_proposed['mae_percent']:.4f} %")
    print(f"  RMSE = {metrics_proposed['rmse_percent']:.4f} %")
    print(f"  N    = {metrics_proposed['n_valid']}")

    print("Min-cell baseline:")
    print(f"  MAE  = {metrics_min_cell['mae_percent']:.4f} %")
    print(f"  RMSE = {metrics_min_cell['rmse_percent']:.4f} %")
    print(f"  N    = {metrics_min_cell['n_valid']}")

    print("\nSaved files:")
    print(table_path)
    print(metrics_path)
    print(figure_path)


if __name__ == "__main__":
    main()
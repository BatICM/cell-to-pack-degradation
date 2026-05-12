# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml, load_pickle
from src.data.sampling import extract_soc_samples, extract_soh_samples
from src.data.dataset import build_train_test_pairs, create_dataloaders
from src.models.dual_cnn_lstm import DualCNNLSTMModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "essl1.yaml"),
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Project root:")
    print(ROOT)

    print("\n" + "=" * 80)
    print("Loading config:")
    print(args.config)
    config = load_yaml(args.config)
    print("Dataset:", config["dataset"]["name"])

    print("\n" + "=" * 80)
    print("Checking cell-level pkl files")
    for rel_path in config["dataset"]["cell_data_paths"]:
        file_path = ROOT / rel_path
        print(file_path, "exists:", file_path.exists())
        if not file_path.exists():
            raise FileNotFoundError(file_path)

    print("\n" + "=" * 80)
    print("Checking pack-level pkl file")
    pack_path = ROOT / config["dataset"]["pack_data_path"]
    print(pack_path, "exists:", pack_path.exists())
    if not pack_path.exists():
        raise FileNotFoundError(pack_path)

    print("\n" + "=" * 80)
    print("Loading one cell-level pkl")
    first_cell_path = ROOT / config["dataset"]["cell_data_paths"][0]
    data = load_pickle(first_cell_path)
    print("Loaded object type:", type(data))
    print("Number of cycles in first file:", len(data))

    first_cycle = data[0]
    print("First cycle keys:", first_cycle.keys())

    cycle_metadata = first_cycle["constant_metadata"]
    cycle_timeseries = first_cycle["timeseries_data"]

    print("Metadata columns:")
    print(cycle_metadata.columns.tolist())

    print("Timeseries columns:")
    print(cycle_timeseries.columns.tolist()[:20])
    print("Timeseries shape:", cycle_timeseries.shape)

    required_columns = [
        config["columns"]["time"],
        config["columns"]["current"],
        config["columns"]["charge_capacity"],
        config["columns"]["soc"],
        config["columns"]["cell_voltage"],
        config["columns"]["cell_temperature"],
    ]

    print("\nChecking required columns:")
    for col in required_columns:
        print(col, "exists:", col in cycle_timeseries.columns)
        if col not in cycle_timeseries.columns:
            raise KeyError(f"Missing column in timeseries_data: {col}")

    required_meta = [
        "SOH",
        "rate_capacity (Ah)",
        "cycle",
        "DR",
        "initial_capacity (Ah)",
    ]

    print("\nChecking required metadata columns:")
    for col in required_meta:
        print(col, "exists:", col in cycle_metadata.columns)
        if col not in cycle_metadata.columns:
            raise KeyError(f"Missing column in constant_metadata: {col}")

    print("\n" + "=" * 80)
    print("Checking model initialization")
    model = DualCNNLSTMModel(
        soh_input_channels=config["model"]["soh"]["input_channels"],
        soc_input_channels=config["model"]["soc"]["input_channels"],
        hidden_dim=config["model"]["hidden_dim"],
        aux_dim=config["model"]["aux_dim"],
    )
    print(model.__class__.__name__, "initialized successfully")

    print("\n" + "=" * 80)
    print("Checking device")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
        print("current device:", torch.cuda.current_device())
        print("device name:", torch.cuda.get_device_name(0))

    print("\n" + "=" * 80)
    print("Setup check passed.")


if __name__ == "__main__":
    main()
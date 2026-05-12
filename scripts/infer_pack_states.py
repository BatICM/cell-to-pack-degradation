# -*- coding: utf-8 -*-

import argparse
import inspect
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml, load_pickle, save_npz
from src.data.sampling import (
    extract_soc_samples,
    extract_soh_samples,
    data_load_soc,
    data_load_soh,
)
from src.models.dual_cnn_lstm import DualCNNLSTMModel


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


def transfer_arr(data):
    n = len(data)
    n_windows = len(data[0])
    sample_shape = data[0][0].shape

    output_array = np.empty((n, n_windows) + sample_shape, dtype=np.float32)

    for i in range(n):
        for j in range(n_windows):
            output_array[i, j] = data[i][j]

    return output_array


def normalize_with_params(data, max_value, min_value):
    data_max_min = max_value - min_value

    normalized = (
        data - min_value[np.newaxis, np.newaxis, :]
    ) / data_max_min[np.newaxis, np.newaxis, :]

    return normalized


def numeric_suffix_key(name):
    numbers = re.findall(r"\d+", str(name))
    if len(numbers) == 0:
        return 10**12
    return int(numbers[-1])


def maybe_sort_columns(columns, sort_numeric_suffix=False):
    if sort_numeric_suffix:
        return sorted(columns, key=numeric_suffix_key)
    return columns


def get_temperature_name(config, temperature_columns, cell_number):
    temp_cfg = config["pack_inference"].get("temperature", {})
    mode = temp_cfg.get("mode", "paired")

    if mode == "fixed":
        return temp_cfg["fixed_temperature_column"]

    if mode == "paired":
        return temperature_columns[cell_number]

    raise ValueError(f"Unsupported temperature mode: {mode}")


def get_pack_cell_count(config, voltage_columns, temperature_columns):
    pack_cfg = config["pack_inference"]
    temp_cfg = pack_cfg.get("temperature", {})
    mode = temp_cfg.get("mode", "paired")

    if "n_cells" in pack_cfg and pack_cfg["n_cells"] is not None:
        n_cells = int(pack_cfg["n_cells"])

        if n_cells > len(voltage_columns):
            raise ValueError(
                f"Configured n_cells={n_cells}, but only "
                f"{len(voltage_columns)} voltage columns were found."
            )

        if mode == "paired" and n_cells > len(temperature_columns):
            raise ValueError(
                f"Configured n_cells={n_cells}, but only "
                f"{len(temperature_columns)} temperature columns were found."
            )

        return n_cells

    if mode == "fixed":
        return len(voltage_columns)

    if len(voltage_columns) < len(temperature_columns):
        raise ValueError(
            "The number of voltage channels is smaller than the number of "
            "temperature channels. Please check pack column names."
        )

    return len(temperature_columns)


def apply_pack_downsampling(config, cycle_timeseries):
    pack_ts_cfg = config["pack_inference"].get("pack_timeseries", {})
    stride = pack_ts_cfg.get("downsample_stride", 1)

    if stride is not None and stride > 1:
        cycle_timeseries = cycle_timeseries[::stride].copy()

    return cycle_timeseries


def apply_current_filter(config, cycle_timeseries):
    columns = config["columns"]
    pack_cfg = config["pack_inference"]
    current_filter = pack_cfg.get("current_filter", {})

    mode = current_filter.get("mode", "positive_charge")
    min_current_a = current_filter.get("min_current_a", 0.0)
    flip_sign_after_filter = current_filter.get("flip_sign_after_filter", False)

    current_col = columns["current"]

    if mode in [None, "positive_charge"]:
        cycle_timeseries = cycle_timeseries[
            cycle_timeseries[current_col] > min_current_a
        ].copy()

    elif mode == "negative_charge":
        cycle_timeseries = cycle_timeseries[
            cycle_timeseries[current_col] < -abs(min_current_a)
        ].copy()

        if flip_sign_after_filter:
            cycle_timeseries[current_col] = -cycle_timeseries[current_col]

    else:
        raise ValueError(f"Unsupported current_filter mode: {mode}")

    return cycle_timeseries


def get_voltage_for_filter(config, cycle_timeseries, voltage_name):
    pack_cfg = config["pack_inference"]
    voltage_filter = pack_cfg.get("voltage_filter", {})

    use_cell_voltage = voltage_filter.get("use_cell_voltage", False)
    use_pack_voltage = voltage_filter.get("use_pack_voltage", True)

    if use_cell_voltage:
        voltage_col = cycle_timeseries[voltage_name].values.astype(float)
        return voltage_col, voltage_name, 1.0

    if use_pack_voltage:
        pack_voltage_col = config["columns"]["pack_voltage"]
        battery_struct = config["dataset"]["pack_structure"]

        voltage_col = (
            cycle_timeseries[pack_voltage_col].values.astype(float)
            / battery_struct
        )
        return voltage_col, pack_voltage_col, battery_struct

    raise ValueError(
        "Either voltage_filter.use_cell_voltage or "
        "voltage_filter.use_pack_voltage must be true."
    )


def filter_voltage_window(config, cycle_timeseries, voltage_name, target_voltage_sequence):
    voltage_col, filter_column, multiplier = get_voltage_for_filter(
        config=config,
        cycle_timeseries=cycle_timeseries,
        voltage_name=voltage_name,
    )

    if len(voltage_col) == 0 or np.all(np.isnan(voltage_col)):
        return None

    min_voltage = np.nanmin(voltage_col)
    max_voltage = np.nanmax(voltage_col)

    min_target_voltage = np.min(target_voltage_sequence)
    max_target_voltage = np.max(target_voltage_sequence)

    if min_voltage > min_target_voltage or max_voltage < max_target_voltage:
        return None

    voltage_filter_cfg = config["pack_inference"].get("voltage_filter", {})
    crop_to_voltage_window = voltage_filter_cfg.get("crop_to_voltage_window", True)

    if not crop_to_voltage_window:
        return cycle_timeseries

    cycle_timeseries = cycle_timeseries[
        (cycle_timeseries[filter_column] >= min_target_voltage * multiplier)
        & (cycle_timeseries[filter_column] <= max_target_voltage * multiplier)
    ]

    if cycle_timeseries.shape[0] == 0:
        return None

    return cycle_timeseries


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

    return DualCNNLSTMModel(**model_kwargs).to(device)


def collect_pack_samples(config, pack_data):
    columns = config["columns"]
    sampling = config["sampling"]
    pack_cfg = config["pack_inference"]

    voltage_range = build_voltage_range(config)
    step_size = sampling["voltage"]["step_v"]

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

    battery_capacity = config["dataset"]["nominal_capacity_ah"]

    current_filter = pack_cfg.get("current_filter", {})
    min_c_rate = current_filter.get("min_c_rate", None)

    sample_filter_cfg = pack_cfg.get("sample_filter", {})
    skip_all_nan_samples = sample_filter_cfg.get("skip_all_nan_samples", True)
    check_lab_soc = sample_filter_cfg.get("check_lab_soc", True)

    voltage_regex = columns["pack_cell_voltage_regex"]
    temperature_prefix = columns["pack_cell_temperature_prefix"]

    first_timeseries = pack_data[0]["timeseries_data"]

    voltage_columns = [
        col for col in first_timeseries.columns
        if re.fullmatch(voltage_regex, col)
    ]

    voltage_sort_cfg = pack_cfg.get("voltage", {})
    voltage_columns = maybe_sort_columns(
        voltage_columns,
        sort_numeric_suffix=voltage_sort_cfg.get("sort_numeric_suffix", False),
    )

    temperature_columns = [
        col for col in first_timeseries.columns
        if col.startswith(temperature_prefix)
    ]

    temp_cfg = pack_cfg.get("temperature", {})
    temperature_columns = maybe_sort_columns(
        temperature_columns,
        sort_numeric_suffix=temp_cfg.get("sort_numeric_suffix", False),
    )

    n_cycles = len(pack_data)
    n_cells = get_pack_cell_count(
        config=config,
        voltage_columns=voltage_columns,
        temperature_columns=temperature_columns,
    )

    if n_cells == 0:
        print("Available columns in the first pack cycle:")
        print(list(first_timeseries.columns))
        raise RuntimeError(
            "No pack cell voltage/temperature channels were detected. "
            "Please check pack_cell_voltage_regex, pack_cell_temperature_prefix, "
            "and fixed_temperature_column in the YAML config."
        )

    all_data_soh = []
    all_lab_soh = []
    all_data_soc = []
    all_lab_soc = []
    sample_indices = []
    real_pack_soh = []

    print(f"Pack cycles: {n_cycles}")
    print(f"Voltage channels: {len(voltage_columns)}")
    print(f"Temperature channels: {len(temperature_columns)}")
    print(f"Used cells: {n_cells}")
    print(f"current_filter mode: {current_filter.get('mode', 'positive_charge')}")
    print(f"min_current_a: {current_filter.get('min_current_a', 0.0)}")
    print(f"flip_sign_after_filter: {current_filter.get('flip_sign_after_filter', False)}")
    print(f"min_c_rate: {min_c_rate}")
    print(f"skip_all_nan_samples: {skip_all_nan_samples}")
    print(f"check_lab_soc: {check_lab_soc}")
    print(
        "crop_to_voltage_window:",
        pack_cfg.get("voltage_filter", {}).get("crop_to_voltage_window", True),
    )

    for charge_number in range(0, len(pack_data)):
        print(f"Processing charge_number {charge_number} for data collection")

        for cell_number in range(0, n_cells):
            voltage_name = voltage_columns[cell_number]
            temperature_name = get_temperature_name(
                config=config,
                temperature_columns=temperature_columns,
                cell_number=cell_number,
            )

            key_columns = [
                columns["time"],
                voltage_name,
                columns["current"],
                columns["charge_capacity"],
                temperature_name,
                columns["soc"],
                columns["pack_voltage"],
            ]

            key_columns = list(dict.fromkeys(key_columns))

            missing_columns = [
                col for col in key_columns
                if col not in pack_data[charge_number]["timeseries_data"].columns
            ]

            if missing_columns:
                raise KeyError(
                    f"Missing columns in pack timeseries: {missing_columns}"
                )

            cycle_timeseries = (
                pack_data[charge_number]["timeseries_data"][key_columns]
                .copy()
            )

            cycle_timeseries = apply_pack_downsampling(config, cycle_timeseries)

            cycle_metadata = pack_data[charge_number]["constant_metadata"].copy()

            if "cycle" not in cycle_metadata.columns:
                cycle_metadata["cycle"] = charge_number

            cycle_metadata["DR"] = np.nan
            cycle_metadata["initial_capacity (Ah)"] = np.nan

            delta_q = np.nan

            for col in key_columns:
                cycle_timeseries[col] = pd.to_numeric(
                    cycle_timeseries[col],
                    errors="coerce",
                )

            cycle_timeseries = apply_current_filter(config, cycle_timeseries)

            if cycle_timeseries.shape[0] == 0:
                continue

            skip_by_c_rate = False

            if min_c_rate is not None:
                max_c_rate = (
                    cycle_timeseries[columns["current"]] / battery_capacity
                ).max()
                skip_by_c_rate = max_c_rate < min_c_rate

            if skip_by_c_rate:
                continue

            cycle_timeseries = filter_voltage_window(
                config=config,
                cycle_timeseries=cycle_timeseries,
                voltage_name=voltage_name,
                target_voltage_sequence=voltage_range,
            )

            if cycle_timeseries is None or cycle_timeseries.shape[0] == 0:
                continue

            voltage_jump_cfg = pack_cfg.get("abnormal_voltage_jump_filter", {})
            jump_filter_enabled = voltage_jump_cfg.get("enabled", False)

            if jump_filter_enabled:
                max_jump = voltage_jump_cfg.get("max_cell_voltage_jump_v", 0.02)

                if (cycle_timeseries[voltage_name].diff() > max_jump).any():
                    continue

            soc_samples = extract_soc_samples(
                cycle_timeseries=cycle_timeseries,
                cycle_metadata=cycle_metadata,
                window_size=soc_window_size,
                num_samples=soc_num_samples,
                voltage_name=voltage_name,
                temperature_name=temperature_name,
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
                voltage_name=voltage_name,
                temperature_name=temperature_name,
                step_size=step_size,
                pad_short_windows=soh_pad_short_windows,
                unique_start_indices=soh_unique_start_indices,
                time_transform=soh_time_transform,
            )

            if len(soc_samples) == 0 or len(soh_samples) == 0:
                continue

            soh_all_nan = all(
                sample is None or np.all(np.isnan(sample))
                for sample in soh_samples
            )

            soc_all_nan = all(
                sample is None or np.all(np.isnan(sample))
                for sample in soc_samples
            )

            if skip_all_nan_samples and (soh_all_nan or soc_all_nan):
                continue

            data_soh, lab_soh = data_load_soh(soh_samples)
            data_soc, lab_soc = data_load_soc(soc_samples)

            lab_soc_invalid = (
                np.any(np.isnan(lab_soc))
                or np.any(np.isinf(lab_soc))
            )

            if check_lab_soc and lab_soc_invalid:
                continue

            all_data_soh.append(data_soh)
            all_lab_soh.append(lab_soh)
            all_data_soc.append(data_soc)
            all_lab_soc.append(lab_soc)
            sample_indices.append((charge_number, cell_number))

        cycle_metadata_for_real = pack_data[charge_number]["constant_metadata"]

        if "SOH" in cycle_metadata_for_real.columns:
            real_pack_soh.append(cycle_metadata_for_real["SOH"].iloc[0] * 100)
        else:
            real_pack_soh.append(np.nan)

        if (charge_number + 1) % 20 == 0:
            print(
                f"Collected cycles: {charge_number + 1}/{n_cycles}, "
                f"valid samples: {len(sample_indices)}"
            )

    return {
        "all_data_soh": all_data_soh,
        "all_lab_soh": all_lab_soh,
        "all_data_soc": all_data_soc,
        "all_lab_soc": all_lab_soc,
        "sample_indices": np.asarray(sample_indices, dtype=int),
        "real_pack_soh": np.asarray(real_pack_soh, dtype=np.float32),
        "voltage_columns": np.asarray(voltage_columns),
        "temperature_columns": np.asarray(temperature_columns),
        "n_cycles": n_cycles,
        "n_cells": n_cells,
        "soh_num_samples": soh_num_samples,
        "soc_num_samples": soc_num_samples,
    }


def predict_pack_states(config, checkpoint, collected, device):
    if "norm_params" not in checkpoint:
        raise KeyError(
            "norm_params was not found in checkpoint. "
            "Please retrain the model with the updated train_cell_model.py."
        )

    norm_params = checkpoint["norm_params"]

    soh_max = norm_params["soh_max"]
    soh_min = norm_params["soh_min"]
    soc_max = norm_params["soc_max"]
    soc_min = norm_params["soc_min"]

    model = build_model(config, device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n_cycles = collected["n_cycles"]
    n_cells = collected["n_cells"]
    soh_num_samples = collected["soh_num_samples"]
    soc_num_samples = collected["soc_num_samples"]

    soh_pred_all = np.full(
        (n_cycles, n_cells, soh_num_samples),
        np.nan,
        dtype=np.float32,
    )

    soc_pred_all = np.full(
        (n_cycles, n_cells, soc_num_samples),
        np.nan,
        dtype=np.float32,
    )

    total_samples = len(collected["all_data_soh"])
    batch_size = config["pack_inference"]["batch_size"]

    require_full_batch = config["pack_inference"].get("require_full_batch", False)

    if total_samples == 0:
        raise RuntimeError(
            "No valid pack samples found. "
            "Please check pack filtering and sample_filter settings."
        )

    n_batches = (total_samples + batch_size - 1) // batch_size

    print(f"Total valid pack samples: {total_samples}")
    print(f"Prediction batches: {n_batches}")
    print(f"require_full_batch: {require_full_batch}")

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)

            if require_full_batch and (end_idx - start_idx) != batch_size:
                print(
                    f"Skipping incomplete batch {batch_idx + 1}/{n_batches}: "
                    f"samples {start_idx} to {end_idx - 1}"
                )
                continue

            batch_data_soh_array = transfer_arr(
                collected["all_data_soh"][start_idx:end_idx]
            )

            batch_lab_soh_array = transfer_arr(
                collected["all_lab_soh"][start_idx:end_idx]
            )

            batch_data_soc_array = transfer_arr(
                collected["all_data_soc"][start_idx:end_idx]
            )

            batch_lab_soc_array = transfer_arr(
                collected["all_lab_soc"][start_idx:end_idx]
            )

            normalized_batch_data_soh = normalize_with_params(
                batch_data_soh_array,
                soh_max,
                soh_min,
            )

            normalized_batch_data_soc = normalize_with_params(
                batch_data_soc_array,
                soc_max,
                soc_min,
            )

            x_soh = torch.FloatTensor(normalized_batch_data_soh).to(device)
            y_soh = torch.FloatTensor(batch_lab_soh_array).to(device)

            x_soc = torch.FloatTensor(normalized_batch_data_soc).to(device)
            y_soc = torch.FloatTensor(batch_lab_soc_array).to(device)

            soh_pred, soc_pred = model(
                (x_soh, y_soh),
                (x_soc, y_soc),
            )

            soh_results = soh_pred.cpu().detach().numpy() * 100
            soc_results = soc_pred.cpu().detach().numpy() * 100

            for i in range(end_idx - start_idx):
                charge_idx, cell_idx = collected["sample_indices"][start_idx + i]
                soh_pred_all[charge_idx][cell_idx] = soh_results[i].reshape(-1)
                soc_pred_all[charge_idx][cell_idx] = soc_results[i].reshape(-1)

            del x_soh, y_soh, x_soc, y_soc, soh_pred, soc_pred

            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(f"Predicted batch {batch_idx + 1}/{n_batches}")

    return soh_pred_all, soc_pred_all


def get_device(config):
    if config["training"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["training"]["device"])

    if device.type == "cuda":
        torch.cuda.set_device(config["training"].get("cuda_device", 0))

    return device


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

    device = get_device(config)
    print(f"Using device: {device}")

    checkpoint_path = (
        ROOT
        / config["output"]["checkpoint_dir"]
        / config["output"]["best_model_name"]
    )

    with open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=device)

    print(f"Loaded checkpoint: {checkpoint_path}")

    pack_path = ROOT / config["dataset"]["pack_data_path"]
    pack_data = load_pickle(pack_path)

    print(f"Loaded pack data: {pack_path}")

    collected = collect_pack_samples(
        config=config,
        pack_data=pack_data,
    )

    soh_pred_all, soc_pred_all = predict_pack_states(
        config=config,
        checkpoint=checkpoint,
        collected=collected,
        device=device,
    )

    prediction_file = ROOT / config["output"]["pack_prediction_file"]
    prediction_file.parent.mkdir(parents=True, exist_ok=True)

    save_npz(
        prediction_file,
        soh_pred=soh_pred_all,
        soc_pred=soc_pred_all,
        real_pack_soh=collected["real_pack_soh"],
        sample_indices=collected["sample_indices"],
        voltage_columns=collected["voltage_columns"],
        temperature_columns=collected["temperature_columns"],
    )

    print("Pack-state inference finished.")
    print(f"Saved to: {prediction_file}")


if __name__ == "__main__":
    main()
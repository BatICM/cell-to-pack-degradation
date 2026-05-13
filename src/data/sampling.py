# -*- coding: utf-8 -*-
"""
Sample extraction for joint SOC/SOH estimation.
"""

import numpy as np
import pandas as pd

from src.data.interpolation import enforce_monotonicity, custom_interp1d


def _make_nan_samples(num_samples, window_size, n_features):
    return [
        np.full((window_size, n_features), np.nan)
        for _ in range(num_samples)
    ]


def _get_start_indices(data_length, window_size, num_samples, unique_start_indices=False):
    starts = np.linspace(
        0,
        data_length - window_size,
        num=num_samples,
        dtype=int,
    )

    if unique_start_indices:
        starts = np.unique(starts)

    return starts


def extract_soc_samples(
    cycle_timeseries: pd.DataFrame,
    cycle_metadata: pd.DataFrame,
    window_size: int,
    num_samples: int,
    voltage_name: str,
    temperature_name: str,
    pad_short_windows: bool = False,
    unique_start_indices: bool = False,
):
    """Extract short-timescale SOC samples from one charging cycle."""
    samples = np.vstack([
        cycle_timeseries["SOC"].values,
        cycle_timeseries["charge_capacity (Ah)"].values,
        cycle_timeseries[voltage_name].values,
        np.full(cycle_timeseries.shape[0], cycle_metadata["SOH"].iloc[0]),
        cycle_timeseries["current (A)"].values / cycle_metadata["rate_capacity (Ah)"].iloc[0],
        cycle_timeseries[temperature_name].values / 60.0,
        np.full(cycle_timeseries.shape[0], cycle_metadata["rate_capacity (Ah)"].iloc[0]),
    ]).T

    soc_samples = []
    data_length = samples.shape[0]

    if data_length < window_size:
        if pad_short_windows:
            return _make_nan_samples(num_samples, window_size, samples.shape[1])
        return soc_samples

    starts = _get_start_indices(
        data_length=data_length,
        window_size=window_size,
        num_samples=num_samples,
        unique_start_indices=unique_start_indices,
    )

    for start in starts:
        end = start + window_size
        window = samples[start:end, :].copy()
        window[:, 1] -= window[0, 1]
        soc_samples.append(window)

    if pad_short_windows:
        while len(soc_samples) < num_samples:
            soc_samples.append(np.full((window_size, samples.shape[1]), np.nan))

    return soc_samples[:num_samples]


def extract_soh_samples(
    cycle_timeseries: pd.DataFrame,
    cycle_metadata: pd.DataFrame,
    delta_q: float,
    voltage_range: np.ndarray,
    window_size: int,
    num_samples: int,
    voltage_name: str,
    temperature_name: str,
    step_size: float,
    pad_short_windows: bool = True,
    unique_start_indices: bool = False,
    time_transform: str = "log10",
):
    """Extract voltage-aligned SOH samples from one charging cycle."""
    if cycle_timeseries.shape[0] == 0:
        if pad_short_windows:
            return _make_nan_samples(num_samples, window_size, 12)
        return []

    voltage_col = cycle_timeseries[voltage_name].values.copy()
    voltage_sorted = enforce_monotonicity(voltage_col)

    min_target_voltage = voltage_range.min()
    max_target_voltage = voltage_range.max()

    interp_min_voltage = max(cycle_timeseries[voltage_name].min(), min_target_voltage)
    interp_max_voltage = min(cycle_timeseries[voltage_name].max(), max_target_voltage)

    target_voltage = np.arange(
        interp_min_voltage,
        interp_max_voltage + step_size,
        step_size,
    )

    if len(target_voltage) == 0:
        if pad_short_windows:
            return _make_nan_samples(num_samples, window_size, 12)
        return []

    interp_functions = [
        custom_interp1d(voltage_sorted, cycle_timeseries[col].values)
        for col in cycle_timeseries.columns
    ]

    interpolated_data = pd.DataFrame({
        col: interp_func(target_voltage)
        for col, interp_func in zip(cycle_timeseries.columns, interp_functions)
    })

    interpolated_data["charge_capacity (Ah)"] -= interpolated_data["charge_capacity (Ah)"].iloc[0]
    interpolated_data["time (s)"] -= interpolated_data["time (s)"].iloc[0]
    interpolated_data["dQ"] = interpolated_data["charge_capacity (Ah)"].diff().fillna(0)

    if time_transform == "log10":
        interpolated_data["time (s)"] = (
            np.log10(interpolated_data["time (s)"].astype(np.float32))
            .replace([np.inf, -np.inf], 0)
        )
    elif time_transform in ["none", None]:
        pass
    else:
        raise ValueError(f"Unsupported time_transform: {time_transform}")

    samples = np.vstack([
        np.full(len(interpolated_data), cycle_metadata["SOH"].iloc[0]),
        np.full(len(interpolated_data), delta_q),
        interpolated_data["charge_capacity (Ah)"],
        interpolated_data["time (s)"],
        interpolated_data["dQ"],
        interpolated_data["current (A)"] / cycle_metadata["rate_capacity (Ah)"].iloc[0],
        interpolated_data[temperature_name] / 60.0,
        interpolated_data[voltage_name],
        np.full(len(interpolated_data), cycle_metadata["cycle"].iloc[0]),
        np.full(len(interpolated_data), cycle_metadata["DR"].iloc[0]),
        np.full(len(interpolated_data), cycle_metadata["initial_capacity (Ah)"].iloc[0]),
        np.full(len(interpolated_data), cycle_metadata["rate_capacity (Ah)"].iloc[0]),
    ]).T

    soh_samples = []
    data_length = samples.shape[0]

    if data_length < window_size:
        if pad_short_windows:
            return _make_nan_samples(num_samples, window_size, samples.shape[1])
        return soh_samples

    starts = _get_start_indices(
        data_length=data_length,
        window_size=window_size,
        num_samples=num_samples,
        unique_start_indices=unique_start_indices,
    )

    for start in starts:
        end = start + window_size
        window = samples[start:end, :].copy()
        window[:, 2] -= window[0, 2]
        window[:, 3] -= window[0, 3]
        window[:, 4] = np.diff(window[:, 2], prepend=0)
        soh_samples.append(window)

    if pad_short_windows:
        while len(soh_samples) < num_samples:
            soh_samples.append(np.full((window_size, samples.shape[1]), np.nan))

    return soh_samples[:num_samples]


def data_load_soc(samples):
    """Convert SOC sample windows into model inputs and labels."""
    data = []
    labels = []

    for sample in samples:
        labels.append(
            np.array([
                sample[-1, 0],
                sample[0, 6],
                np.mean(sample[:, 4]),
                np.mean(sample[:, 5]),
            ], dtype=np.float32)
        )

        data.append(sample[:, [1, 2]].astype(np.float32))

    return data, labels


def data_load_soh(samples):
    """Convert SOH sample windows into model inputs and labels."""
    data = []
    labels = []

    for sample in samples:
        labels.append(
            np.array([
                sample[0, 0],
                sample[0, 1],
                sample[0, 9],
                sample[0, 8],
                sample[0, 10],
                sample[0, 11],
                np.mean(sample[:, 5]),
                np.mean(sample[:, 6]),
            ], dtype=np.float32)
        )

        data.append(sample[:, [2, 3, 4, 7]].astype(np.float32))

    return data, labels
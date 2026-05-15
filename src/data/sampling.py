# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from src.data.interpolation import enforce_monotonicity, custom_interp1d


TIME_COL = "time(s)"
CURRENT_COL = "current(A)"
CHARGE_CAPACITY_COL = "charge_capacity(Ah)"
SOC_COL = "SOC"

RATE_CAPACITY_COL = "rate_capacity (Ah)"
CYCLE_COL = "cycle"
INITIAL_CAPACITY_COL = "initial_capacity (Ah)"
DR_COL = "DR"
SOH_COL = "SOH"


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
  
    samples = np.vstack([
        cycle_timeseries[SOC_COL].values,                                    # 0: SOC
        cycle_timeseries[CHARGE_CAPACITY_COL].values,                        # 1: charge capacity
        cycle_timeseries[voltage_name].values,                               # 2: voltage
        np.full(cycle_timeseries.shape[0], cycle_metadata[SOH_COL].iloc[0]), # 3: SOH
        cycle_timeseries[CURRENT_COL].values
        / cycle_metadata[RATE_CAPACITY_COL].iloc[0],                         # 4: C-rate
        cycle_timeseries[temperature_name].values / 60.0,                    # 5: normalized temperature
        np.full(
            cycle_timeseries.shape[0],
            cycle_metadata[RATE_CAPACITY_COL].iloc[0],
        ),                                                                   # 6: rate capacity
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

    interpolated_data[CHARGE_CAPACITY_COL] -= interpolated_data[CHARGE_CAPACITY_COL].iloc[0]
    interpolated_data[TIME_COL] -= interpolated_data[TIME_COL].iloc[0]

    interpolated_data["dQ"] = interpolated_data[CHARGE_CAPACITY_COL].diff().fillna(0)

    if time_transform == "log10":
        interpolated_data[TIME_COL] = (
            np.log10(interpolated_data[TIME_COL].astype(np.float32))
            .replace([np.inf, -np.inf], 0)
        )
    elif time_transform in ["none", None]:
        pass
    else:
        raise ValueError(f"Unsupported time_transform: {time_transform}")

    samples = np.vstack([
        np.full(len(interpolated_data), cycle_metadata[SOH_COL].iloc[0]),               # 0: SOH
        np.full(len(interpolated_data), delta_q),                                       # 1: delta_q
        interpolated_data[CHARGE_CAPACITY_COL],                                        # 2: charge capacity
        interpolated_data[TIME_COL],                                                   # 3: time
        interpolated_data["dQ"],                                                       # 4: dQ
        interpolated_data[CURRENT_COL]
        / cycle_metadata[RATE_CAPACITY_COL].iloc[0],                                   # 5: C-rate
        interpolated_data[temperature_name] / 60.0,                                    # 6: normalized temperature
        interpolated_data[voltage_name],                                               # 7: voltage
        np.full(len(interpolated_data), cycle_metadata[CYCLE_COL].iloc[0]),             # 8: cycle
        np.full(len(interpolated_data), cycle_metadata[DR_COL].iloc[0]),                # 9: DR
        np.full(len(interpolated_data), cycle_metadata[INITIAL_CAPACITY_COL].iloc[0]),  # 10: initial capacity
        np.full(len(interpolated_data), cycle_metadata[RATE_CAPACITY_COL].iloc[0]),     # 11: rate capacity
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

        # sample[:, 2] is charge_capacity(Ah)
        window[:, 2] -= window[0, 2]

        # sample[:, 3] is time(s)
        window[:, 3] -= window[0, 3]

        # sample[:, 4] is dQ, recalculated locally in this SOH window
        window[:, 4] = np.diff(window[:, 2], prepend=0)

        soh_samples.append(window)

    if pad_short_windows:
        while len(soh_samples) < num_samples:
            soh_samples.append(np.full((window_size, samples.shape[1]), np.nan))

    return soh_samples[:num_samples]


def data_load_soc(samples):

    data = []
    labels = []

    for sample in samples:
        labels.append(
            np.array([
                sample[-1, 0],          # SOC label
                sample[0, 6],           # rate capacity
                np.mean(sample[:, 4]),  # mean C-rate
                np.mean(sample[:, 5]),  # mean normalized temperature
            ], dtype=np.float32)
        )

        # Input: charge capacity + voltage
        data.append(sample[:, [1, 2]].astype(np.float32))

    return data, labels


def data_load_soh(samples):
   
    data = []
    labels = []

    for sample in samples:
        labels.append(
            np.array([
                sample[0, 0],           # SOH
                sample[0, 1],           # delta_q
                sample[0, 9],           # DR
                sample[0, 8],           # cycle
                sample[0, 10],          # initial capacity
                sample[0, 11],          # rate capacity
                np.mean(sample[:, 5]),  # mean C-rate
                np.mean(sample[:, 6]),  # mean normalized temperature
            ], dtype=np.float32)
        )

        # Input: charge capacity + time + dQ + voltage
        data.append(sample[:, [2, 3, 4, 7]].astype(np.float32))

    return data, labels
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def remove_outliers_and_mean(data, iqr_multiplier=1.5):
    """IQR-based mean used for pack SOH aggregation."""
    data = np.array(data)

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    mean_value = np.mean(filtered_data)

    return mean_value


def reduce_soh_prediction(soh_pred, reduce_soh_windows=None):
    """
    Reduce SOH-window dimension if required.

    Parameters
    ----------
    soh_pred : np.ndarray
        SOH prediction, usually [n_cycles, n_cells, n_soh_windows].
    reduce_soh_windows : str or None
        None: keep original SOH-window dimension.
        "mean": average over SOH-window dimension.

    Returns
    -------
    np.ndarray
        Reduced or original SOH prediction.
    """
    soh_pred = np.array(soh_pred)

    if reduce_soh_windows in [None, "none", "None", "null", ""]:
        return soh_pred

    if reduce_soh_windows == "mean":
        if soh_pred.ndim == 3:
            return np.nanmean(soh_pred, axis=2)
        return soh_pred

    raise ValueError(f"Unsupported reduce_soh_windows: {reduce_soh_windows}")


def clean_soc_data(soc_pred, method="median_std", k=1.0):
    """
    Clean SOC prediction before pack SOH calculation.

    This matches the ESSH1 original logic:
        for each cycle, remove SOC values outside median ± k * std
        along the cell dimension for each SOC sample.
    """
    soc_pred = np.array(soc_pred)

    if method in [None, "none", "None", "null", ""]:
        return soc_pred

    if method != "median_std":
        raise ValueError(f"Unsupported SOC cleaning method: {method}")

    cleaned_soc = []

    for soc_matrix in soc_pred:
        soc_matrix = np.array(soc_matrix, dtype=float)

        medians = np.nanmedian(soc_matrix, axis=0)
        stds = np.nanstd(soc_matrix, axis=0)

        lower_bound = medians - k * stds
        upper_bound = medians + k * stds

        mask = (soc_matrix >= lower_bound) & (soc_matrix <= upper_bound)
        cleaned_matrix = np.where(mask, soc_matrix, np.nan)

        cleaned_soc.append(cleaned_matrix)

    return np.array(cleaned_soc)


def estimate_pack_soh_from_cell_states(
    soh_pred,
    soc_pred,
    reduce_soh_windows=None,
    iqr_multiplier=1.5,
    soc_cleaning_enabled=False,
    soc_cleaning_method="median_std",
    soc_cleaning_k=1.0,
):
    """
    Estimate pack SOH from reconstructed cell SOH/SOC states.

    Parameters
    ----------
    soh_pred : np.ndarray
        SOH prediction in %, shape:
        [n_cycles, n_cells, n_soh_windows] or [n_cycles, n_cells].
    soc_pred : np.ndarray
        SOC prediction in %, shape:
        [n_cycles, n_cells, n_soc_samples].
    reduce_soh_windows : str or None
        None keeps original ESSL1/ESSL2 logic.
        "mean" matches PBSL1 logic: SOH_estimation = nanmean(SOH, axis=2).
    iqr_multiplier : float
        IQR filtering multiplier. ESSL1/ESSL2 use 1.5; PBSL1 uses 2.5.
    soc_cleaning_enabled : bool
        If True, clean SOC before pack SOH calculation. Used by ESSH1.
    soc_cleaning_method : str
        SOC cleaning method. Currently supports "median_std".
    soc_cleaning_k : float
        Multiplier for median ± k * std SOC cleaning.
    """
    SOH_estimation = reduce_soh_prediction(
        soh_pred,
        reduce_soh_windows=reduce_soh_windows,
    )

    SOC = np.array(soc_pred)

    if soc_cleaning_enabled:
        SOC = clean_soc_data(
            SOC,
            method=soc_cleaning_method,
            k=soc_cleaning_k,
        )

    NUM_SAMPLES = SOC.shape[2]
    voltage_columns = range(SOC.shape[1])

    Pack_SOH = []
    min_SOH = []
    max_SOH = []
    min_chargeable_cell = []
    min_dischargeable_cell = []
    min_soh_cell = []

    for i in range(len(SOH_estimation)):

        Pack_SOH_samples = []
        min_current_idx = np.nan
        min_remain_idx = np.nan

        for k in range(NUM_SAMPLES):
            cell_current_level = []
            cell_remain_level = []

            for j in range(len(voltage_columns)):

                current = SOH_estimation[i][j] * (SOC[i][j][k]) / 100
                remain = SOH_estimation[i][j] * (100 - SOC[i][j][k]) / 100

                cell_current_level.append(current)
                cell_remain_level.append(remain)

            current_array = np.array(cell_current_level)
            remain_array = np.array(cell_remain_level)

            if np.all(np.isnan(current_array)):
                min_current_cell = np.nan
                current_idx = np.nan
            else:
                min_current_cell = np.nanmin(current_array)
                flat_current_idx = np.nanargmin(current_array)
                current_idx = np.unravel_index(
                    flat_current_idx,
                    current_array.shape,
                )[0]

            if np.all(np.isnan(remain_array)):
                min_remain_cell = np.nan
                remain_idx = np.nan
            else:
                min_remain_cell = np.nanmin(remain_array)
                flat_remain_idx = np.nanargmin(remain_array)
                remain_idx = np.unravel_index(
                    flat_remain_idx,
                    remain_array.shape,
                )[0]

            min_current_idx = current_idx
            min_remain_idx = remain_idx

            Pack_SOH_samples.append(min_current_cell + min_remain_cell)

        soh_array = np.array(SOH_estimation[i])

        if np.all(np.isnan(soh_array)):
            min_soh = np.nan
            max_soh = np.nan
            cell_idx = np.nan
        else:
            min_soh = np.nanmin(soh_array)
            max_soh = np.nanmax(soh_array)
            flat_soh_idx = np.nanargmin(soh_array)
            cell_idx = np.unravel_index(
                flat_soh_idx,
                soh_array.shape,
            )[0]

        min_SOH.append(min_soh)
        max_SOH.append(max_soh)
        min_soh_cell.append(cell_idx)

        Pack_SOH.append(
            remove_outliers_and_mean(
                Pack_SOH_samples,
                iqr_multiplier=iqr_multiplier,
            )
        )

        min_chargeable_cell.append(min_remain_idx)
        min_dischargeable_cell.append(min_current_idx)

    return {
        "pack_soh": np.array(Pack_SOH),
        "min_cell_soh": np.array(min_SOH),
        "max_cell_soh": np.array(max_SOH),
        "min_soh_cell": np.array(min_soh_cell),
        "min_chargeable_cell": np.array(min_chargeable_cell),
        "min_dischargeable_cell": np.array(min_dischargeable_cell),
    }


def compute_error_metrics(pred, target):
    """
    Original error calculation style.

    pred and target are both in %, so:
    original MAE = mean(abs(target - pred)) / 100
    """
    pred = np.array(pred)
    target = np.array(target)

    absolute_error = np.abs(target - pred)
    mae_fraction = np.nanmean(absolute_error) / 100

    squared_error = (target - pred) ** 2
    rmse_fraction = np.sqrt(np.nanmean(squared_error)) / 100

    valid_mask = np.isfinite(absolute_error)

    return {
        "mae_fraction": float(mae_fraction),
        "rmse_fraction": float(rmse_fraction),
        "mae_percent": float(mae_fraction * 100),
        "rmse_percent": float(rmse_fraction * 100),
        "n_valid": int(np.sum(valid_mask)),
    }


def build_pack_estimation_table(real_pack_soh, estimated):
    return pd.DataFrame({
        "Index": range(len(real_pack_soh)),
        "Pack_SOH": estimated["pack_soh"],
        "Min_cells": estimated["min_cell_soh"],
        "Real": real_pack_soh,
        "Max_cells": estimated["max_cell_soh"],
        "min_soh_cell": estimated["min_soh_cell"],
        "min_chargeable_cell": estimated["min_chargeable_cell"],
        "min_dischargeable_cell": estimated["min_dischargeable_cell"],
    })
# -*- coding: utf-8 -*-
"""
Evaluation utilities for joint SOC/SOH estimation.
"""

from typing import Dict

import numpy as np
import torch

from src.training.losses import compute_joint_loss


def nan_mae(pred: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def nan_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((pred[mask] - target[mask]) ** 2)))


def nan_mape(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    mask = np.isfinite(pred) & np.isfinite(target) & (np.abs(target) > eps)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((pred[mask] - target[mask]) / target[mask])) * 100)


def compute_dr_prediction_and_label(
    soh_pred_2d: torch.Tensor,
    y_soh: torch.Tensor,
    dr_mode: str = "capacity_difference",
    dr_target_scale: float = 1.0,
    eps: float = 1e-8,
):
    """
    Compute DR prediction and label using the same mode as training.

    y_soh columns:
        0: SOH
        1: delta_q
        2: DR
        3: cycle
        4: initial capacity
        5: rated capacity
        6: mean current rate
        7: mean temperature
    """
    if dr_mode in [None, "capacity_difference", "original"]:
        pred_dr = (
            y_soh[:, :, 4]
            - soh_pred_2d * y_soh[:, :, 5]
        ) / (y_soh[:, :, 3] + eps)

        target_dr = y_soh[:, :, 2]

    elif dr_mode == "normalized_capacity_ratio":
        pred_dr = (
            y_soh[:, :, 4] / y_soh[:, :, 5]
            - soh_pred_2d
        ) / (y_soh[:, :, 3] + eps)

        target_dr = y_soh[:, :, 2] / dr_target_scale

    else:
        raise ValueError(f"Unsupported dr_mode: {dr_mode}")

    return pred_dr, target_dr


def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    loss_weights: Dict[str, float],
    dataset_name: str = "Dataset",
    print_results: bool = True,
    dr_mode: str = "capacity_difference",
    dr_target_scale: float = 1.0,
) -> Dict[str, object]:
    """Evaluate model and return predictions, labels, losses, and metrics."""
    model.eval()

    soh_preds, soh_labels = [], []
    soc_preds, soc_labels = [], []
    dr_preds, dr_labels = [], []

    loss_records = {
        "total": [],
        "soh": [],
        "soc": [],
        "soh_soc": [],
        "dr": [],
    }

    with torch.no_grad():
        for (x_soh, y_soh), (x_soc, y_soc) in data_loader:
            x_soh = x_soh.to(device)
            y_soh = y_soh.to(device)
            x_soc = x_soc.to(device)
            y_soc = y_soc.to(device)

            soh_pred, soc_pred = model((x_soh, y_soh), (x_soc, y_soc))

            _, loss_dict = compute_joint_loss(
                soh_pred=soh_pred,
                soc_pred=soc_pred,
                y_soh=y_soh,
                y_soc=y_soc,
                loss_weights=loss_weights,
                dr_mode=dr_mode,
                dr_target_scale=dr_target_scale,
            )

            for key in loss_records:
                loss_records[key].append(float(loss_dict[key].cpu().item()))

            soh_pred_2d = soh_pred.squeeze(-1)
            soc_pred_2d = soc_pred.squeeze(-1)

            pred_dr, target_dr = compute_dr_prediction_and_label(
                soh_pred_2d=soh_pred_2d,
                y_soh=y_soh,
                dr_mode=dr_mode,
                dr_target_scale=dr_target_scale,
            )

            soh_preds.append(soh_pred_2d.cpu().numpy())
            soh_labels.append(y_soh[:, :, 0].cpu().numpy())

            soc_preds.append(soc_pred_2d.cpu().numpy())
            soc_labels.append(y_soc[:, :, 0].cpu().numpy())

            dr_preds.append(pred_dr.cpu().numpy())
            dr_labels.append(target_dr.cpu().numpy())

    soh_preds = np.concatenate(soh_preds, axis=0)
    soh_labels = np.concatenate(soh_labels, axis=0)

    soc_preds = np.concatenate(soc_preds, axis=0)
    soc_labels = np.concatenate(soc_labels, axis=0)

    dr_preds = np.concatenate(dr_preds, axis=0)
    dr_labels = np.concatenate(dr_labels, axis=0)

    losses = {
        key: float(np.mean(values)) if len(values) > 0 else np.nan
        for key, values in loss_records.items()
    }

    metrics = {
        "soh_mae": nan_mae(soh_preds, soh_labels),
        "soh_rmse": nan_rmse(soh_preds, soh_labels),
        "soh_mape": nan_mape(soh_preds, soh_labels),
        "soc_mae": nan_mae(soc_preds, soc_labels),
        "soc_rmse": nan_rmse(soc_preds, soc_labels),
        "soc_mape": nan_mape(soc_preds, soc_labels),
        "dr_mae": nan_mae(dr_preds, dr_labels),
    }

    if print_results:
        print(f"\n{dataset_name} Evaluation")
        print("-" * 60)
        print(f"DR mode    : {dr_mode}")
        print(f"DR scale   : {dr_target_scale}")
        print(f"Total Loss : {losses['total']:.6f}")
        print(f"SOH MAE    : {metrics['soh_mae']:.6f}")
        print(f"SOH RMSE   : {metrics['soh_rmse']:.6f}")
        print(f"SOH MAPE % : {metrics['soh_mape']:.4f}")
        print(f"SOC MAE    : {metrics['soc_mae']:.6f}")
        print(f"SOC RMSE   : {metrics['soc_rmse']:.6f}")
        print(f"SOC MAPE % : {metrics['soc_mape']:.4f}")
        print(f"DR MAE     : {metrics['dr_mae']:.6f}")

    return {
        "predictions": {
            "soh": soh_preds,
            "soc": soc_preds,
            "dr": dr_preds,
        },
        "labels": {
            "soh": soh_labels,
            "soc": soc_labels,
            "dr": dr_labels,
        },
        "losses": losses,
        "metrics": metrics,
    }
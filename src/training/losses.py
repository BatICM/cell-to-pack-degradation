# -*- coding: utf-8 -*-
"""
Loss functions for joint SOC/SOH estimation.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


def compute_dr_loss(
    soh_pred_2d: torch.Tensor,
    y_soh: torch.Tensor,
    criterion: nn.Module,
    dr_mode: str = "capacity_difference",
    dr_target_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute degradation-rate loss.

    y_soh columns:
        0: SOH
        1: delta_q
        2: DR
        3: cycle
        4: initial capacity
        5: rated capacity
        6: mean current rate
        7: mean temperature

    Supported modes:
        capacity_difference:
            pred_dr = (initial_capacity - predicted_SOH * rated_capacity) / cycle
            target  = DR

        normalized_capacity_ratio:
            pred_dr = (initial_capacity / rated_capacity - predicted_SOH) / cycle
            target  = DR / dr_target_scale
    """
    if dr_mode in [None, "capacity_difference", "original"]:
        pred_dr = (
            y_soh[:, :, 4]
            - soh_pred_2d * y_soh[:, :, 5]
        ) / (y_soh[:, :, 3] + 1e-8)

        target_dr = y_soh[:, :, 2]

    elif dr_mode == "normalized_capacity_ratio":
        pred_dr = (
            y_soh[:, :, 4] / y_soh[:, :, 5]
            - soh_pred_2d
        ) / (y_soh[:, :, 3] + 1e-8)

        target_dr = y_soh[:, :, 2] / dr_target_scale

    else:
        raise ValueError(f"Unsupported dr_mode: {dr_mode}")

    loss_dr = criterion(pred_dr, target_dr)

    return loss_dr, pred_dr, target_dr


def compute_joint_loss(
    soh_pred: torch.Tensor,
    soc_pred: torch.Tensor,
    y_soh: torch.Tensor,
    y_soc: torch.Tensor,
    loss_weights: Dict[str, float],
    criterion: nn.Module = None,
    dr_mode: str = "capacity_difference",
    dr_target_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute joint loss:
    SOH loss + SOC loss + SOH-SOC capacity-change loss + DR loss.
    """
    if criterion is None:
        criterion = nn.MSELoss()

    # SOH loss
    soh_pred_flat = soh_pred.view(-1)
    soh_label_flat = y_soh[:, :, 0].view(-1)
    loss_soh = criterion(soh_pred_flat, soh_label_flat)

    # DR loss
    soh_pred_2d = soh_pred.squeeze(-1)

    loss_dr, pred_dr, target_dr = compute_dr_loss(
        soh_pred_2d=soh_pred_2d,
        y_soh=y_soh,
        criterion=criterion,
        dr_mode=dr_mode,
        dr_target_scale=dr_target_scale,
    )

    # SOC loss
    soc_pred_flat = soc_pred.view(-1)
    soc_label_flat = y_soc[:, :, 0].view(-1)
    loss_soc = criterion(soc_pred_flat, soc_label_flat)

    # SOH x SOC capacity-change consistency loss
    soh_pred_mean = soh_pred.mean(dim=1).squeeze()

    if soc_pred.size(1) >= 2:
        soc_changes = soc_pred[:, -1, :].squeeze() - soc_pred[:, 0, :].squeeze()
        delta_q_label = y_soh[:, :, 1].mean(dim=1).squeeze()

        loss_soh_soc = criterion(
            soh_pred_mean.squeeze() * soc_changes,
            delta_q_label,
        )
    else:
        loss_soh_soc = torch.tensor(0.0).to(soh_pred.device)

    total_loss = (
        loss_weights["soh"] * loss_soh
        + loss_weights["soc"] * loss_soc
        + loss_weights["soh_soc"] * loss_soh_soc
        + loss_weights["dr"] * loss_dr
    )

    loss_dict = {
        "total": total_loss.detach(),
        "soh": loss_soh.detach(),
        "soc": loss_soc.detach(),
        "soh_soc": loss_soh_soc.detach(),
        "dr": loss_dr.detach(),
    }

    return total_loss, loss_dict
# -*- coding: utf-8 -*-
"""
Training loop for the joint SOC/SOH model.
"""

from pathlib import Path
from typing import Dict, Optional

import torch

from src.training.losses import compute_joint_loss


def reset_dataloader_generator(train_loader, seed: int, epoch: int):
    """Reset DataLoader generator as in the original script."""
    generator = getattr(train_loader, "generator", None)

    if generator is None:
        generator = getattr(train_loader, "fixed_generator", None)

    if generator is not None:
        generator.manual_seed(seed + epoch)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    loss_weights: Dict[str, float],
    dr_mode: str = "capacity_difference",
    dr_target_scale: float = 1.0,
):
    """Train model for one epoch."""
    model.train()

    running = {
        "total": 0.0,
        "soh": 0.0,
        "soc": 0.0,
        "soh_soc": 0.0,
        "dr": 0.0,
    }

    n_batches = 0

    for (x_soh, y_soh), (x_soc, y_soc) in train_loader:
        optimizer.zero_grad()

        x_soh = x_soh.to(device)
        y_soh = y_soh.to(device)
        x_soc = x_soc.to(device)
        y_soc = y_soc.to(device)

        soh_pred, soc_pred = model(
            (x_soh, y_soh),
            (x_soc, y_soc),
        )

        loss, loss_dict = compute_joint_loss(
            soh_pred=soh_pred,
            soc_pred=soc_pred,
            y_soh=y_soh,
            y_soc=y_soc,
            loss_weights=loss_weights,
            dr_mode=dr_mode,
            dr_target_scale=dr_target_scale,
        )

        loss.backward()
        optimizer.step()

        running["total"] += float(loss.item())
        running["soh"] += float(loss_dict["soh"].cpu().item())
        running["soc"] += float(loss_dict["soc"].cpu().item())
        running["soh_soc"] += float(loss_dict["soh_soc"].cpu().item())
        running["dr"] += float(loss_dict["dr"].cpu().item())

        n_batches += 1

    if n_batches == 0:
        return {key: float("nan") for key in running}

    return {
        key: value / n_batches
        for key, value in running.items()
    }


def train_model(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    loss_weights: Dict[str, float],
    num_epochs: int,
    checkpoint_path: str,
    early_stopping_patience: Optional[int] = 20,
    extra_state: Optional[Dict] = None,
    seed: int = 42,
    dr_mode: str = "capacity_difference",
    dr_target_scale: float = 1.0,
):
    """
    Train model with scheduler, early stopping, and checkpoint saving.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    patience_counter = 0

    history = {
        "total": [],
        "soh": [],
        "soc": [],
        "soh_soc": [],
        "dr": [],
        "lr": [],
    }

    print(f"DR loss mode: {dr_mode}")
    print(f"DR target scale: {dr_target_scale}")

    for epoch in range(num_epochs):
        reset_dataloader_generator(train_loader, seed=seed, epoch=epoch)

        epoch_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_weights=loss_weights,
            dr_mode=dr_mode,
            dr_target_scale=dr_target_scale,
        )

        avg_loss = epoch_loss["total"]

        history["total"].append(avg_loss)
        history["soh"].append(epoch_loss["soh"])
        history["soc"].append(epoch_loss["soc"])
        history["soh_soc"].append(epoch_loss["soh_soc"])
        history["dr"].append(epoch_loss["dr"])

        if scheduler is not None:
            scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | "
            f"Loss={avg_loss:.6f} | "
            f"SOH={epoch_loss['soh']:.6f} | "
            f"SOC={epoch_loss['soc']:.6f} | "
            f"dQ={epoch_loss['soh_soc']:.6f} | "
            f"DR={epoch_loss['dr']:.6f} | "
            f"LR={current_lr:.6e}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "history": history,
                "dr_mode": dr_mode,
                "dr_target_scale": dr_target_scale,
            }

            if extra_state is not None:
                state.update(extra_state)

            with open(checkpoint_path, "wb") as f:
                torch.save(state, f)

            print(
                f"  Best model saved: {checkpoint_path} | "
                f"loss={best_loss:.6f}"
            )

        else:
            patience_counter += 1

            if early_stopping_patience is not None:
                print(
                    f"  No improvement: "
                    f"{patience_counter}/{early_stopping_patience}"
                )

                if patience_counter >= early_stopping_patience:
                    print("  Early stopping triggered.")
                    break
            else:
                print("  No improvement.")

    return history, best_loss
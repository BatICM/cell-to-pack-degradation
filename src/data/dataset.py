# -*- coding: utf-8 -*-
"""
Dataset and dataloader utilities for paired SOC/SOH samples.
"""

from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class BatteryDataset(Dataset):
    """Cycle-level paired SOH/SOC dataset."""

    def __init__(self, data_pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        x_soh, y_soh, x_soc, y_soc = self.data_pairs[idx]

        soh_tensors = []
        for soh_data, soh_label in zip(x_soh, y_soh):
            soh_tensors.append((
                torch.FloatTensor(soh_data),
                torch.FloatTensor(soh_label),
            ))

        soc_tensors = []
        for soc_data, soc_label in zip(x_soc, y_soc):
            soc_tensors.append((
                torch.FloatTensor(soc_data),
                torch.FloatTensor(soc_label),
            ))

        return soh_tensors, soc_tensors


def collate_batch(batch):
    """Original-style batch collation."""
    soh_data_lists = [item[0] for item in batch]
    soc_data_lists = [item[1] for item in batch]

    x_soh_batch = []
    y_soh_batch = []

    for soh_list in soh_data_lists:
        x_soh = torch.stack([item[0] for item in soh_list])
        y_soh = torch.stack([item[1] for item in soh_list])
        x_soh_batch.append(x_soh)
        y_soh_batch.append(y_soh)

    x_soh_batch = torch.stack(x_soh_batch)
    y_soh_batch = torch.stack(y_soh_batch)

    x_soc_batch = []
    y_soc_batch = []

    for soc_list in soc_data_lists:
        x_soc = torch.stack([item[0] for item in soc_list])
        y_soc = torch.stack([item[1] for item in soc_list])
        x_soc_batch.append(x_soc)
        y_soc_batch.append(y_soc)

    x_soc_batch = torch.stack(x_soc_batch)
    y_soc_batch = torch.stack(y_soc_batch)

    return (x_soh_batch, y_soh_batch), (x_soc_batch, y_soc_batch)


def build_train_test_pairs(
    processed_data_pairs,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Original-style normalization and cycle-level train/test split.
    """
    data_graph_soh_list = [pair[0] for pair in processed_data_pairs]
    lab_soh_list = [pair[1] for pair in processed_data_pairs]

    data_graph_soc_list = [pair[2] for pair in processed_data_pairs]
    lab_soc_list = [pair[3] for pair in processed_data_pairs]

    data_soh_flat = []
    lab_soh_flat = []
    sample_indices_soh = []

    for i, (soh_samples, soh_labs) in enumerate(zip(data_graph_soh_list, lab_soh_list)):
        data_soh_flat.extend(soh_samples)
        lab_soh_flat.extend(soh_labs)
        sample_indices_soh.extend([i] * len(soh_samples))

    data_array_soh = np.array(data_soh_flat)
    lab_array_soh = np.array(lab_soh_flat)
    sample_indices_soh = np.array(sample_indices_soh)

    data_graph_soc_flat = []
    lab_soc_flat = []
    sample_indices_soc = []

    for i, (soc_samples, soc_labs) in enumerate(zip(data_graph_soc_list, lab_soc_list)):
        data_graph_soc_flat.extend(soc_samples)
        lab_soc_flat.extend(soc_labs)
        sample_indices_soc.extend([i] * len(soc_samples))

    data_array_soc = np.array(data_graph_soc_flat)
    lab_array_soc = np.array(lab_soc_flat)
    sample_indices_soc = np.array(sample_indices_soc)

    # Keep exactly the same normalization as the original script.
    soh_norm_p1 = np.max(data_array_soh, axis=(0, 1))
    soh_norm_p2 = np.min(data_array_soh, axis=(0, 1))
    data_array_soh_max_min = soh_norm_p1 - soh_norm_p2

    normalized_data_soh = (
        data_array_soh - soh_norm_p2[np.newaxis, np.newaxis, :]
    ) / data_array_soh_max_min[np.newaxis, np.newaxis, :]

    print("normalized_data_soh shape:", normalized_data_soh.shape)
    print("data_array_soc shape:", data_array_soc.shape)

    soc_norm_p1 = np.max(data_array_soc, axis=(0, 1))
    soc_norm_p2 = np.min(data_array_soc, axis=(0, 1))
    data_array_soc_max_min = soc_norm_p1 - soc_norm_p2

    normalized_data_soc = (
        data_array_soc - soc_norm_p2[np.newaxis, np.newaxis, :]
    ) / data_array_soc_max_min[np.newaxis, np.newaxis, :]

    normalized_data_pairs = []

    for i in range(len(data_graph_soh_list)):
        soh_mask = sample_indices_soh == i
        soc_mask = sample_indices_soc == i

        data_graph_soh_samples = normalized_data_soh[soh_mask]
        lab_soh_samples = lab_array_soh[soh_mask]

        data_graph_soc_samples = normalized_data_soc[soc_mask]
        lab_soc_samples = lab_array_soc[soc_mask]

        normalized_data_pairs.append((
            data_graph_soh_samples,
            lab_soh_samples,
            data_graph_soc_samples,
            lab_soc_samples,
        ))

    indices = np.arange(len(normalized_data_pairs))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
    )

    train_pairs = [normalized_data_pairs[i] for i in train_indices]
    test_pairs = [normalized_data_pairs[i] for i in test_indices]

    norm_params = {
        "soh_max": soh_norm_p1,
        "soh_min": soh_norm_p2,
        "soc_max": soc_norm_p1,
        "soc_min": soc_norm_p2,
    }

    return train_pairs, test_pairs, norm_params


def create_dataloaders(
    train_pairs,
    test_pairs,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
):
    """Create train/test dataloaders using a fixed generator."""
    train_dataset = BatteryDataset(train_pairs)
    test_dataset = BatteryDataset(test_pairs)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        generator=generator,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
    )

    return train_loader, test_loader
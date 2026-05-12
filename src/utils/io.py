# -*- coding: utf-8 -*-

from pathlib import Path
import pickle

import yaml


def load_yaml(path):
    """Load a YAML configuration file."""
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def ensure_dir(path):
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_pickle(path):
    """Load a pickle file."""
    path = Path(path)

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


def save_pickle(obj, path):
    """Save an object as a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_npz(path, **arrays):
    """Save numpy arrays to an npz file."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(path, **arrays)
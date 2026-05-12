# -*- coding: utf-8 -*-

import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
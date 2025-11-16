# src/utils/logging.py

import os
import random
import numpy as np
import torch
import logging


def set_seed(seed: int) -> None:
    """
    Sets random seeds across Python, NumPy, and PyTorch
    to ensure reproducibility.

    Args:
        seed: integer seed
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cudnn deterministic (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensures hash-based ops are deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[utils.logging] Seed set to {seed}")


def get_logger(name: str = "rna-diffusion") -> logging.Logger:
    """
    Create a simple logger for experiments and training scripts.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

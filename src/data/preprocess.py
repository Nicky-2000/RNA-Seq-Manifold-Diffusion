# src/data/preprocess.py

import torch

def identity_preprocess(X: torch.Tensor) -> torch.Tensor:
    """
    For now, do nothing. 
    RNA data is assumed to already be preprocessed outside this codebase.
    Swiss roll is synthetic and does not need scaling for our early experiments.
    """
    return X
# src/data/__init__.py

from .loaders import load_dataset, load_swiss_roll, DatasetName
from .preprocess import identity_preprocess

__all__ = [
    "DatasetName",
    "load_dataset",
    "load_swiss_roll",
    "identity_preprocess",
]
